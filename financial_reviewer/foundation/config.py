"""Fail closed unless the reviewer is configured for local-only inference.

Why this file exists:
    Configuration is a security boundary: a convenient but unsafe URL, model,
    tracing flag, or fallback must be rejected before document text can leave a
    trusted code path.

What it owns:
    Strict loopback URL/model validation, cloud-tracing guards, immutable local
    model settings, and a small explicit environment-variable reader.

What it does not own:
    It does not construct requests, call Ollama, load ``.env`` files, or select
    a cloud provider.  This module deliberately uses :class:`pydantic.BaseModel`
    instead of a ``BaseSettings`` implementation so configuration is copied
    only from the documented allowlist.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Final, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# Ambient switches checked before construction and again before model egress.
# Any enabled value fails closed because callbacks could expose sensitive text.
TRACING_ENVIRONMENT_VARIABLES: Final[tuple[str, ...]] = (
    "LANGSMITH_TRACING",
    "LANGSMITH_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_HANDLER",
    "LANGCHAIN_DEBUG",
    "LANGCHAIN_VERBOSE",
)
"""Ambient tracing switches that are forbidden for the local workflow."""

# Narrow default model allowlist; additional local identifiers require an
# explicit configuration decision rather than silently accepting any model.
DEFAULT_ALLOWED_MODELS: Final[tuple[str, ...]] = ("qwen2.5:3b",)

# Values treated as definitely disabled when interpreting ambient switches.
# Unrecognized values are considered enabled, which preserves fail-closed behavior.
_EXPLICIT_FALSE_VALUES: Final[frozenset[str]] = frozenset(
    {"", "0", "false", "no", "off", "disabled"}
)
# Model identifiers must also fit the PII-safe audit ``SafeVersion`` envelope;
# rejecting wider names here prevents a valid runtime configuration from making
# the first mandatory audit record impossible to persist.
_MODEL_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+-]{0,63}$")
# Reject identifiers explicitly marked as cloud-hosted even if allowlisted.
_CLOUD_MODEL_MARKER = re.compile(r"(?:^|[._:/-])cloud(?:$|[._:/-])", re.IGNORECASE)
# Literal IP loopback hosts only; DNS names and remote origins are not accepted.
_LOOPBACK_HOSTS: Final[frozenset[str]] = frozenset(
    {"127.0.0.1", "::1"}
)


class UnsafeRuntimeConfigurationError(RuntimeError):
    """Raised without values when ambient configuration could disclose data."""


def _ambient_value_is_truthy(value: str | None) -> bool:
    """Treat every value except a small explicit-false set as enabled."""

    if value is None:
        return False
    return value.strip().casefold() not in _EXPLICIT_FALSE_VALUES


def ensure_cloud_tracing_disabled(
    environ: Mapping[str, str] | None = None,
) -> None:
    """Fail closed if ambient or programmatic LangSmith tracing is enabled.

    Values are intentionally omitted from the exception so a configuration
    secret cannot be copied into a log or audit record.
    """

    source = os.environ if environ is None else environ
    enabled_names = [
        name
        for name in TRACING_ENVIRONMENT_VARIABLES
        if _ambient_value_is_truthy(source.get(name))
    ]
    if enabled_names:
        names = ", ".join(enabled_names)
        raise UnsafeRuntimeConfigurationError(
            "Cloud tracing is forbidden for the financial document reviewer; "
            f"disable: {names}."
        )

    # LangGraph installs LangSmith as a transitive dependency.  Its
    # ``tracing_context(enabled=True)`` API can activate tracing without an
    # environment variable, so environment checks alone are insufficient.
    # Importing this read-only helper does not construct a client or perform
    # I/O; any active parent/client/replica context is rejected before raw
    # workflow state can enter the graph.
    try:
        from langchain_core.globals import get_debug, get_verbose
        from langchain_core.tracers.context import (
            _configure_hooks,
            tracing_v2_callback_var,
        )
        from langsmith.run_helpers import get_tracing_context
        from langsmith.utils import tracing_is_enabled
    except (ImportError, AttributeError):
        raise UnsafeRuntimeConfigurationError(
            "The tracing safety guard is unavailable."
        ) from None

    try:
        context = get_tracing_context()
        context_is_active = context.get("enabled") not in (None, False) or any(
            context.get(name) is not None
            for name in ("parent", "client", "replicas", "distributed_parent_id")
        )
        callback_is_active = tracing_v2_callback_var.get() is not None
        hook_is_active = any(
            context_variable.get() is not None
            for context_variable, _inheritable, _handler_class, _environment_name
            in _configure_hooks
        )
        tracing_active = tracing_is_enabled() not in (None, False)
        unsafe_console_mode = get_debug() or get_verbose()
    except Exception:
        raise UnsafeRuntimeConfigurationError(
            "The tracing safety guard could not verify the runtime."
        ) from None
    if (
        context_is_active
        or tracing_active
        or callback_is_active
        or hook_is_active
        or unsafe_console_mode
    ):
        raise UnsafeRuntimeConfigurationError(
            "Programmatic tracing and debug callbacks are forbidden for the "
            "financial document reviewer."
        )


class LocalModelSettings(BaseModel):
    """Validated, deterministic settings for the loopback-only Ollama client."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        revalidate_instances="always",
    )

    provider: Literal["ollama"] = "ollama"
    base_url: str = "http://127.0.0.1:11434"
    model: str = DEFAULT_ALLOWED_MODELS[0]
    allowed_models: tuple[str, ...] = DEFAULT_ALLOWED_MODELS
    allow_cloud_fallback: Literal[False] = False

    # Deterministic generation controls.  Literal values prevent callers from
    # silently weakening determinism for this milestone.
    temperature: Literal[0.0] = 0.0
    seed: int = Field(default=0, ge=0, le=2_147_483_647)
    num_predict: int = Field(default=2_048, ge=1, le=8_192)

    # Resource bounds protect both the model process and the caller.  They are
    # configurable within conservative hard ceilings for local deployments.
    connect_timeout_seconds: float = Field(default=5.0, gt=0.0, le=30.0)
    request_timeout_seconds: float = Field(default=90.0, gt=0.0, le=300.0)
    max_document_chars: int = Field(default=250_000, ge=1, le=1_000_000)
    max_schema_bytes: int = Field(default=64_000, ge=1, le=1_000_000)
    max_response_bytes: int = Field(default=1_000_000, ge=1, le=5_000_000)
    max_output_chars: int = Field(default=500_000, ge=1, le=1_000_000)

    @field_validator("base_url")
    @classmethod
    def validate_loopback_base_url(cls, value: str) -> str:
        """Accept only a bare HTTP origin on an explicitly allowed loopback host."""

        if not value or any(character.isspace() for character in value):
            raise ValueError("Ollama base URL must be a loopback HTTP origin")
        # urlsplit cannot distinguish an empty query/fragment from no delimiter.
        if "?" in value or "#" in value:
            raise ValueError("Ollama base URL cannot contain a query or fragment")

        parsed = urlsplit(value)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("Ollama base URL contains an invalid port") from exc

        if parsed.scheme.casefold() != "http":
            raise ValueError("Ollama base URL must use HTTP on loopback")
        if parsed.hostname is None or parsed.hostname.casefold() not in _LOOPBACK_HOSTS:
            raise ValueError("Ollama base URL must use an approved loopback host")
        if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
            raise ValueError("Ollama base URL cannot contain credentials")
        if parsed.path not in ("", "/"):
            raise ValueError("Ollama base URL must not contain a path")
        if port is not None and not 1 <= port <= 65_535:
            raise ValueError("Ollama base URL contains an invalid port")

        return value.rstrip("/")

    @field_validator("allowed_models")
    @classmethod
    def validate_allowed_models(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Require a small, unique allowlist containing no cloud identifiers."""

        if not value or len(value) > 32:
            raise ValueError("The local model allowlist must contain 1 to 32 entries")
        if len(set(value)) != len(value):
            raise ValueError("The local model allowlist cannot contain duplicates")
        if any(not _MODEL_IDENTIFIER.fullmatch(item) for item in value):
            raise ValueError("The local model allowlist contains an invalid identifier")
        if any(_CLOUD_MODEL_MARKER.search(item) for item in value):
            raise ValueError("Ollama cloud-model identifiers are forbidden")
        return value

    @field_validator("model")
    @classmethod
    def validate_model_identifier(cls, value: str) -> str:
        """Reject malformed or explicitly cloud-routed model identifiers."""

        if not _MODEL_IDENTIFIER.fullmatch(value):
            raise ValueError("The local model identifier is invalid")
        if _CLOUD_MODEL_MARKER.search(value):
            raise ValueError("Ollama cloud-model identifiers are forbidden")
        return value

    @model_validator(mode="after")
    def validate_security_invariants(self) -> Self:
        """Recheck tracing and require the selected model to be allowlisted."""

        # Re-check the process environment on every construction, even when a
        # caller supplies all model fields explicitly.
        ensure_cloud_tracing_disabled()
        if self.model not in self.allowed_models:
            raise ValueError("The selected model is not in the local model allowlist")
        return self

    @property
    def endpoint_url(self) -> str:
        """The only inference route the adapter is permitted to call."""

        return f"{self.base_url}/api/generate"

    @property
    def model_name(self) -> str:
        """Compatibility/readability alias for the selected allowlisted model."""

        return self.model

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> Self:
        """Build settings from an explicit allowlist of process variables.

        The method never calls ``load_dotenv`` and never reads a ``.env`` file.
        Unlisted ambient variables, including proxy and cloud-provider settings,
        are ignored.
        """

        source = os.environ if environ is None else environ
        ensure_cloud_tracing_disabled(source)
        field_names = {
            "FINANCIAL_REVIEWER_MODEL_PROVIDER": "provider",
            "FINANCIAL_REVIEWER_OLLAMA_BASE_URL": "base_url",
            "FINANCIAL_REVIEWER_OLLAMA_MODEL": "model",
            "FINANCIAL_REVIEWER_ALLOW_CLOUD_FALLBACK": "allow_cloud_fallback",
            "FINANCIAL_REVIEWER_MODEL_SEED": "seed",
            "FINANCIAL_REVIEWER_MODEL_NUM_PREDICT": "num_predict",
            "FINANCIAL_REVIEWER_CONNECT_TIMEOUT_SECONDS": "connect_timeout_seconds",
            "FINANCIAL_REVIEWER_REQUEST_TIMEOUT_SECONDS": "request_timeout_seconds",
            "FINANCIAL_REVIEWER_MAX_DOCUMENT_CHARS": "max_document_chars",
            "FINANCIAL_REVIEWER_MAX_SCHEMA_BYTES": "max_schema_bytes",
            "FINANCIAL_REVIEWER_MAX_RESPONSE_BYTES": "max_response_bytes",
            "FINANCIAL_REVIEWER_MAX_OUTPUT_CHARS": "max_output_chars",
        }
        values: dict[str, object] = {
            field_name: source[environment_name]
            for environment_name, field_name in field_names.items()
            if environment_name in source
            and environment_name != "FINANCIAL_REVIEWER_ALLOW_CLOUD_FALLBACK"
        }

        fallback_value = source.get("FINANCIAL_REVIEWER_ALLOW_CLOUD_FALLBACK")
        if fallback_value is not None:
            # Literal[False] remains the final enforcement point; the explicit
            # conversion only makes conventional textual false values usable.
            values["allow_cloud_fallback"] = (
                False if not _ambient_value_is_truthy(fallback_value) else True
            )

        allowlist_value = source.get("FINANCIAL_REVIEWER_OLLAMA_ALLOWED_MODELS")
        if allowlist_value is not None:
            values["allowed_models"] = tuple(
                item.strip() for item in allowlist_value.split(",") if item.strip()
            )

        return cls.model_validate(values)


# Deliberately narrow aliases for callers that describe this object by its role.
LocalOnlyModelConfig = LocalModelSettings
__all__ = [
    "DEFAULT_ALLOWED_MODELS",
    "LocalModelSettings",
    "LocalOnlyModelConfig",
    "TRACING_ENVIRONMENT_VARIABLES",
    "UnsafeRuntimeConfigurationError",
    "ensure_cloud_tracing_disabled",
]
