"""Export sanitized review spans to a loopback OTLP/HTTP receiver.

Why this file exists:
    Keep Traceboard and other OTLP-compatible local backends behind the
    vendor-neutral ``ReviewTraceSink`` boundary. The workflow knows only the
    closed span contract; this adapter owns protocol encoding and transport.

What it owns:
    Strict loopback endpoint validation, deterministic OTLP protobuf mapping,
    an explicit four-variable OTEL configuration allowlist, bounded HTTP
    transport, response validation, and sanitized exceptions.

What it excludes:
    It performs no auto-instrumentation and has no prompt, document, financial
    value, model-output, exception-message, arbitrary-attribute, retry, cloud,
    or fallback path. Constructing this sink is the explicit opt-in; the
    reviewer still uses ``NoOpReviewTraceSink`` by default.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import AbstractContextManager
from datetime import datetime
from typing import Any, Literal, Self
from urllib.parse import urlsplit

import httpx
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.trace.v1.trace_pb2 import Span, Status
from pydantic import BaseModel, ConfigDict, Field, field_validator

from financial_reviewer.foundation.config import (
    UnsafeRuntimeConfigurationError,
    ensure_cloud_tracing_disabled,
)
from financial_reviewer.local.telemetry import (
    ReviewTraceExportError,
    SanitizedTraceSpan,
    TraceSpanOutcome,
)


_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1"})
_SERVICE_NAME = "multi-agent-financial-document-reviewer"
_PROJECT_ID = "multi-agent-financial-document-reviewer"
_TRACE_SCHEMA_VERSION = "1.0"
_OTLP_PROTOCOL = "http/protobuf"
_RESOURCE_ATTRIBUTES = f"traceboard.project.id={_PROJECT_ID}"


class LocalOtlpTraceError(ReviewTraceExportError):
    """Base exception that never carries endpoint, payload, or response text."""


class LocalOtlpTraceUnavailableError(LocalOtlpTraceError):
    """The approved loopback receiver could not accept a sanitized span."""


class LocalOtlpTraceResponseError(LocalOtlpTraceError):
    """The loopback receiver returned an invalid or oversized response."""


class LocalOtlpTraceSettings(BaseModel):
    """Immutable configuration for one explicit local OTLP trace destination."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        str_strip_whitespace=True,
        revalidate_instances="always",
    )

    endpoint: str = "http://127.0.0.1:4318/v1/traces"
    service_name: Literal["multi-agent-financial-document-reviewer"] = _SERVICE_NAME
    project_id: Literal["multi-agent-financial-document-reviewer"] = _PROJECT_ID
    connect_timeout_seconds: float = Field(default=1.0, gt=0.0, le=10.0)
    request_timeout_seconds: float = Field(default=5.0, gt=0.0, le=30.0)
    max_request_bytes: int = Field(default=262_144, ge=1, le=1_048_576)
    max_response_bytes: int = Field(default=65_536, ge=0, le=262_144)

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> Self:
        """Read only the four approved Traceboard OTEL configuration values.

        This is explicit exporter configuration, not OpenTelemetry
        auto-instrumentation. Missing, remote, protocol-changing, or arbitrary
        resource settings fail before the sink can construct a transport.
        """

        source = os.environ if environ is None else environ
        required = (
            "OTEL_EXPORTER_OTLP_ENDPOINT",
            "OTEL_EXPORTER_OTLP_PROTOCOL",
            "OTEL_SERVICE_NAME",
            "OTEL_RESOURCE_ATTRIBUTES",
        )
        if any(name not in source for name in required):
            raise UnsafeRuntimeConfigurationError(
                "The required local OTEL trace configuration is incomplete."
            )
        base_endpoint = source["OTEL_EXPORTER_OTLP_ENDPOINT"]
        try:
            parsed = urlsplit(base_endpoint)
            port = parsed.port
        except (TypeError, ValueError):
            raise UnsafeRuntimeConfigurationError(
                "The local OTEL trace configuration failed security validation."
            ) from None
        if (
            not base_endpoint
            or any(character.isspace() for character in base_endpoint)
            or "?" in base_endpoint
            or "#" in base_endpoint
            or parsed.scheme.casefold() != "http"
            or parsed.hostname is None
            or parsed.hostname.casefold() not in _LOOPBACK_HOSTS
            or parsed.username is not None
            or parsed.password is not None
            or "@" in parsed.netloc
            or port is None
            or not 1 <= port <= 65_535
            or parsed.path not in ("", "/")
            or source["OTEL_EXPORTER_OTLP_PROTOCOL"] != _OTLP_PROTOCOL
            or source["OTEL_SERVICE_NAME"] != _SERVICE_NAME
            or source["OTEL_RESOURCE_ATTRIBUTES"] != _RESOURCE_ATTRIBUTES
        ):
            raise UnsafeRuntimeConfigurationError(
                "The local OTEL trace configuration failed security validation."
            )
        try:
            return cls(
                endpoint=f"{base_endpoint.rstrip('/')}/v1/traces",
                service_name=_SERVICE_NAME,
                project_id=_PROJECT_ID,
            )
        except (TypeError, ValueError):
            raise UnsafeRuntimeConfigurationError(
                "The local OTEL trace configuration failed security validation."
            ) from None

    @field_validator("endpoint")
    @classmethod
    def validate_loopback_trace_endpoint(cls, value: str) -> str:
        """Accept only an exact OTLP trace path on a literal loopback host."""

        if not value or any(character.isspace() for character in value):
            raise ValueError("OTLP trace endpoint must be loopback HTTP")
        if "?" in value or "#" in value:
            raise ValueError("OTLP trace endpoint cannot contain query or fragment")
        parsed = urlsplit(value)
        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError("OTLP trace endpoint contains an invalid port") from exc
        if parsed.scheme.casefold() != "http":
            raise ValueError("OTLP trace endpoint must use loopback HTTP")
        if parsed.hostname is None or parsed.hostname.casefold() not in _LOOPBACK_HOSTS:
            raise ValueError("OTLP trace endpoint must use a literal loopback host")
        if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
            raise ValueError("OTLP trace endpoint cannot contain credentials")
        if port is None or not 1 <= port <= 65_535:
            raise ValueError("OTLP trace endpoint requires a valid explicit port")
        if parsed.path != "/v1/traces":
            raise ValueError("OTLP trace endpoint must end at /v1/traces")
        return value


class LocalOtlpTraceSink:
    """Map each safe completed span to one bounded loopback OTLP request."""

    __slots__ = ("_sealed", "_settings")

    def __init__(self, settings: LocalOtlpTraceSettings | None = None) -> None:
        candidate = settings or LocalOtlpTraceSettings()
        try:
            validated = LocalOtlpTraceSettings.model_validate(
                candidate.model_dump(mode="python", warnings="none")
            )
        except (AttributeError, TypeError, ValueError):
            raise UnsafeRuntimeConfigurationError(
                "The local OTLP trace settings failed security validation."
            ) from None
        object.__setattr__(self, "_settings", validated)
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent endpoint mutation after the loopback policy is validated."""

        if getattr(self, "_sealed", False):
            raise AttributeError("The local OTLP trace adapter is immutable.")
        object.__setattr__(self, name, value)

    @property
    def settings(self) -> LocalOtlpTraceSettings:
        """Return the immutable validated configuration snapshot."""

        return self._settings

    def emit_span(self, span: SanitizedTraceSpan) -> None:
        """Validate, encode, and send one span with no retry or fallback."""

        ensure_cloud_tracing_disabled()
        settings = self._validated_settings_snapshot()
        safe_span = self._revalidate_span(span)
        payload = self._encode_span(safe_span, settings=settings)
        if len(payload) > settings.max_request_bytes:
            raise LocalOtlpTraceResponseError(
                "The sanitized OTLP trace request exceeded its configured limit."
            )
        try:
            with self._client_context(settings) as client:
                self._post_and_verify(client, payload, settings=settings)
        except LocalOtlpTraceError:
            raise
        except httpx.HTTPError:
            raise LocalOtlpTraceUnavailableError(
                "The local OTLP trace receiver could not complete the request."
            ) from None

    def _validated_settings_snapshot(self) -> LocalOtlpTraceSettings:
        """Revalidate immutable settings immediately before local egress."""

        try:
            return LocalOtlpTraceSettings.model_validate(
                self._settings.model_dump(mode="python", warnings="none")
            )
        except (AttributeError, TypeError, ValueError):
            raise UnsafeRuntimeConfigurationError(
                "The local OTLP trace runtime failed security validation."
            ) from None

    @staticmethod
    def _revalidate_span(span: object) -> SanitizedTraceSpan:
        """Reject constructed or wrong-type spans before serialization."""

        if not isinstance(span, SanitizedTraceSpan):
            raise TypeError("OTLP trace sinks accept only SanitizedTraceSpan")
        try:
            return SanitizedTraceSpan.model_validate(
                span.model_dump(mode="python", warnings="none")
            )
        except (AttributeError, TypeError, ValueError):
            raise TypeError("The sanitized OTLP trace span is invalid") from None

    @staticmethod
    def _encode_span(
        span: SanitizedTraceSpan,
        *,
        settings: LocalOtlpTraceSettings,
    ) -> bytes:
        """Create a minimal OTLP request from the closed span schema."""

        request = ExportTraceServiceRequest()
        resource_spans = request.resource_spans.add()
        resource_spans.resource.attributes.extend(
            (
                _string_attribute("service.name", settings.service_name),
                _string_attribute("traceboard.project.id", settings.project_id),
                _string_attribute("deployment.environment.name", "local"),
            )
        )
        scope_spans = resource_spans.scope_spans.add()
        scope_spans.scope.name = "financial_reviewer.local.otlp_trace"
        scope_spans.scope.version = _TRACE_SCHEMA_VERSION

        otlp_span = scope_spans.spans.add()
        otlp_span.trace_id = bytes.fromhex(span.trace_id)
        otlp_span.span_id = bytes.fromhex(span.span_id)
        if span.parent_span_id is not None:
            otlp_span.parent_span_id = bytes.fromhex(span.parent_span_id)
        otlp_span.name = span.stage.value
        otlp_span.kind = Span.SPAN_KIND_INTERNAL
        otlp_span.start_time_unix_nano = _datetime_to_unix_nanos(span.started_at)
        otlp_span.end_time_unix_nano = _datetime_to_unix_nanos(span.ended_at)
        otlp_span.status.code = (
            Status.STATUS_CODE_ERROR
            if span.outcome is TraceSpanOutcome.FAILED
            else Status.STATUS_CODE_OK
        )
        otlp_span.attributes.extend(
            (
                _string_attribute("review.stage", span.stage.value),
                _string_attribute("review.outcome", span.outcome.value),
                _string_attribute("review.schema.version", _TRACE_SCHEMA_VERSION),
                _integer_attribute("review.cause.count", len(span.causes)),
            )
        )
        if span.document_type is not None:
            # Traceboard intentionally filters attribute names containing
            # "document". The value here is only a closed classification, so
            # use a semantically accurate non-payload key.
            otlp_span.attributes.append(
                _string_attribute("review.classification", span.document_type)
            )
        if span.causes:
            otlp_span.attributes.extend(
                (
                    _string_array_attribute(
                        "review.cause.origin_stages",
                        tuple(cause.origin_stage.value for cause in span.causes),
                    ),
                    _string_array_attribute(
                        "review.cause.kinds",
                        tuple(cause.kind.value for cause in span.causes),
                    ),
                    _string_array_attribute(
                        "review.cause.reason_codes",
                        tuple(str(cause.reason_code) for cause in span.causes),
                    ),
                )
            )
        return request.SerializeToString()

    @staticmethod
    def _client_context(
        settings: LocalOtlpTraceSettings,
    ) -> AbstractContextManager[httpx.Client]:
        """Construct a one-connection client that ignores proxy environment."""

        timeout = httpx.Timeout(
            timeout=settings.request_timeout_seconds,
            connect=settings.connect_timeout_seconds,
        )
        return httpx.Client(
            timeout=timeout,
            trust_env=False,
            follow_redirects=False,
            limits=httpx.Limits(max_connections=1, max_keepalive_connections=0),
        )

    @staticmethod
    def _post_and_verify(
        client: httpx.Client,
        payload: bytes,
        *,
        settings: LocalOtlpTraceSettings,
    ) -> None:
        """Post one protobuf body and validate a bounded OTLP response."""

        try:
            with client.stream(
                "POST",
                settings.endpoint,
                content=payload,
                headers={
                    "Content-Type": "application/x-protobuf",
                    "Accept": "application/x-protobuf",
                },
            ) as response:
                if response.status_code < 200 or response.status_code >= 300:
                    raise LocalOtlpTraceUnavailableError(
                        "The local OTLP trace receiver rejected the request "
                        f"(HTTP {response.status_code})."
                    )
                declared = response.headers.get("content-length")
                if declared is not None:
                    try:
                        declared_size = int(declared)
                    except ValueError:
                        raise LocalOtlpTraceResponseError(
                            "The local OTLP receiver returned an invalid response."
                        ) from None
                    if declared_size < 0 or declared_size > settings.max_response_bytes:
                        raise LocalOtlpTraceResponseError(
                            "The local OTLP response exceeded its configured limit."
                        )
                response_body = bytearray()
                for chunk in response.iter_bytes():
                    if len(response_body) + len(chunk) > settings.max_response_bytes:
                        raise LocalOtlpTraceResponseError(
                            "The local OTLP response exceeded its configured limit."
                        )
                    response_body.extend(chunk)
        except LocalOtlpTraceError:
            raise
        except httpx.HTTPError:
            raise LocalOtlpTraceUnavailableError(
                "The local OTLP trace receiver could not complete the request."
            ) from None

        if response_body:
            try:
                ExportTraceServiceResponse.FromString(bytes(response_body))
            except Exception:
                raise LocalOtlpTraceResponseError(
                    "The local OTLP receiver returned an invalid response."
                ) from None


def _datetime_to_unix_nanos(value: datetime) -> int:
    """Convert an aware UTC-normalized datetime without float rounding."""

    seconds = int(value.timestamp())
    return seconds * 1_000_000_000 + value.microsecond * 1_000


def _string_attribute(key: str, value: str) -> KeyValue:
    return KeyValue(key=key, value=AnyValue(string_value=value))


def _integer_attribute(key: str, value: int) -> KeyValue:
    return KeyValue(key=key, value=AnyValue(int_value=value))


def _string_array_attribute(key: str, values: tuple[str, ...]) -> KeyValue:
    attribute = KeyValue(key=key)
    attribute.value.array_value.values.extend(
        AnyValue(string_value=value) for value in values
    )
    return attribute


__all__ = [
    "LocalOtlpTraceError",
    "LocalOtlpTraceResponseError",
    "LocalOtlpTraceSettings",
    "LocalOtlpTraceSink",
    "LocalOtlpTraceUnavailableError",
]
