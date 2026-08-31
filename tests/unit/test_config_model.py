from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.globals import set_debug, set_verbose
from langchain_core.tracers.context import collect_runs, tracing_v2_callback_var
from langsmith import tracing_context
from pydantic import ValidationError

from financial_reviewer.foundation.config import (
    LocalModelSettings,
    UnsafeRuntimeConfigurationError,
    ensure_cloud_tracing_disabled,
)
from financial_reviewer.local.model import (
    LocalModelInputError,
    LocalModelResponseError,
    LocalModelUnavailableError,
    OllamaModel,
)


def _install_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
    transport: httpx.MockTransport,
) -> None:
    real_client = httpx.Client

    def client_context(_model: OllamaModel, _settings: object) -> httpx.Client:
        return real_client(
            transport=transport,
            trust_env=False,
            follow_redirects=False,
        )

    monkeypatch.setattr(OllamaModel, "_client_context", client_context)


@pytest.mark.parametrize(
    "base_url",
    [
        "https://127.0.0.1:11434",
        "http://192.168.1.50:11434",
        "http://ollama.example.com",
        "http://localhost:11434",
        "http://user:secret@127.0.0.1:11434",
        "http://127.0.0.1:11434/v1",
        "http://127.0.0.1:11434?target=cloud",
        "http://127.0.0.1:11434#fragment",
    ],
)
def test_local_model_settings_reject_non_loopback_routes(base_url: str) -> None:
    with pytest.raises(ValidationError):
        LocalModelSettings(base_url=base_url)


def test_local_model_settings_accept_explicit_loopback_origins() -> None:
    for origin in (
        "http://127.0.0.1:11434",
        "http://[::1]:11434",
    ):
        assert LocalModelSettings(base_url=origin).endpoint_url.endswith("/api/generate")


def test_settings_read_only_financial_reviewer_environment_names() -> None:
    settings = LocalModelSettings.from_environment(
        {
            "FINANCIAL_REVIEWER_MODEL_PROVIDER": "ollama",
            "FINANCIAL_REVIEWER_OLLAMA_BASE_URL": "http://[::1]:11435",
            "FINANCIAL_REVIEWER_OLLAMA_MODEL": "qwen3.5:9b",
            "FINANCIAL_REVIEWER_OLLAMA_ALLOWED_MODELS": (
                "qwen2.5:3b,qwen3.5:9b"
            ),
            "FINANCIAL_REVIEWER_ALLOW_CLOUD_FALLBACK": "false",
            "FINANCIAL_REVIEWER_MODEL_SEED": "17",
            "FINANCIAL_REVIEWER_MODEL_NUM_PREDICT": "1024",
            "FINANCIAL_REVIEWER_CONNECT_TIMEOUT_SECONDS": "4",
            "FINANCIAL_REVIEWER_REQUEST_TIMEOUT_SECONDS": "60",
            "FINANCIAL_REVIEWER_MAX_DOCUMENT_CHARS": "200000",
            "FINANCIAL_REVIEWER_MAX_SCHEMA_BYTES": "32000",
            "FINANCIAL_REVIEWER_MAX_RESPONSE_BYTES": "500000",
            "FINANCIAL_REVIEWER_MAX_OUTPUT_CHARS": "250000",
            # Unlisted variables cannot alter the validated settings snapshot.
            "UNLISTED_OLLAMA_BASE_URL": "https://cloud.example.invalid",
        }
    )

    assert settings.provider == "ollama"
    assert settings.base_url == "http://[::1]:11435"
    assert settings.model == "qwen3.5:9b"
    assert settings.allowed_models == ("qwen2.5:3b", "qwen3.5:9b")
    assert settings.allow_cloud_fallback is False
    assert settings.seed == 17
    assert settings.num_predict == 1024
    assert settings.connect_timeout_seconds == 4
    assert settings.request_timeout_seconds == 60
    assert settings.max_document_chars == 200_000
    assert settings.max_schema_bytes == 32_000
    assert settings.max_response_bytes == 500_000
    assert settings.max_output_chars == 250_000


def test_environment_cannot_enable_cloud_fallback() -> None:
    with pytest.raises(ValidationError):
        LocalModelSettings.from_environment(
            {"FINANCIAL_REVIEWER_ALLOW_CLOUD_FALLBACK": "true"}
        )


def test_cloud_fallback_and_unallowlisted_models_are_impossible() -> None:
    with pytest.raises(ValidationError):
        LocalModelSettings(allow_cloud_fallback=True)
    with pytest.raises(ValidationError):
        LocalModelSettings(model="remote-provider/model")
    with pytest.raises(ValidationError):
        LocalModelSettings(temperature=0.2)
    for cloud_model in ("gpt-oss:120b-cloud", "glm-4.7:cloud"):
        with pytest.raises(ValidationError):
            LocalModelSettings(
                model=cloud_model,
                allowed_models=(cloud_model,),
            )


@pytest.mark.parametrize(
    "model_name",
    [
        "local/team-model:1",
        "m" * 65,
    ],
)
def test_model_names_must_fit_the_safe_audit_version_contract(
    model_name: str,
) -> None:
    with pytest.raises(ValidationError):
        LocalModelSettings(model=model_name, allowed_models=(model_name,))


def test_truthy_cloud_tracing_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    with pytest.raises(UnsafeRuntimeConfigurationError):
        ensure_cloud_tracing_disabled()
    with pytest.raises(UnsafeRuntimeConfigurationError):
        LocalModelSettings()


def test_programmatic_langsmith_tracing_fails_closed() -> None:
    with tracing_context(enabled=True):
        with pytest.raises(UnsafeRuntimeConfigurationError):
            ensure_cloud_tracing_disabled()
        with pytest.raises(UnsafeRuntimeConfigurationError):
            LocalModelSettings()


@pytest.mark.parametrize("setter", [set_debug, set_verbose])
def test_langchain_console_modes_fail_closed(setter) -> None:
    setter(True)
    try:
        with pytest.raises(UnsafeRuntimeConfigurationError):
            ensure_cloud_tracing_disabled()
    finally:
        setter(False)


def test_langchain_callback_and_collector_contexts_fail_closed() -> None:
    token = tracing_v2_callback_var.set(object())  # type: ignore[arg-type]
    try:
        with pytest.raises(UnsafeRuntimeConfigurationError):
            ensure_cloud_tracing_disabled()
    finally:
        tracing_v2_callback_var.reset(token)

    with collect_runs():
        with pytest.raises(UnsafeRuntimeConfigurationError):
            ensure_cloud_tracing_disabled()


def test_constructed_remote_settings_are_revalidated_by_model() -> None:
    unsafe = LocalModelSettings.model_construct(
        base_url="https://cloud.example.invalid",
    )
    with pytest.raises(UnsafeRuntimeConfigurationError):
        OllamaModel(unsafe)


def test_ollama_adapter_uses_schema_and_disables_proxy_inheritance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_factory: dict[str, object] = {}
    captured_payload: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5:3b",
                "done": True,
                "response": '{"document_type":"pay_stub"}',
            },
        )

    transport = httpx.MockTransport(handler)

    real_client = httpx.Client

    def client_factory(**kwargs: object) -> httpx.Client:
        captured_factory.update(kwargs)
        kwargs["transport"] = transport
        return real_client(**kwargs)

    monkeypatch.setattr(httpx, "Client", client_factory)

    schema = {"type": "object", "additionalProperties": False}
    model = OllamaModel(LocalModelSettings())
    assert model.extract(
        "pay_stub",
        "synthetic local document",
        schema,
    ).startswith("{")
    assert captured_factory["trust_env"] is False
    assert captured_factory["follow_redirects"] is False
    assert captured_payload["format"] == schema
    assert captured_payload["model"] == "qwen2.5:3b"
    assert captured_payload["stream"] is False
    assert captured_payload["options"] == {
        "temperature": 0.0,
        "seed": 0,
        "num_predict": 2048,
    }


def test_generic_structured_call_uses_exact_caller_prompt_and_same_local_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_payload: dict[str, object] = {}
    prompt = "SAFE-AGENT2-DECISION-PROMPT"
    schema = {
        "type": "object",
        "properties": {"action": {"type": "string"}},
        "required": ["action"],
        "additionalProperties": False,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        captured_payload.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "qwen2.5:3b",
                "done": True,
                "response": '{"action":"complete"}',
            },
        )

    _install_mock_transport(monkeypatch, httpx.MockTransport(handler))

    output = OllamaModel().generate_structured(
        "agent2_income_verification",
        prompt,
        schema,
    )

    assert output == '{"action":"complete"}'
    assert captured_payload["prompt"] == prompt
    assert captured_payload["format"] == schema
    assert captured_payload["stream"] is False
    assert captured_payload["think"] is False


def test_invalid_generic_prompt_makes_zero_transport_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"model": "qwen2.5:3b", "done": True, "response": "{}"},
        )

    _install_mock_transport(monkeypatch, httpx.MockTransport(handler))

    with pytest.raises(LocalModelInputError):
        OllamaModel().generate_structured(
            "agent2_income_verification",
            "",
            {"type": "object"},
        )

    assert calls == 0


@pytest.mark.parametrize(
    "envelope",
    [
        {"model": "qwen2.5:3b", "done": False, "response": "{}"},
        {"model": "qwen2.5:3b", "response": "{}"},
        {"model": "unexpected:latest", "done": True, "response": "{}"},
        {"done": True, "response": "{}"},
        {
            "model": "qwen2.5:3b",
            "done": True,
            "response": "{}",
            "error": "synthetic local error",
        },
    ],
)
def test_adapter_rejects_nonterminal_or_wrong_model_envelopes(
    monkeypatch: pytest.MonkeyPatch,
    envelope: dict[str, object],
) -> None:
    _install_mock_transport(
        monkeypatch,
        httpx.MockTransport(lambda request: httpx.Response(200, json=envelope)),
    )
    with pytest.raises(LocalModelResponseError):
        OllamaModel().extract("pay_stub", "synthetic", {"type": "object"})


def test_adapter_rejects_duplicate_response_envelope_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = (
        b'{"model":"qwen2.5:3b","done":true,'
        b'"response":"{}","response":"altered"}'
    )
    _install_mock_transport(
        monkeypatch,
        httpx.MockTransport(lambda request: httpx.Response(200, content=raw)),
    )
    with pytest.raises(LocalModelResponseError):
        OllamaModel().extract("pay_stub", "synthetic", {"type": "object"})


def test_one_validated_settings_snapshot_drives_the_whole_exchange(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = LocalModelSettings()
    alternate = LocalModelSettings(
        model="qwen2.5:7b",
        allowed_models=("qwen2.5:7b",),
    )
    captured: dict[str, object] = {}
    real_client = httpx.Client

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"model": original.model, "done": True, "response": "{}"},
        )

    def mutate_after_validation(
        model: OllamaModel,
        settings_snapshot: LocalModelSettings,
    ) -> httpx.Client:
        assert settings_snapshot == original
        object.__setattr__(model, "_settings", alternate)
        return real_client(
            transport=httpx.MockTransport(handler),
            trust_env=False,
            follow_redirects=False,
        )

    monkeypatch.setattr(OllamaModel, "_client_context", mutate_after_validation)
    model = OllamaModel(original)
    assert model.extract(
        "pay_stub",
        "synthetic",
        {"type": "object"},
        _expected_settings=original,
    ) == "{}"
    assert captured["url"] == original.endpoint_url
    assert captured["payload"]["model"] == original.model  # type: ignore[index]


def test_invalid_model_input_makes_zero_transport_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            json={"model": "qwen2.5:3b", "done": True, "response": "{}"},
        )

    _install_mock_transport(monkeypatch, httpx.MockTransport(handler))
    model = OllamaModel()
    with pytest.raises(LocalModelInputError):
        model.extract("pay_stub", "", {"type": "object"})
    assert calls == 0


def test_adapter_never_follows_redirect_or_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            302,
            headers={"location": "https://cloud.example.invalid/inference"},
        )

    _install_mock_transport(monkeypatch, httpx.MockTransport(handler))
    model = OllamaModel()
    with pytest.raises(LocalModelUnavailableError):
        model.extract("pay_stub", "synthetic", {"type": "object"})
    assert calls == 1


def test_model_response_is_bounded_and_errors_do_not_echo_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "MODEL-OUTPUT-PRIVATE-SENTINEL"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=(sentinel * 4).encode("utf-8"))

    settings = LocalModelSettings(max_response_bytes=32)
    _install_mock_transport(monkeypatch, httpx.MockTransport(handler))
    model = OllamaModel(settings)
    with pytest.raises(LocalModelResponseError) as captured:
        model.extract("pay_stub", "synthetic", {"type": "object"})
    assert sentinel not in str(captured.value)
