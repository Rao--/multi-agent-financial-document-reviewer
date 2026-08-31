from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Iterator

import httpx
import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
    ExportTraceServiceResponse,
)
from pydantic import ValidationError

from financial_reviewer.foundation.config import UnsafeRuntimeConfigurationError
from financial_reviewer.local.otlp_trace import (
    LocalOtlpTraceResponseError,
    LocalOtlpTraceSettings,
    LocalOtlpTraceSink,
    LocalOtlpTraceUnavailableError,
)
from financial_reviewer.local.telemetry import (
    SanitizedTraceCause,
    SanitizedTraceSpan,
    TraceCauseKind,
    TraceSpanOutcome,
    TraceStage,
)
from tests.helpers import SYNTHETIC_EMPLOYER, SYNTHETIC_NAME


def _safe_span() -> SanitizedTraceSpan:
    started = datetime(2026, 8, 30, 12, 0, tzinfo=timezone.utc)
    return SanitizedTraceSpan(
        trace_id="1" * 32,
        span_id="2" * 16,
        parent_span_id="3" * 16,
        stage=TraceStage.DETERMINISTIC_EXTRACTION,
        outcome=TraceSpanOutcome.BLOCKED,
        started_at=started,
        ended_at=started + timedelta(milliseconds=7),
        duration_ms=7,
        document_type="pay_stub",
        causes=(
            SanitizedTraceCause(
                origin_stage=TraceStage.DETERMINISTIC_EXTRACTION,
                kind=TraceCauseKind.POLICY_BLOCK,
                reason_code="unsupported_required_field",
            ),
        ),
    )


def _attribute_values(message) -> dict[str, object]:
    values: dict[str, object] = {}
    for attribute in message.attributes:
        selected = attribute.value.WhichOneof("value")
        if selected == "string_value":
            values[attribute.key] = attribute.value.string_value
        elif selected == "int_value":
            values[attribute.key] = attribute.value.int_value
        elif selected == "array_value":
            values[attribute.key] = [
                item.string_value for item in attribute.value.array_value.values
            ]
    return values


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:4318/v1/traces",
        "http://localhost:4318/v1/traces",
        "http://192.168.1.10:4318/v1/traces",
        "http://127.0.0.1:4318/v1/logs",
        "http://127.0.0.1:4318/v1/traces?token=unsafe",
        "http://user:secret@127.0.0.1:4318/v1/traces",
        "http://127.0.0.1/v1/traces",
    ],
)
def test_otlp_settings_reject_every_non_loopback_trace_destination(
    endpoint: str,
) -> None:
    with pytest.raises(ValidationError):
        LocalOtlpTraceSettings(endpoint=endpoint)


def test_otlp_settings_read_only_the_approved_traceboard_environment() -> None:
    settings = LocalOtlpTraceSettings.from_environment(
        {
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318",
            "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
            "OTEL_SERVICE_NAME": "multi-agent-financial-document-reviewer",
            "OTEL_RESOURCE_ATTRIBUTES": (
                "traceboard.project.id=multi-agent-financial-document-reviewer"
            ),
            "UNRELATED_SECRET": SYNTHETIC_NAME,
        }
    )

    assert settings.endpoint == "http://127.0.0.1:4318/v1/traces"
    assert settings.service_name == "multi-agent-financial-document-reviewer"
    assert settings.project_id == "multi-agent-financial-document-reviewer"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("OTEL_EXPORTER_OTLP_ENDPOINT", "https://127.0.0.1:4318"),
        ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318"),
        ("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318/v1/traces"),
        ("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc"),
        ("OTEL_SERVICE_NAME", "another-project"),
        (
            "OTEL_RESOURCE_ATTRIBUTES",
            "traceboard.project.id=multi-agent-financial-document-reviewer,unsafe=value",
        ),
    ],
)
def test_otlp_environment_rejects_unsafe_or_expanded_configuration(
    name: str,
    value: str,
) -> None:
    environment = {
        "OTEL_EXPORTER_OTLP_ENDPOINT": "http://127.0.0.1:4318",
        "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
        "OTEL_SERVICE_NAME": "multi-agent-financial-document-reviewer",
        "OTEL_RESOURCE_ATTRIBUTES": (
            "traceboard.project.id=multi-agent-financial-document-reviewer"
        ),
    }
    environment[name] = value

    with pytest.raises(
        UnsafeRuntimeConfigurationError,
        match="failed security validation",
    ):
        LocalOtlpTraceSettings.from_environment(environment)


def test_otlp_environment_requires_all_four_approved_values() -> None:
    with pytest.raises(
        UnsafeRuntimeConfigurationError,
        match="configuration is incomplete",
    ):
        LocalOtlpTraceSettings.from_environment({})


def test_otlp_sink_maps_only_the_closed_safe_span_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            content=ExportTraceServiceResponse().SerializeToString(),
            headers={"content-type": "application/x-protobuf"},
        )

    @contextmanager
    def client_context(
        settings: LocalOtlpTraceSettings,
    ) -> Iterator[httpx.Client]:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            yield client

    monkeypatch.setattr(
        LocalOtlpTraceSink,
        "_client_context",
        staticmethod(client_context),
    )
    sink = LocalOtlpTraceSink()

    sink.emit_span(_safe_span())

    assert len(captured) == 1
    request = captured[0]
    assert str(request.url) == "http://127.0.0.1:4318/v1/traces"
    assert request.headers["content-type"] == "application/x-protobuf"
    envelope = ExportTraceServiceRequest.FromString(request.content)
    assert len(envelope.resource_spans) == 1
    resource = envelope.resource_spans[0]
    resource_attributes = _attribute_values(resource.resource)
    assert resource_attributes == {
        "service.name": "multi-agent-financial-document-reviewer",
        "traceboard.project.id": "multi-agent-financial-document-reviewer",
        "deployment.environment.name": "local",
    }
    otlp_span = resource.scope_spans[0].spans[0]
    assert otlp_span.trace_id == bytes.fromhex("1" * 32)
    assert otlp_span.span_id == bytes.fromhex("2" * 16)
    assert otlp_span.parent_span_id == bytes.fromhex("3" * 16)
    assert otlp_span.name == TraceStage.DETERMINISTIC_EXTRACTION.value
    assert otlp_span.status.message == ""
    assert _attribute_values(otlp_span) == {
        "review.stage": TraceStage.DETERMINISTIC_EXTRACTION.value,
        "review.outcome": TraceSpanOutcome.BLOCKED.value,
        "review.schema.version": "1.0",
        "review.cause.count": 1,
        "review.classification": "pay_stub",
        "review.cause.origin_stages": [
            TraceStage.DETERMINISTIC_EXTRACTION.value
        ],
        "review.cause.kinds": [TraceCauseKind.POLICY_BLOCK.value],
        "review.cause.reason_codes": ["unsupported_required_field"],
    }
    serialized = json.dumps(
        {
            "resource": resource_attributes,
            "span": _attribute_values(otlp_span),
        },
        sort_keys=True,
    )
    assert SYNTHETIC_NAME not in serialized
    assert SYNTHETIC_EMPLOYER not in serialized


def test_otlp_sink_rejects_constructed_span_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport_calls = 0

    @contextmanager
    def forbidden_client(
        settings: LocalOtlpTraceSettings,
    ) -> Iterator[httpx.Client]:
        nonlocal transport_calls
        transport_calls += 1
        raise AssertionError("transport must not be created")
        yield  # pragma: no cover

    monkeypatch.setattr(
        LocalOtlpTraceSink,
        "_client_context",
        staticmethod(forbidden_client),
    )
    malformed = SanitizedTraceSpan.model_construct(
        **{
            **_safe_span().model_dump(mode="python", warnings="none"),
            "span_id": SYNTHETIC_NAME,
        }
    )

    with pytest.raises(TypeError, match="sanitized OTLP trace span is invalid"):
        LocalOtlpTraceSink().emit_span(malformed)
    assert transport_calls == 0


@pytest.mark.parametrize(
    ("status_code", "expected_error"),
    [
        (503, LocalOtlpTraceUnavailableError),
        (200, LocalOtlpTraceResponseError),
    ],
)
def test_otlp_receiver_failures_are_sanitized(
    status_code: int,
    expected_error: type[Exception],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = b"not-protobuf" if status_code == 200 else SYNTHETIC_NAME.encode()
        return httpx.Response(status_code, content=body)

    @contextmanager
    def client_context(
        settings: LocalOtlpTraceSettings,
    ) -> Iterator[httpx.Client]:
        with httpx.Client(transport=httpx.MockTransport(handler)) as client:
            yield client

    monkeypatch.setattr(
        LocalOtlpTraceSink,
        "_client_context",
        staticmethod(client_context),
    )

    with pytest.raises(expected_error) as captured:
        LocalOtlpTraceSink().emit_span(_safe_span())
    assert SYNTHETIC_NAME not in str(captured.value)
