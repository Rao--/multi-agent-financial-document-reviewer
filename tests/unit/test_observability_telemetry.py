from __future__ import annotations

import io
import logging
import socket
import stat
import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from financial_reviewer.foundation.config import UnsafeRuntimeConfigurationError
from financial_reviewer.local.observability import (
    AuditIntegrityError,
    LocalJsonlAuditStore,
    PIISafeStructuredLogger,
    SafeEventMetadata,
    new_correlation_id,
    new_idempotency_key,
    new_opaque_document_id,
    validate_correlation_id,
)
from financial_reviewer.local.telemetry import (
    DisabledLangSmithTelemetrySink,
    LangSmithTelemetrySettings,
    NoOpReviewTraceSink,
    ReviewTraceExportError,
    ReviewTraceSession,
    SanitizedTraceCause,
    SanitizedTraceSpan,
    SanitizedTelemetryEvent,
    TelemetryEventType,
    TraceCauseKind,
    TraceSpanOutcome,
    TraceStage,
)
from tests.helpers import (
    MODEL_OUTPUT_SENTINEL,
    SYNTHETIC_EIN,
    SYNTHETIC_EMPLOYER,
    SYNTHETIC_NAME,
)


def _safe_metadata() -> SafeEventMetadata:
    return SafeEventMetadata(
        correlation_id=new_correlation_id(),
        idempotency_key=new_idempotency_key(),
        opaque_document_id=new_opaque_document_id(),
        component="agent_1_extraction",
        action="extraction_completed",
        status="succeeded",
        document_type="pay_stub",
        field_count=6,
        evidence_count=6,
        model_version="qwen2.5:3b",
        extraction_schema_version="1.0",
        workflow_version="financial-reviewer-v1",
    )


def test_opaque_correlation_ids_are_uuid_based() -> None:
    correlation_id = new_correlation_id()
    assert validate_correlation_id(correlation_id) == correlation_id
    assert correlation_id.startswith("corr_")
    assert SYNTHETIC_NAME.casefold() not in correlation_id.casefold()


@pytest.mark.parametrize(
    "forbidden_field",
    ["document_text", "prompt", "model_output", "account_number", "employee_name"],
)
def test_safe_log_schema_has_no_sensitive_payload_escape_hatch(
    forbidden_field: str,
) -> None:
    values = _safe_metadata().model_dump()
    values[forbidden_field] = SYNTHETIC_NAME
    with pytest.raises(ValidationError):
        SafeEventMetadata.model_validate(values)


def test_local_logger_and_audit_contain_only_allowlisted_metadata(tmp_path: Path) -> None:
    stream = io.StringIO()
    logger = logging.Logger("financial_reviewer_test_safe")
    logger.addHandler(logging.StreamHandler(stream))
    safe_logger = PIISafeStructuredLogger(logger)
    metadata = _safe_metadata()
    event = safe_logger.emit(metadata)

    audit_path = tmp_path / "private_audit" / "events.jsonl"
    audit = LocalJsonlAuditStore(audit_path)
    record = audit.append_event(event)
    assert record.metadata == metadata
    assert audit.verify_integrity().valid is True
    assert stat.S_IMODE(audit_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(audit_path.stat().st_mode) == 0o600

    rendered = stream.getvalue() + audit_path.read_text(encoding="utf-8")
    for forbidden in (
        SYNTHETIC_NAME,
        SYNTHETIC_EMPLOYER,
        SYNTHETIC_EIN,
        MODEL_OUTPUT_SENTINEL,
    ):
        assert forbidden not in rendered


def test_constructed_metadata_is_revalidated_before_log_or_audit(
    tmp_path: Path,
) -> None:
    sentinel = "SYNTHETIC PERSON MUST NEVER BE LOGGED"
    malicious = SafeEventMetadata.model_construct(
        **{
            **_safe_metadata().model_dump(),
            "model_version": sentinel,
        }
    )
    stream = io.StringIO()
    logger = logging.Logger("financial_reviewer_test_constructed")
    logger.addHandler(logging.StreamHandler(stream))
    safe_logger = PIISafeStructuredLogger(logger)
    audit = LocalJsonlAuditStore(tmp_path / "constructed_audit" / "events.jsonl")

    with pytest.raises(TypeError):
        safe_logger.emit(malicious)
    with pytest.raises(TypeError):
        audit.append(malicious)
    assert sentinel not in stream.getvalue()
    assert sentinel not in audit.path.read_text(encoding="utf-8")


def test_constructed_wrong_type_emits_no_pydantic_warning_with_pii() -> None:
    sentinel = "SYNTHETIC-PII-IN-WRONG-TYPE"
    malicious = SafeEventMetadata.model_construct(
        **{
            **_safe_metadata().model_dump(),
            "document_byte_count": sentinel,
        }
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        with pytest.raises(TypeError):
            PIISafeStructuredLogger(logging.Logger("warning_test")).emit(malicious)
    assert all(sentinel not in str(item.message) for item in captured)


def test_telemetry_contract_has_no_local_or_reversible_identifier() -> None:
    with pytest.raises(ValidationError):
        SanitizedTelemetryEvent(
            event_type=TelemetryEventType.WORKFLOW_COMPLETED,
            component="workflow",
            outcome="succeeded",
            correlation_id=new_correlation_id(),
        )


def test_constructed_langsmith_settings_cannot_enable_sink() -> None:
    bypass_attempt = LangSmithTelemetrySettings.model_construct(enabled=True)
    with pytest.raises(TypeError):
        DisabledLangSmithTelemetrySink(bypass_attempt)


def test_audit_integrity_check_detects_rewritten_records(tmp_path: Path) -> None:
    audit_path = tmp_path / "private_audit" / "events.jsonl"
    audit = LocalJsonlAuditStore(audit_path)
    audit.append(_safe_metadata())
    original = audit_path.read_text(encoding="utf-8")
    audit_path.write_text(original.replace('"sequence":1', '"sequence":2'), encoding="utf-8")
    audit_path.chmod(0o600)
    with pytest.raises(AuditIntegrityError):
        audit.verify_integrity()


def test_langsmith_sink_is_disabled_and_makes_no_socket_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network_calls = 0

    def forbidden_connect(*args: object, **kwargs: object) -> None:
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network access is forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)
    sink = DisabledLangSmithTelemetrySink()
    assert sink.enabled is False
    sink.emit(
        SanitizedTelemetryEvent(
            event_type=TelemetryEventType.WORKFLOW_COMPLETED,
            component="workflow",
            outcome="succeeded",
            document_type="pay_stub",
            schema_version="1.0",
        )
    )
    assert network_calls == 0


def test_langsmith_sink_rechecks_ambient_tracing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sink = DisabledLangSmithTelemetrySink()
    event = SanitizedTelemetryEvent(
        event_type=TelemetryEventType.WORKFLOW_FAILED,
        component="workflow",
        outcome="failed",
        reason_code="unsafe_configuration",
        document_type="unknown",
    )
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "1")
    with pytest.raises(UnsafeRuntimeConfigurationError):
        sink.emit(event)


def test_review_trace_session_enforces_parent_child_lifecycle() -> None:
    spans: list[SanitizedTraceSpan] = []

    class CapturingSink:
        def emit_span(self, span: SanitizedTraceSpan) -> None:
            spans.append(span)

    session = ReviewTraceSession(CapturingSink())
    root = session.start_root()
    child = root.start_child(TraceStage.INPUT_VALIDATION)

    with pytest.raises(RuntimeError, match="active child"):
        root.finish(TraceSpanOutcome.FAILED)

    cause = SanitizedTraceCause(
        origin_stage=TraceStage.INPUT_VALIDATION,
        kind=TraceCauseKind.VALIDATION_FAILURE,
        reason_code="invalid_input",
    )
    child.finish(TraceSpanOutcome.REJECTED, causes=(cause,))
    root.finish(TraceSpanOutcome.HUMAN_REVIEW, causes=session.causes)

    assert len(spans) == 2
    assert spans[0].parent_span_id == spans[1].span_id
    assert spans[0].trace_id == spans[1].trace_id
    assert spans[1].parent_span_id is None
    with pytest.raises(RuntimeError, match="already complete"):
        child.finish(TraceSpanOutcome.REJECTED, causes=(cause,))


def test_optional_trace_export_failure_does_not_change_trace_lifecycle() -> None:
    """Traceboard projection is optional; the mandatory local audit remains."""

    class UnavailableSink:
        def emit_span(self, span: SanitizedTraceSpan) -> None:
            raise ReviewTraceExportError("sanitized projection failure")

    session = ReviewTraceSession(UnavailableSink())
    root = session.start_root()
    child = root.start_child(TraceStage.AGENT_2_VERIFICATION)

    child.finish(TraceSpanOutcome.SUCCEEDED)
    root.finish(TraceSpanOutcome.SUCCEEDED)

    assert session.export_failure_count == 2


def test_trace_span_schema_has_no_sensitive_payload_escape_hatch() -> None:
    session = ReviewTraceSession(NoOpReviewTraceSink())
    root = session.start_root()
    completed = root.finish(TraceSpanOutcome.SUCCEEDED)
    unsafe = {
        **completed.model_dump(mode="python", warnings="none"),
        "document_text": SYNTHETIC_NAME,
    }

    with pytest.raises(ValidationError):
        SanitizedTraceSpan.model_validate(unsafe)


def test_noop_trace_sink_revalidates_constructed_span() -> None:
    session = ReviewTraceSession(NoOpReviewTraceSink())
    root = session.start_root()
    completed = root.finish(TraceSpanOutcome.SUCCEEDED)
    malformed = SanitizedTraceSpan.model_construct(
        **{
            **completed.model_dump(mode="python", warnings="none"),
            "span_id": SYNTHETIC_NAME,
        }
    )

    with pytest.raises(TypeError, match="sanitized trace span is invalid"):
        NoOpReviewTraceSink().emit_span(malformed)
