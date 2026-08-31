from __future__ import annotations

import json
import socket
import stat
from pathlib import Path

import httpx
import pytest
from langchain_core.globals import set_debug

import financial_reviewer.workflow as workflow_module
from financial_reviewer.agents.agent1_extraction import extract_pay_stub_deterministically
from financial_reviewer.foundation.config import LocalModelSettings, UnsafeRuntimeConfigurationError
from financial_reviewer.foundation.intake import SYNTHETIC_MARKER, DocumentSubmission
from financial_reviewer.foundation.schemas import (
    DocumentType,
    FailureCode,
    WorkflowStatus,
    iter_extracted_fields,
)
from financial_reviewer.local.model import OllamaModel
from financial_reviewer.local.observability import AuditWriteError, LocalJsonlAuditStore
from financial_reviewer.local.storage import LocalDocumentStore, StorageError
from financial_reviewer.local.telemetry import (
    ReviewTraceSink,
    SanitizedTelemetryEvent,
    SanitizedTraceSpan,
    TelemetrySink,
    TraceSpanOutcome,
    TraceStage,
)
from financial_reviewer.workflow import DocumentExtractionReviewer, WorkflowState
from tests.helpers import (
    ScriptedOllama,
    SYNTHETIC_EIN,
    SYNTHETIC_EMPLOYEE_ID,
    SYNTHETIC_EMPLOYER,
    SYNTHETIC_INCOME_SOURCE,
    SYNTHETIC_NAME,
)


class CapturingTraceSink:
    """Test adapter retaining only the already-sanitized span contract."""

    def __init__(self) -> None:
        self.spans: list[SanitizedTraceSpan] = []

    def emit_span(self, span: SanitizedTraceSpan) -> None:
        self.spans.append(
            SanitizedTraceSpan.model_validate(
                span.model_dump(mode="python", warnings="none")
            )
        )


def _reviewer(
    tmp_path: Path,
    script: ScriptedOllama,
    monkeypatch: pytest.MonkeyPatch,
    *,
    audit_store: LocalJsonlAuditStore | None = None,
    telemetry: TelemetrySink | None = None,
    trace_sink: ReviewTraceSink | None = None,
) -> tuple[DocumentExtractionReviewer, LocalJsonlAuditStore]:
    runtime = tmp_path / "financial_reviewer_runtime"
    local_audit = LocalJsonlAuditStore(runtime / "audit" / "events.jsonl")
    script.install(monkeypatch)
    reviewer = DocumentExtractionReviewer(
        settings=LocalModelSettings(),
        document_store=LocalDocumentStore(runtime / "documents"),
        audit_store=audit_store or local_audit,
        telemetry=telemetry,
        trace_sink=trace_sink,
    )
    return reviewer, local_audit


def test_synthetic_end_to_end_runs_agent1_with_no_network(
    tmp_path: Path,
    synthetic_submission: DocumentSubmission,
    synthetic_pay_stub_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    network_calls = 0

    def forbidden_connect(*args: object, **kwargs: object) -> None:
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network access is forbidden in the integration test")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)
    model = ScriptedOllama([])
    reviewer, audit = _reviewer(tmp_path, model, monkeypatch)

    outcome = reviewer.review(synthetic_submission)

    assert network_calls == 0
    assert model.calls == []
    assert outcome.status is WorkflowStatus.RELEASED
    assert outcome.human_review_required is False
    assert outcome.failure_code is None
    assert outcome.validated_extraction is not None
    assert outcome.validated_extraction.document_type == "pay_stub"
    assert outcome.validated_extraction.metadata.extraction_method == (
        "deterministic_labels_v1"
    )
    assert outcome.validated_extraction.metadata.attempt_count == 0
    assert outcome.validated_extraction.metadata.model_provider is None
    assert outcome.validated_extraction.metadata.model_name is None
    assert all(
        field.provenance
        for _, field in iter_extracted_fields(outcome.validated_extraction)
    )
    assert set(reviewer.workflow_node_names) == {
        "classify",
        "deterministic_extract",
        "schema_validate",
        "evidence_guard",
        "finalize",
    }
    assert not any(name.startswith("agent_2") or name.startswith("agent_3") for name in reviewer.workflow_node_names)

    rendered_audit = audit.path.read_text(encoding="utf-8")
    for forbidden in (
        SYNTHETIC_NAME,
        SYNTHETIC_EMPLOYEE_ID,
        SYNTHETIC_EMPLOYER,
        SYNTHETIC_EIN,
        SYNTHETIC_INCOME_SOURCE,
        synthetic_pay_stub_text,
    ):
        assert forbidden not in rendered_audit
    assert audit.verify_integrity().valid is True


@pytest.mark.parametrize(
    ("fixture_name", "filename", "expected_type", "expected_field_count"),
    [
        ("synthetic_w2_text", "synthetic_w2.txt", DocumentType.TAX_FORM, 8),
        (
            "synthetic_bank_statement_text",
            "synthetic_bank_statement.txt",
            DocumentType.BANK_STATEMENT,
            5,
        ),
    ],
)
def test_new_document_types_run_through_agent1_without_model_or_network(
    request: pytest.FixtureRequest,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture_name: str,
    filename: str,
    expected_type: DocumentType,
    expected_field_count: int,
) -> None:
    """W-2 and bank inputs use the same validation and evidence release gates."""

    document_text = request.getfixturevalue(fixture_name)
    submission = DocumentSubmission(
        filename=filename,
        content_type="text/plain",
        content=document_text.encode("utf-8"),
        declared_synthetic=True,
    )
    network_calls = 0

    def forbidden_connect(*args: object, **kwargs: object) -> None:
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network access is forbidden in the integration test")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)
    model = ScriptedOllama([])
    reviewer, audit = _reviewer(tmp_path, model, monkeypatch)

    outcome = reviewer.review(submission)

    assert network_calls == 0
    assert model.calls == []
    assert outcome.status is WorkflowStatus.RELEASED
    assert outcome.document_type is expected_type
    assert outcome.failure_code is None
    assert outcome.human_review_required is False
    assert outcome.validated_extraction is not None
    assert outcome.validated_extraction.metadata.attempt_count == 0
    fields = list(iter_extracted_fields(outcome.validated_extraction))
    assert len(fields) == expected_field_count
    assert all(field.status == "supported" and field.provenance for _, field in fields)

    rendered_audit = audit.path.read_text(encoding="utf-8")
    for forbidden in (
        document_text,
        "SYNTHETIC PERSON ALPHA",
        "000-00-0001",
        "100 SYNTHETIC WAY",
        "SYN-ACCT-0001",
        "SYNTHETIC COMMUNITY BANK",
        "75,000.00",
        "6,250.00",
    ):
        assert forbidden not in rendered_audit
    assert audit.verify_integrity().valid is True


def test_agent1_success_has_one_private_parent_child_trace(
    tmp_path: Path,
    synthetic_submission: DocumentSubmission,
    synthetic_pay_stub_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_sink = CapturingTraceSink()
    reviewer, _ = _reviewer(
        tmp_path,
        ScriptedOllama([]),
        monkeypatch,
        trace_sink=trace_sink,
    )

    outcome = reviewer.review(synthetic_submission)

    assert outcome.status is WorkflowStatus.RELEASED
    spans_by_stage = {span.stage: span for span in trace_sink.spans}
    assert set(spans_by_stage) == {
        TraceStage.FINANCIAL_REVIEW,
        TraceStage.RUNTIME_PREFLIGHT,
        TraceStage.INPUT_VALIDATION,
        TraceStage.DOCUMENT_STORAGE,
        TraceStage.TELEMETRY_POLICY,
        TraceStage.AGENT_1_EXTRACTION,
        TraceStage.CLASSIFICATION,
        TraceStage.DETERMINISTIC_EXTRACTION,
        TraceStage.SCHEMA_VALIDATION,
        TraceStage.EVIDENCE_GUARD,
        TraceStage.AGENT_1_FINALIZE,
        TraceStage.FINAL_REVIEW_DECISION,
    }
    assert len({span.trace_id for span in trace_sink.spans}) == 1

    root = spans_by_stage[TraceStage.FINANCIAL_REVIEW]
    agent_1 = spans_by_stage[TraceStage.AGENT_1_EXTRACTION]
    root_children = {
        span.stage
        for span in trace_sink.spans
        if span.parent_span_id == root.span_id
    }
    assert root_children == {
        TraceStage.RUNTIME_PREFLIGHT,
        TraceStage.INPUT_VALIDATION,
        TraceStage.DOCUMENT_STORAGE,
        TraceStage.TELEMETRY_POLICY,
        TraceStage.AGENT_1_EXTRACTION,
        TraceStage.FINAL_REVIEW_DECISION,
    }
    assert {
        span.stage
        for span in trace_sink.spans
        if span.parent_span_id == agent_1.span_id
    } == {
        TraceStage.CLASSIFICATION,
        TraceStage.DETERMINISTIC_EXTRACTION,
        TraceStage.SCHEMA_VALIDATION,
        TraceStage.EVIDENCE_GUARD,
        TraceStage.AGENT_1_FINALIZE,
    }
    assert all(span.outcome is not TraceSpanOutcome.FAILED for span in trace_sink.spans)
    assert root.outcome is TraceSpanOutcome.SUCCEEDED
    assert root.causes == ()

    rendered = json.dumps(
        [span.model_dump(mode="json") for span in trace_sink.spans],
        sort_keys=True,
    )
    for forbidden in (
        outcome.correlation_id,
        outcome.idempotency_key,
        SYNTHETIC_NAME,
        SYNTHETIC_EMPLOYEE_ID,
        SYNTHETIC_EMPLOYER,
        SYNTHETIC_EIN,
        SYNTHETIC_INCOME_SOURCE,
        synthetic_pay_stub_text,
    ):
        assert forbidden not in rendered


def test_invalid_input_trace_identifies_pre_agent_failure_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_sink = CapturingTraceSink()
    reviewer, _ = _reviewer(
        tmp_path,
        ScriptedOllama([]),
        monkeypatch,
        trace_sink=trace_sink,
    )

    outcome = reviewer.review(
        {
            "filename": "invalid.txt",
            "content_type": "text/plain",
            "content": b"",
            "declared_synthetic": True,
        }
    )

    assert outcome.failure_code is FailureCode.INVALID_INPUT
    stages = {span.stage for span in trace_sink.spans}
    assert TraceStage.INPUT_VALIDATION in stages
    assert TraceStage.DOCUMENT_STORAGE not in stages
    assert TraceStage.AGENT_1_EXTRACTION not in stages
    input_span = next(
        span
        for span in trace_sink.spans
        if span.stage is TraceStage.INPUT_VALIDATION
    )
    assert input_span.outcome is TraceSpanOutcome.REJECTED
    assert [(cause.origin_stage, cause.reason_code) for cause in input_span.causes] == [
        (TraceStage.INPUT_VALIDATION, FailureCode.INVALID_INPUT.value)
    ]
    root = next(
        span
        for span in trace_sink.spans
        if span.stage is TraceStage.FINANCIAL_REVIEW
    )
    assert root.outcome is TraceSpanOutcome.HUMAN_REVIEW
    assert [(cause.origin_stage, cause.reason_code) for cause in root.causes] == [
        (TraceStage.INPUT_VALIDATION, FailureCode.INVALID_INPUT.value)
    ]


def test_agent1_failure_trace_preserves_leaf_cause_without_duplication(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_sink = CapturingTraceSink()
    reviewer, _ = _reviewer(
        tmp_path,
        ScriptedOllama([]),
        monkeypatch,
        trace_sink=trace_sink,
    )
    submission = DocumentSubmission(
        filename="synthetic_pay_stub_missing_id.txt",
        content_type="text/plain",
        content=synthetic_pay_stub_text.replace(
            f"Employee ID: {SYNTHETIC_EMPLOYEE_ID}\n",
            "",
        ).encode(),
        declared_synthetic=True,
    )

    outcome = reviewer.review(submission)

    assert outcome.failure_code is FailureCode.UNSUPPORTED_REQUIRED_FIELD
    spans_by_stage = {span.stage: span for span in trace_sink.spans}
    assert TraceStage.SCHEMA_VALIDATION not in spans_by_stage
    assert TraceStage.EVIDENCE_GUARD not in spans_by_stage
    extraction = spans_by_stage[TraceStage.DETERMINISTIC_EXTRACTION]
    assert extraction.outcome is TraceSpanOutcome.BLOCKED
    expected_cause = (
        TraceStage.DETERMINISTIC_EXTRACTION,
        FailureCode.UNSUPPORTED_REQUIRED_FIELD.value,
    )
    assert [
        (cause.origin_stage, cause.reason_code) for cause in extraction.causes
    ] == [expected_cause]
    root = spans_by_stage[TraceStage.FINANCIAL_REVIEW]
    final_decision = spans_by_stage[TraceStage.FINAL_REVIEW_DECISION]
    assert [
        (cause.origin_stage, cause.reason_code) for cause in root.causes
    ] == [expected_cause]
    assert [
        (cause.origin_stage, cause.reason_code)
        for cause in final_decision.causes
    ] == [expected_cause]


@pytest.mark.parametrize(
    "invalid_submission",
    [
        {
            "filename": "bad.txt",
            "content_type": "text/plain",
            "content": b"",
            "declared_synthetic": True,
        },
        {
            "filename": "bad.txt",
            "content_type": "text/plain",
            "content": b"NOT SYNTHETIC\nPAY STUB\n",
            "declared_synthetic": True,
        },
        {
            "filename": "../bad.txt",
            "content_type": "text/plain",
            "content": (SYNTHETIC_MARKER + "\nPAY STUB\n").encode(),
            "declared_synthetic": True,
        },
    ],
)
def test_invalid_input_never_reaches_model(
    tmp_path: Path,
    invalid_submission: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedOllama([])
    reviewer, _ = _reviewer(tmp_path, model, monkeypatch)
    outcome = reviewer.review(invalid_submission)
    assert model.calls == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.validated_extraction is None
    assert outcome.failure_code is FailureCode.INVALID_INPUT


def test_constructed_invalid_submission_never_reaches_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedOllama([])
    reviewer, _ = _reviewer(tmp_path, model, monkeypatch)
    bypass_attempt = DocumentSubmission.model_construct(
        filename="../synthetic.pdf",
        content_type="application/pdf",
        content=(SYNTHETIC_MARKER + "\nPAY STUB\n").encode(),
        declared_synthetic=False,
    )
    outcome = reviewer.review(bypass_attempt)
    assert model.calls == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.failure_code is FailureCode.INVALID_INPUT


def test_langchain_debug_is_blocked_before_input_or_console_output(
    tmp_path: Path,
    synthetic_submission: DocumentSubmission,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = ScriptedOllama([])
    reviewer, audit = _reviewer(tmp_path, model, monkeypatch)
    set_debug(True)
    try:
        outcome = reviewer.review(synthetic_submission)
    finally:
        set_debug(False)
    captured = capsys.readouterr()
    assert model.calls == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.failure_code is FailureCode.UNSAFE_CONFIGURATION
    assert SYNTHETIC_NAME not in captured.out + captured.err
    records = audit.list_records(correlation_id=outcome.correlation_id)
    assert len(records) == 1
    assert records[0].metadata.error_code == "UNSAFE_CONFIGURATION"


def test_callback_race_never_exposes_sensitive_node_state(
    tmp_path: Path,
    synthetic_submission: DocumentSubmission,
    synthetic_pay_stub_text: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    script = ScriptedOllama([])
    reviewer, audit = _reviewer(tmp_path, script, monkeypatch)
    original_extractor = extract_pay_stub_deterministically

    def enable_debug_during_extraction(**kwargs):
        # Simulate a process-global tracing toggle in the narrow interval after
        # graph preflight but before the deterministic node returns.
        extraction = original_extractor(**kwargs)
        set_debug(True)
        return extraction

    monkeypatch.setattr(
        workflow_module,
        "extract_pay_stub_deterministically",
        enable_debug_during_extraction,
    )
    try:
        outcome = reviewer.review(synthetic_submission)
    finally:
        set_debug(False)

    captured = capsys.readouterr()
    rendered = captured.out + captured.err + audit.path.read_text(encoding="utf-8")
    assert script.calls == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.failure_code is FailureCode.UNSAFE_CONFIGURATION
    for forbidden in (
        SYNTHETIC_NAME,
        SYNTHETIC_EMPLOYEE_ID,
        SYNTHETIC_EMPLOYER,
        SYNTHETIC_EIN,
        SYNTHETIC_INCOME_SOURCE,
        synthetic_pay_stub_text,
    ):
        assert forbidden not in rendered


def test_callback_visible_workflow_state_has_no_sensitive_artifact_fields(
    tmp_path: Path,
) -> None:
    forbidden_state_fields = {
        "correlation_id",
        "idempotency_key",
        "document_id",
        "document_text",
        "document_sha256",
        "raw_model_output",
        "extraction_proposal",
        "candidate_extraction",
        "validated_extraction",
        "verification_findings",
    }
    assert forbidden_state_fields.isdisjoint(WorkflowState.__annotations__)

    runtime = tmp_path / "state_allowlist"
    reviewer = DocumentExtractionReviewer(
        settings=LocalModelSettings(),
        document_store=LocalDocumentStore(runtime / "documents"),
        audit_store=LocalJsonlAuditStore(runtime / "audit" / "events.jsonl"),
    )
    with pytest.raises(RuntimeError, match="undeclared fields"):
        reviewer._workflow._normalize_graph_input(  # noqa: SLF001
            {
                "document_type": "unknown",
                "document_text": SYNTHETIC_NAME,
            }  # type: ignore[typeddict-unknown-key,typeddict-item]
        )


def test_unknown_document_type_routes_to_human_without_model_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedOllama([])
    reviewer, _ = _reviewer(tmp_path, model, monkeypatch)
    submission = DocumentSubmission(
        filename="synthetic_unknown.txt",
        content_type="text/plain",
        content=(
            SYNTHETIC_MARKER + "\nUNSUPPORTED FINANCIAL DOCUMENT\nField: value\n"
        ).encode(),
        declared_synthetic=True,
    )
    outcome = reviewer.review(submission)
    assert model.calls == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.failure_code is FailureCode.UNSUPPORTED_DOCUMENT_TYPE
    assert outcome.validated_extraction is None


def test_missing_required_label_fails_closed_without_model_fallback(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedOllama([])
    reviewer, _ = _reviewer(tmp_path, model, monkeypatch)
    submission = DocumentSubmission(
        filename="synthetic_pay_stub_missing_id.txt",
        content_type="text/plain",
        content=synthetic_pay_stub_text.replace(
            f"Employee ID: {SYNTHETIC_EMPLOYEE_ID}\n",
            "",
        ).encode(),
        declared_synthetic=True,
    )

    outcome = reviewer.review(submission)

    assert model.calls == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.failure_code is FailureCode.UNSUPPORTED_REQUIRED_FIELD
    assert outcome.validated_extraction is None


@pytest.mark.parametrize(
    ("replacement", "expected_failure"),
    [
        ("Monthly Income: not-a-number", FailureCode.SCHEMA_VALIDATION_FAILED),
        (
            "Monthly Income: $6,250.00\nMonthly Income: $9,999.99",
            FailureCode.EVIDENCE_VALIDATION_FAILED,
        ),
    ],
)
def test_malformed_or_conflicting_value_fails_closed_without_model_fallback(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    replacement: str,
    expected_failure: FailureCode,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedOllama([])
    reviewer, _ = _reviewer(tmp_path, model, monkeypatch)
    altered = synthetic_pay_stub_text.replace(
        f"Monthly Income: {SYNTHETIC_INCOME_SOURCE}",
        replacement,
    )
    submission = DocumentSubmission(
        filename="synthetic_pay_stub_unresolved.txt",
        content_type="text/plain",
        content=altered.encode(),
        declared_synthetic=True,
    )

    outcome = reviewer.review(submission)

    assert model.calls == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.failure_code is expected_failure
    assert outcome.validated_extraction is None


@pytest.mark.parametrize("header", ["LOAN APPLICATION", "UNRECOGNIZED FORM"])
def test_unknown_document_types_route_to_human_without_model_call(
    tmp_path: Path,
    header: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedOllama([])
    reviewer, _ = _reviewer(tmp_path, model, monkeypatch)
    submission = DocumentSubmission(
        filename="synthetic_unknown_type.txt",
        content_type="text/plain",
        content=(
            f"{SYNTHETIC_MARKER}\n{header}\n"
            "Employee Name: SYNTHETIC PERSON BETA\n"
        ).encode(),
        declared_synthetic=True,
    )

    outcome = reviewer.review(submission)

    assert model.calls == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.failure_code is FailureCode.UNSUPPORTED_DOCUMENT_TYPE
    assert outcome.validated_extraction is None


def test_final_audit_failure_prevents_validated_release(
    tmp_path: Path,
    synthetic_submission: DocumentSubmission,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedOllama([])
    runtime = tmp_path / "financial_reviewer_runtime"
    delegate = LocalJsonlAuditStore(runtime / "audit" / "events.jsonl")
    original_append_event = LocalJsonlAuditStore.append_event
    append_count = 0

    def fail_twelfth_append(self, event):
        nonlocal append_count
        append_count += 1
        if self is delegate and append_count == 12:
            raise OSError("synthetic audit failure")
        return original_append_event(self, event)

    monkeypatch.setattr(LocalJsonlAuditStore, "append_event", fail_twelfth_append)
    reviewer, _ = _reviewer(
        tmp_path,
        model,
        monkeypatch,
        audit_store=delegate,
    )

    outcome = reviewer.review(synthetic_submission)

    assert model.calls == []
    assert append_count == 12
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.failure_code is FailureCode.AUDIT_WRITE_FAILED
    assert outcome.validated_extraction is None


def test_post_storage_telemetry_failure_has_terminal_local_audit(
    tmp_path: Path,
    synthetic_submission: DocumentSubmission,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingTelemetry:
        def emit(self, event: SanitizedTelemetryEvent) -> None:
            raise RuntimeError("synthetic telemetry failure")

    model = ScriptedOllama([])
    reviewer, audit = _reviewer(
        tmp_path,
        model,
        monkeypatch,
        telemetry=FailingTelemetry(),
    )
    outcome = reviewer.review(synthetic_submission)
    records = audit.list_records(correlation_id=outcome.correlation_id)

    assert model.calls == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.failure_code is FailureCode.UNSAFE_CONFIGURATION
    assert records[-1].metadata.component == "telemetry"
    assert records[-1].metadata.action == "release_blocked"
    assert records[-1].metadata.error_code == "UNSAFE_CONFIGURATION"


def test_invoke_preflight_failure_has_terminal_local_audit(
    tmp_path: Path,
    synthetic_submission: DocumentSubmission,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DebugTogglingTelemetry:
        def emit(self, event: SanitizedTelemetryEvent) -> None:
            set_debug(True)

    model = ScriptedOllama([])
    reviewer, audit = _reviewer(
        tmp_path,
        model,
        monkeypatch,
        telemetry=DebugTogglingTelemetry(),
    )
    try:
        outcome = reviewer.review(synthetic_submission)
    finally:
        set_debug(False)
    records = audit.list_records(correlation_id=outcome.correlation_id)

    assert model.calls == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.failure_code is FailureCode.UNSAFE_CONFIGURATION
    assert records[-1].metadata.component == "workflow"
    assert records[-1].metadata.action == "workflow_failed"
    assert records[-1].metadata.error_code == "UNSAFE_CONFIGURATION"


def test_invoke_audit_error_is_not_misreported_as_unsafe_configuration(
    tmp_path: Path,
    synthetic_submission: DocumentSubmission,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = ScriptedOllama([])
    reviewer, audit = _reviewer(tmp_path, model, monkeypatch)

    def fail_invoke(*args: object, **kwargs: object):
        raise AuditWriteError("synthetic audit failure")

    monkeypatch.setattr(type(reviewer._workflow), "invoke", fail_invoke)  # noqa: SLF001
    outcome = reviewer.review(synthetic_submission)
    records = audit.list_records(correlation_id=outcome.correlation_id)

    assert model.calls == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.failure_code is FailureCode.AUDIT_WRITE_FAILED
    assert records[-1].metadata.action == "workflow_failed"
    assert records[-1].metadata.error_code == "AUDIT_WRITE_FAILED"


def test_reviewer_rejects_nonstandard_or_mismatched_model_adapter(
    tmp_path: Path,
) -> None:
    settings = LocalModelSettings()
    runtime = tmp_path / "sealed_runtime"
    store = LocalDocumentStore(runtime / "documents")
    audit = LocalJsonlAuditStore(runtime / "audit" / "events.jsonl")
    with pytest.raises(TypeError):
        OllamaModel(  # type: ignore[call-arg]
            settings,
            transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"response": "{}"})
            ),
        )

    alternate_settings = LocalModelSettings(
        model="qwen2.5:7b",
        allowed_models=("qwen2.5:7b",),
    )
    with pytest.raises(UnsafeRuntimeConfigurationError):
        DocumentExtractionReviewer(
            settings=settings,
            document_store=store,
            audit_store=audit,
            model=OllamaModel(alternate_settings),
        )


def test_post_construction_model_mutation_fails_before_egress(
    tmp_path: Path,
    synthetic_submission: DocumentSubmission,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = ScriptedOllama([])
    reviewer, _ = _reviewer(tmp_path, script, monkeypatch)
    alternate = LocalModelSettings(
        model="qwen2.5:7b",
        allowed_models=("qwen2.5:7b",),
    )
    with pytest.raises(AttributeError):
        reviewer._model._settings = alternate  # type: ignore[misc]  # noqa: SLF001

    # Simulate memory corruption or a hostile trusted caller bypassing the seal;
    # the workflow's expected-settings attestation must still stop egress.
    object.__setattr__(reviewer._model, "_settings", alternate)
    outcome = reviewer.review(synthetic_submission)
    assert script.calls == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.failure_code is FailureCode.UNSAFE_CONFIGURATION


def test_reviewer_rejects_nonlocal_audit_implementation(tmp_path: Path) -> None:
    class NoOpAudit:
        def append_event(self, event):
            return None

    runtime = tmp_path / "audit_boundary"
    with pytest.raises(TypeError):
        DocumentExtractionReviewer(
            settings=LocalModelSettings(),
            document_store=LocalDocumentStore(runtime / "documents"),
            audit_store=NoOpAudit(),  # type: ignore[arg-type]
        )

    class RewritingLogger:
        def emit(self, metadata):
            raise AssertionError("untrusted logger must not be called")

    with pytest.raises(TypeError):
        DocumentExtractionReviewer(
            settings=LocalModelSettings(),
            document_store=LocalDocumentStore(runtime / "other_documents"),
            audit_store=LocalJsonlAuditStore(
                runtime / "other_audit" / "events.jsonl"
            ),
            logger=RewritingLogger(),  # type: ignore[arg-type]
        )


def test_local_factory_rejects_broad_existing_root_without_chmod(
    tmp_path: Path,
) -> None:
    shared_root = tmp_path / "shared_workspace"
    shared_root.mkdir(mode=0o755)
    shared_root.chmod(0o755)
    with pytest.raises(StorageError):
        DocumentExtractionReviewer.local(shared_root)
    assert stat.S_IMODE(shared_root.stat().st_mode) == 0o755
