"""Synthetic integration tests for the gated three-agent orchestration."""

from __future__ import annotations

import socket
from collections import Counter
from pathlib import Path

import pytest

from financial_reviewer.agents.agent2_verification import (
    DecisionAction,
    IncomeVerificationAgent,
    IncomeVerificationResult,
    InvalidToolDecisionError,
    InvalidToolDecisionReason,
    MAX_VERIFICATION_MODEL_DECISIONS,
    MAX_VERIFICATION_TOOL_CALLS,
    OllamaIncomeToolDecisionModel,
    VerificationDecisionContext,
    VerificationFailureCode,
    VerificationReason,
    VerificationStatus,
    VerificationToolDecision,
    VerificationToolName,
)
from financial_reviewer.agents.agent3_critic import (
    CriticDecision,
    CriticDecisionContext,
    CriticDisposition,
    CriticFailureCode,
    CriticReasonCode,
    CriticRepairReason,
    CriticStatus,
    InvalidCriticDecisionError,
    InvalidCriticDecisionReason,
    OllamaCriticDecisionModel,
)
from financial_reviewer.foundation.handoffs import (
    CriticHandoffError,
    CriticHandoffFailureCode,
    CriticInputAssembler,
    VerificationHandoffFailureCode,
)
from financial_reviewer.foundation.intake import DocumentSubmission
from financial_reviewer.foundation.schemas import (
    DocumentType,
    FailureCode,
    WorkflowStatus,
)
from financial_reviewer.local.telemetry import (
    SanitizedTraceSpan,
    TraceSpanOutcome,
    TraceStage,
)
from financial_reviewer.orchestration.income_review import (
    IncomeReviewBundle,
    IncomeReviewFailureCode,
    IncomeReviewOutcome,
    IncomeReviewOrchestrator,
    IncomeReviewReasonCode,
    IncomeReviewState,
)


class _CapturingTraceSink:
    """Retain only revalidated sanitized spans for hierarchy assertions."""

    def __init__(self) -> None:
        self.spans: list[SanitizedTraceSpan] = []

    def emit_span(self, span: SanitizedTraceSpan) -> None:
        self.spans.append(
            SanitizedTraceSpan.model_validate(
                span.model_dump(mode="python", warnings="none")
            )
        )


def _submission(filename: str, document_text: str) -> DocumentSubmission:
    """Wrap one approved synthetic fixture in the secure intake contract."""

    return DocumentSubmission(
        filename=filename,
        content_type="text/plain",
        content=document_text.encode("utf-8"),
        declared_synthetic=True,
    )


def _bundle(pay_stub_text: str, w2_text: str) -> IncomeReviewBundle:
    """Build the supported two-document request."""

    return IncomeReviewBundle(
        documents=(
            _submission("synthetic_pay_stub.txt", pay_stub_text),
            _submission("synthetic_w2.txt", w2_text),
        )
    )


def _install_happy_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> list[VerificationDecisionContext]:
    """Replace only local model decisions while executing real guarded tools."""

    contexts: list[VerificationDecisionContext] = []

    def decide(
        _adapter: OllamaIncomeToolDecisionModel,
        context: VerificationDecisionContext,
    ) -> VerificationToolDecision:
        contexts.append(context)
        return _next_happy_decision(context)

    monkeypatch.setattr(OllamaIncomeToolDecisionModel, "decide", decide)
    return contexts


def _next_happy_decision(
    context: VerificationDecisionContext,
) -> VerificationToolDecision:
    """Select the next valid action while letting the real tools execute."""

    normalized_types = {item.document_type for item in context.normalized_income}
    if DocumentType.PAY_STUB not in normalized_types:
        pay_stub = next(
            item
            for item in context.evidence
            if item.document_type is DocumentType.PAY_STUB
        )
        return VerificationToolDecision(
            action=DecisionAction.CALL_TOOL,
            tool_name=VerificationToolName.NORMALIZE_PAYSTUB_INCOME,
            evidence_ref=pay_stub.evidence_ref,
        )
    if DocumentType.TAX_FORM not in normalized_types:
        tax_form = next(
            item
            for item in context.evidence
            if item.document_type is DocumentType.TAX_FORM
        )
        return VerificationToolDecision(
            action=DecisionAction.CALL_TOOL,
            tool_name=VerificationToolName.NORMALIZE_W2_INCOME,
            evidence_ref=tax_form.evidence_ref,
        )
    comparison_done = any(
        item.tool_name is VerificationToolName.COMPARE_INCOME_SOURCES
        for item in context.observations
    )
    if not comparison_done:
        left, right = context.normalized_income
        return VerificationToolDecision(
            action=DecisionAction.CALL_TOOL,
            tool_name=VerificationToolName.COMPARE_INCOME_SOURCES,
            left_normalized_ref=left.normalized_ref,
            right_normalized_ref=right.normalized_ref,
        )
    return VerificationToolDecision(action=DecisionAction.COMPLETE)


def _install_compatible_critic_decisions(
    monkeypatch: pytest.MonkeyPatch,
) -> list[CriticDecisionContext]:
    """Return the exact Critic disposition allowed for each Agent 2 status."""

    contexts: list[CriticDecisionContext] = []

    def decide(
        _adapter: OllamaCriticDecisionModel,
        context: CriticDecisionContext,
    ) -> CriticDecision:
        contexts.append(context)
        decisions = {
            VerificationStatus.CONSISTENT: CriticDecision(
                outcome=CriticDisposition.GROUNDED,
                reason_code=CriticReasonCode.EVIDENCE_CONSISTENT,
            ),
            VerificationStatus.INCONSISTENT: CriticDecision(
                outcome=CriticDisposition.ESCALATE,
                reason_code=CriticReasonCode.INCOME_INCONSISTENT,
            ),
            VerificationStatus.NOT_COMPARABLE: CriticDecision(
                outcome=CriticDisposition.ESCALATE,
                reason_code=CriticReasonCode.INCOME_NOT_COMPARABLE,
            ),
            VerificationStatus.INSUFFICIENT_EVIDENCE: CriticDecision(
                outcome=CriticDisposition.REFUSE,
                reason_code=CriticReasonCode.INSUFFICIENT_EVIDENCE,
            ),
        }
        return decisions[context.request.verification_status]

    monkeypatch.setattr(OllamaCriticDecisionModel, "decide", decide)
    return contexts


def test_consistent_grounded_bundle_releases_review_result_without_network(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact deterministic policy releases only the consistent review result."""

    network_calls = 0

    def forbidden_connect(*args: object, **kwargs: object) -> None:
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("network access is forbidden in the gated integration test")

    monkeypatch.setattr(socket.socket, "connect", forbidden_connect)
    contexts = _install_happy_decisions(monkeypatch)
    critic_contexts = _install_compatible_critic_decisions(monkeypatch)
    runtime = tmp_path / "income_review_runtime"
    orchestrator = IncomeReviewOrchestrator.local(runtime)

    outcome = orchestrator.review(_bundle(synthetic_pay_stub_text, synthetic_w2_text))

    assert network_calls == 0
    assert outcome.status is WorkflowStatus.RELEASED
    assert outcome.release_allowed is True
    assert outcome.final_reason_code is (
        IncomeReviewReasonCode.CONSISTENT_INCOME_GROUNDED
    )
    assert outcome.verification_status is VerificationStatus.CONSISTENT
    assert outcome.tool_call_count == 3
    assert outcome.invalid_decision_count == 0
    assert outcome.critic_status is CriticStatus.COMPLETED
    assert outcome.critic_disposition is CriticDisposition.GROUNDED
    assert outcome.critic_reason_code is CriticReasonCode.EVIDENCE_CONSISTENT
    assert outcome.critic_attempt_count == 1
    assert outcome.critic_repair_count == 0
    assert outcome.failure_code is None
    assert set(outcome.document_types) == {DocumentType.PAY_STUB, DocumentType.TAX_FORM}
    assert len(contexts) == 4
    assert set(orchestrator.workflow_node_names) == {
        "extract_documents",
        "verification_input_assemble",
        "verification",
        "critic_input_assemble",
        "critic",
        "final_gate",
    }
    assert orchestrator._workflow._artifacts == {}

    assert len(critic_contexts) == 1
    safe_context = "".join(item.model_dump_json() for item in contexts)
    safe_context += "".join(item.model_dump_json() for item in critic_contexts)
    safe_context += outcome.model_dump_json()
    audit_text = (runtime / "audit" / "events.jsonl").read_text(encoding="utf-8")
    for forbidden in (
        synthetic_pay_stub_text,
        synthetic_w2_text,
        "SYNTHETIC PERSON ALPHA",
        "SYN-EMP-0001",
        "000-00-0001",
        "SYNTHETIC LABS LLC",
        "00-0000001",
        "$6,250.00",
        "$75,000.00",
    ):
        assert forbidden not in safe_context
        assert forbidden not in audit_text


def test_complete_income_review_has_one_private_parent_child_trace(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both documents and all six outer nodes share exactly one trace ID."""

    _install_happy_decisions(monkeypatch)
    _install_compatible_critic_decisions(monkeypatch)
    trace_sink = _CapturingTraceSink()
    orchestrator = IncomeReviewOrchestrator.local(
        tmp_path / "complete_trace",
        trace_sink=trace_sink,
    )

    outcome = orchestrator.review(_bundle(synthetic_pay_stub_text, synthetic_w2_text))

    assert outcome.status is WorkflowStatus.RELEASED
    assert len(trace_sink.spans) == 30
    assert len({span.trace_id for span in trace_sink.spans}) == 1
    roots = [span for span in trace_sink.spans if span.parent_span_id is None]
    assert len(roots) == 1
    root = roots[0]
    assert root.stage is TraceStage.FINANCIAL_REVIEW
    assert root.outcome is TraceSpanOutcome.SUCCEEDED
    span_ids = {span.span_id for span in trace_sink.spans}
    assert all(
        span.parent_span_id is None or span.parent_span_id in span_ids
        for span in trace_sink.spans
    )
    stage_counts = Counter(span.stage for span in trace_sink.spans)
    assert stage_counts[TraceStage.INCOME_REVIEW_INPUT_VALIDATION] == 1
    assert stage_counts[TraceStage.EXTRACT_DOCUMENTS] == 1
    assert stage_counts[TraceStage.VERIFICATION_INPUT_ASSEMBLE] == 1
    assert stage_counts[TraceStage.AGENT_2_VERIFICATION] == 1
    assert stage_counts[TraceStage.CRITIC_INPUT_ASSEMBLE] == 1
    assert stage_counts[TraceStage.AGENT_3_CRITIC] == 1
    assert stage_counts[TraceStage.FINAL_GATE] == 1
    assert stage_counts[TraceStage.AGENT_1_EXTRACTION] == 2
    extract_span = next(
        span
        for span in trace_sink.spans
        if span.stage is TraceStage.EXTRACT_DOCUMENTS
    )
    assert extract_span.parent_span_id == root.span_id
    assert all(
        span.parent_span_id == extract_span.span_id
        for span in trace_sink.spans
        if span.stage is TraceStage.AGENT_1_EXTRACTION
    )
    rendered = "".join(span.model_dump_json() for span in trace_sink.spans)
    rendered += outcome.model_dump_json()
    for forbidden in (
        synthetic_pay_stub_text,
        synthetic_w2_text,
        "SYNTHETIC PERSON ALPHA",
        "SYN-EMP-0001",
        "000-00-0001",
        "$6,250.00",
        "$75,000.00",
    ):
        assert forbidden not in rendered
    assert root.trace_id not in outcome.model_dump_json()


def test_invalid_bundle_trace_stops_before_agents(
    tmp_path: Path,
) -> None:
    """A pre-agent rejection still closes one sanitized root trace."""

    trace_sink = _CapturingTraceSink()
    orchestrator = IncomeReviewOrchestrator.local(
        tmp_path / "invalid_bundle_trace",
        trace_sink=trace_sink,
    )

    outcome = orchestrator.review({"documents": ()})

    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.failure_code is IncomeReviewFailureCode.INVALID_BUNDLE
    assert outcome.release_allowed is False
    assert outcome.final_reason_code is IncomeReviewReasonCode.PROCESSING_FAILURE
    assert [span.stage for span in trace_sink.spans] == [
        TraceStage.INCOME_REVIEW_INPUT_VALIDATION,
        TraceStage.FINANCIAL_REVIEW,
    ]
    validation, root = trace_sink.spans
    assert root.parent_span_id is None
    assert validation.parent_span_id == root.span_id
    assert validation.outcome is TraceSpanOutcome.REJECTED
    assert validation.causes[0].reason_code == "invalid_bundle"
    assert root.outcome is TraceSpanOutcome.HUMAN_REVIEW
    assert root.causes == validation.causes


def test_agent2_invalid_decision_retries_in_subgraph_then_releases(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One malformed Agent 2 decision loops once before the guarded happy path."""

    contexts: list[VerificationDecisionContext] = []

    def decide(
        _adapter: OllamaIncomeToolDecisionModel,
        context: VerificationDecisionContext,
    ) -> VerificationToolDecision:
        contexts.append(context)
        if len(contexts) == 1:
            raise InvalidToolDecisionError(
                InvalidToolDecisionReason.SCHEMA_VIOLATION
            )
        return _next_happy_decision(context)

    monkeypatch.setattr(OllamaIncomeToolDecisionModel, "decide", decide)
    critic_contexts = _install_compatible_critic_decisions(monkeypatch)
    runtime = tmp_path / "agent2_graph_repair"
    orchestrator = IncomeReviewOrchestrator.local(runtime)

    outcome = orchestrator.review(_bundle(synthetic_pay_stub_text, synthetic_w2_text))

    assert len(contexts) == 5
    assert len(critic_contexts) == 1
    assert outcome.status is WorkflowStatus.RELEASED
    assert outcome.release_allowed is True
    assert outcome.verification_status is VerificationStatus.CONSISTENT
    assert outcome.tool_call_count == 3
    assert outcome.invalid_decision_count == 1
    assert outcome.critic_disposition is CriticDisposition.GROUNDED
    assert outcome.failure_code is None
    audit_text = (runtime / "audit" / "events.jsonl").read_text(encoding="utf-8")
    for forbidden in (
        synthetic_pay_stub_text,
        synthetic_w2_text,
        "SYNTHETIC PERSON ALPHA",
        "000-00-0001",
        "$6,250.00",
        "$75,000.00",
    ):
        assert forbidden not in outcome.model_dump_json()
        assert forbidden not in audit_text


def test_agent2_second_invalid_decision_fails_before_critic(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second invalid decision exhausts repair and blocks Agent 3."""

    model_calls = 0

    def reject(
        _adapter: OllamaIncomeToolDecisionModel,
        _context: VerificationDecisionContext,
    ) -> VerificationToolDecision:
        nonlocal model_calls
        model_calls += 1
        raise InvalidToolDecisionError(
            InvalidToolDecisionReason.SCHEMA_VIOLATION
        )

    monkeypatch.setattr(OllamaIncomeToolDecisionModel, "decide", reject)
    critic_contexts = _install_compatible_critic_decisions(monkeypatch)
    orchestrator = IncomeReviewOrchestrator.local(
        tmp_path / "agent2_graph_repair_exhausted"
    )

    outcome = orchestrator.review(_bundle(synthetic_pay_stub_text, synthetic_w2_text))

    assert model_calls == 2
    assert critic_contexts == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.release_allowed is False
    assert outcome.failure_code is IncomeReviewFailureCode.VERIFICATION_FAILED
    assert outcome.verification_status is VerificationStatus.FAILED
    assert outcome.verification_failure_code is (
        VerificationFailureCode.INVALID_MODEL_DECISION
    )
    assert outcome.tool_call_count == 0
    assert outcome.invalid_decision_count == 2


def test_agent2_rejected_tool_call_recovers_within_four_call_ceiling(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A wrong opaque reference is rejected, observed safely, and corrected."""

    contexts: list[VerificationDecisionContext] = []

    def decide(
        _adapter: OllamaIncomeToolDecisionModel,
        context: VerificationDecisionContext,
    ) -> VerificationToolDecision:
        contexts.append(context)
        if len(contexts) == 1:
            pay_stub = next(
                item
                for item in context.evidence
                if item.document_type is DocumentType.PAY_STUB
            )
            return VerificationToolDecision(
                action=DecisionAction.CALL_TOOL,
                tool_name=VerificationToolName.NORMALIZE_W2_INCOME,
                evidence_ref=pay_stub.evidence_ref,
            )
        return _next_happy_decision(context)

    monkeypatch.setattr(OllamaIncomeToolDecisionModel, "decide", decide)
    critic_contexts = _install_compatible_critic_decisions(monkeypatch)
    orchestrator = IncomeReviewOrchestrator.local(
        tmp_path / "agent2_graph_tool_repair"
    )

    outcome = orchestrator.review(_bundle(synthetic_pay_stub_text, synthetic_w2_text))

    assert len(contexts) == 5
    assert len(critic_contexts) == 1
    assert outcome.status is WorkflowStatus.RELEASED
    assert outcome.release_allowed is True
    assert outcome.verification_status is VerificationStatus.CONSISTENT
    assert outcome.tool_call_count == 4
    assert outcome.invalid_decision_count == 1
    assert outcome.critic_disposition is CriticDisposition.GROUNDED
    assert outcome.failure_code is None


def test_agent2_tool_call_ceiling_fails_before_critic(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The outer graph preserves Agent 2's tool ceiling and blocks Agent 3."""

    model_calls = 0

    def repeat_paystub_normalization(
        _adapter: OllamaIncomeToolDecisionModel,
        context: VerificationDecisionContext,
    ) -> VerificationToolDecision:
        nonlocal model_calls
        model_calls += 1
        pay_stub = next(
            item
            for item in context.evidence
            if item.document_type is DocumentType.PAY_STUB
        )
        return VerificationToolDecision(
            action=DecisionAction.CALL_TOOL,
            tool_name=VerificationToolName.NORMALIZE_PAYSTUB_INCOME,
            evidence_ref=pay_stub.evidence_ref,
        )

    monkeypatch.setattr(
        OllamaIncomeToolDecisionModel,
        "decide",
        repeat_paystub_normalization,
    )
    critic_contexts = _install_compatible_critic_decisions(monkeypatch)
    orchestrator = IncomeReviewOrchestrator.local(
        tmp_path / "agent2_tool_limit_exhausted"
    )

    outcome = orchestrator.review(_bundle(synthetic_pay_stub_text, synthetic_w2_text))

    assert model_calls == MAX_VERIFICATION_MODEL_DECISIONS
    assert critic_contexts == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.release_allowed is False
    assert outcome.final_reason_code is IncomeReviewReasonCode.PROCESSING_FAILURE
    assert outcome.failure_code is IncomeReviewFailureCode.VERIFICATION_FAILED
    assert outcome.verification_status is VerificationStatus.FAILED
    assert outcome.verification_failure_code is (
        VerificationFailureCode.TOOL_CALL_LIMIT_REACHED
    )
    assert outcome.tool_call_count == MAX_VERIFICATION_TOOL_CALLS
    assert outcome.invalid_decision_count == 0


def test_failed_extraction_prevents_verification_agent_call(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing Agent 1 evidence routes directly to human review."""

    contexts = _install_happy_decisions(monkeypatch)
    critic_contexts = _install_compatible_critic_decisions(monkeypatch)
    trace_sink = _CapturingTraceSink()
    orchestrator = IncomeReviewOrchestrator.local(
        tmp_path / "extraction_failure",
        trace_sink=trace_sink,
    )
    malformed = synthetic_pay_stub_text.replace("Pay Period Year: 2025\n", "")

    outcome = orchestrator.review(_bundle(malformed, synthetic_w2_text))

    assert contexts == []
    assert critic_contexts == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.release_allowed is False
    extraction_span = next(
        span
        for span in trace_sink.spans
        if span.stage is TraceStage.EXTRACT_DOCUMENTS
    )
    assert extraction_span.outcome is TraceSpanOutcome.FAILED
    assert extraction_span.causes[0].reason_code == "extraction_failed"
    assert not any(
        span.stage is TraceStage.AGENT_2_VERIFICATION
        for span in trace_sink.spans
    )
    assert outcome.final_reason_code is IncomeReviewReasonCode.PROCESSING_FAILURE
    assert outcome.failure_code is IncomeReviewFailureCode.EXTRACTION_FAILED
    assert outcome.upstream_failure_codes == (FailureCode.UNSUPPORTED_REQUIRED_FIELD,)


def test_bank_statement_pair_fails_at_handoff_before_agent2(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    synthetic_bank_statement_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Successful bank extraction still cannot become payroll evidence."""

    contexts = _install_happy_decisions(monkeypatch)
    critic_contexts = _install_compatible_critic_decisions(monkeypatch)
    orchestrator = IncomeReviewOrchestrator.local(tmp_path / "bank_handoff")
    bundle = IncomeReviewBundle(
        documents=(
            _submission("synthetic_pay_stub.txt", synthetic_pay_stub_text),
            _submission(
                "synthetic_bank_statement.txt",
                synthetic_bank_statement_text,
            ),
        )
    )

    outcome = orchestrator.review(bundle)

    assert contexts == []
    assert critic_contexts == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.failure_code is IncomeReviewFailureCode.HANDOFF_FAILED
    assert outcome.handoff_failure_code is (
        VerificationHandoffFailureCode.UNSUPPORTED_DOCUMENT_COMBINATION
    )
    assert outcome.release_allowed is False
    assert outcome.final_reason_code is IncomeReviewReasonCode.PROCESSING_FAILURE


def test_agent2_operational_failure_is_preserved_and_fails_closed(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local decision failure cannot appear ready for the Critic Agent."""

    model_calls = 0

    def fail_decision(
        _adapter: OllamaIncomeToolDecisionModel,
        _context: VerificationDecisionContext,
    ) -> VerificationToolDecision:
        nonlocal model_calls
        model_calls += 1
        raise RuntimeError("sanitized test failure")

    monkeypatch.setattr(OllamaIncomeToolDecisionModel, "decide", fail_decision)
    critic_contexts = _install_compatible_critic_decisions(monkeypatch)
    trace_sink = _CapturingTraceSink()
    orchestrator = IncomeReviewOrchestrator.local(
        tmp_path / "agent2_failure",
        trace_sink=trace_sink,
    )

    outcome = orchestrator.review(_bundle(synthetic_pay_stub_text, synthetic_w2_text))

    assert model_calls == 1
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert critic_contexts == []
    assert outcome.release_allowed is False
    assert outcome.final_reason_code is IncomeReviewReasonCode.PROCESSING_FAILURE
    assert outcome.failure_code is IncomeReviewFailureCode.VERIFICATION_FAILED
    assert outcome.verification_status is VerificationStatus.FAILED
    assert outcome.verification_failure_code is (
        VerificationFailureCode.MODEL_DECISION_FAILED
    )
    verification_span = next(
        span
        for span in trace_sink.spans
        if span.stage is TraceStage.AGENT_2_VERIFICATION
    )
    assert verification_span.outcome is TraceSpanOutcome.FAILED
    assert verification_span.causes[0].reason_code == "model_decision_failed"
    assert not any(
        span.stage is TraceStage.AGENT_3_CRITIC
        for span in trace_sink.spans
    )


def test_invalid_bundle_cannot_enter_graph_or_reach_agent2(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bundle cardinality validation happens before graph or model execution."""

    contexts = _install_happy_decisions(monkeypatch)
    critic_contexts = _install_compatible_critic_decisions(monkeypatch)
    orchestrator = IncomeReviewOrchestrator.local(tmp_path / "invalid_bundle")
    invalid = {
        "documents": [
            _submission("synthetic_pay_stub.txt", synthetic_pay_stub_text),
        ]
    }

    outcome = orchestrator.review(invalid)

    assert contexts == []
    assert critic_contexts == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.failure_code is IncomeReviewFailureCode.INVALID_BUNDLE
    assert outcome.document_types == ()
    assert outcome.release_allowed is False
    assert outcome.final_reason_code is IncomeReviewReasonCode.PROCESSING_FAILURE


def test_cross_year_result_routes_to_domain_human_review(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Not-comparable evidence is a domain result, not a processing failure."""

    _install_happy_decisions(monkeypatch)
    critic_contexts = _install_compatible_critic_decisions(monkeypatch)
    older_w2 = synthetic_w2_text.replace("Tax Year: 2025", "Tax Year: 2024")
    orchestrator = IncomeReviewOrchestrator.local(tmp_path / "cross_year")

    outcome = orchestrator.review(_bundle(synthetic_pay_stub_text, older_w2))

    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.verification_status is VerificationStatus.NOT_COMPARABLE
    assert outcome.critic_disposition is CriticDisposition.ESCALATE
    assert outcome.critic_reason_code is CriticReasonCode.INCOME_NOT_COMPARABLE
    assert outcome.release_allowed is False
    assert outcome.final_reason_code is IncomeReviewReasonCode.INCOME_NOT_COMPARABLE
    assert outcome.failure_code is None
    assert len(critic_contexts) == 1


def test_inconsistent_income_routes_to_domain_human_review(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A verified amount mismatch is preserved without becoming system failure."""

    _install_happy_decisions(monkeypatch)
    critic_contexts = _install_compatible_critic_decisions(monkeypatch)
    mismatched_w2 = synthetic_w2_text.replace(
        "Annual Wages: $75,000.00",
        "Annual Wages: $72,000.00",
    )
    orchestrator = IncomeReviewOrchestrator.local(tmp_path / "inconsistent_income")

    outcome = orchestrator.review(_bundle(synthetic_pay_stub_text, mismatched_w2))

    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.release_allowed is False
    assert outcome.final_reason_code is IncomeReviewReasonCode.INCOME_INCONSISTENT
    assert outcome.verification_status is VerificationStatus.INCONSISTENT
    assert outcome.critic_disposition is CriticDisposition.ESCALATE
    assert outcome.critic_reason_code is CriticReasonCode.INCOME_INCONSISTENT
    assert outcome.failure_code is None
    assert len(critic_contexts) == 1


def test_insufficient_evidence_routes_to_domain_human_review(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid evidence-gap result reaches refuse but can never be released."""

    def insufficient(
        _agent: IncomeVerificationAgent,
        _request: object,
    ) -> IncomeVerificationResult:
        return IncomeVerificationResult(
            normalized_income=(),
            comparisons=(),
            status=VerificationStatus.INSUFFICIENT_EVIDENCE,
            evidence_complete=False,
            unsupported_reasons=(
                VerificationReason.UNSUPPORTED_DOCUMENT_COMBINATION,
            ),
            tool_call_count=0,
            invalid_decision_count=0,
        )

    monkeypatch.setattr(IncomeVerificationAgent, "verify", insufficient)
    critic_contexts = _install_compatible_critic_decisions(monkeypatch)
    orchestrator = IncomeReviewOrchestrator.local(tmp_path / "insufficient_evidence")

    outcome = orchestrator.review(_bundle(synthetic_pay_stub_text, synthetic_w2_text))

    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.release_allowed is False
    assert outcome.final_reason_code is IncomeReviewReasonCode.INSUFFICIENT_EVIDENCE
    assert outcome.verification_status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert outcome.critic_disposition is CriticDisposition.REFUSE
    assert outcome.critic_reason_code is CriticReasonCode.INSUFFICIENT_EVIDENCE
    assert outcome.failure_code is None
    assert len(critic_contexts) == 1


def test_critic_handoff_failure_prevents_agent3_call(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed deterministic trust boundary blocks the Critic model."""

    _install_happy_decisions(monkeypatch)
    critic_contexts = _install_compatible_critic_decisions(monkeypatch)

    def reject_handoff(_result: object) -> object:
        raise CriticHandoffError(
            CriticHandoffFailureCode.INVALID_COMPARISON_LINKAGE
        )

    monkeypatch.setattr(CriticInputAssembler, "assemble", reject_handoff)
    trace_sink = _CapturingTraceSink()
    orchestrator = IncomeReviewOrchestrator.local(
        tmp_path / "critic_handoff_failure",
        trace_sink=trace_sink,
    )

    outcome = orchestrator.review(_bundle(synthetic_pay_stub_text, synthetic_w2_text))

    assert critic_contexts == []
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.release_allowed is False
    assert outcome.final_reason_code is IncomeReviewReasonCode.PROCESSING_FAILURE
    assert outcome.failure_code is IncomeReviewFailureCode.CRITIC_HANDOFF_FAILED
    assert outcome.critic_handoff_failure_code is (
        CriticHandoffFailureCode.INVALID_COMPARISON_LINKAGE
    )
    stages = [span.stage for span in trace_sink.spans]
    assert TraceStage.AGENT_2_VERIFICATION in stages
    assert TraceStage.AGENT_3_CRITIC not in stages
    assert stages.index(TraceStage.AGENT_2_VERIFICATION) < stages.index(
        TraceStage.CRITIC_INPUT_ASSEMBLE
    )
    assert stages.index(TraceStage.CRITIC_INPUT_ASSEMBLE) < stages.index(
        TraceStage.FINAL_GATE
    )

    verification = next(
        span
        for span in trace_sink.spans
        if span.stage is TraceStage.AGENT_2_VERIFICATION
    )
    critic_handoff = next(
        span
        for span in trace_sink.spans
        if span.stage is TraceStage.CRITIC_INPUT_ASSEMBLE
    )
    final_gate = next(
        span for span in trace_sink.spans if span.stage is TraceStage.FINAL_GATE
    )
    root = next(
        span
        for span in trace_sink.spans
        if span.stage is TraceStage.FINANCIAL_REVIEW
    )

    assert verification.outcome is TraceSpanOutcome.SUCCEEDED
    assert critic_handoff.outcome is TraceSpanOutcome.FAILED
    assert critic_handoff.causes[0].reason_code == "invalid_comparison_linkage"
    assert final_gate.outcome is TraceSpanOutcome.FAILED
    assert final_gate.causes[0].reason_code == "critic_handoff_failed"
    assert root.outcome is TraceSpanOutcome.HUMAN_REVIEW
    assert [cause.reason_code for cause in root.causes] == [
        "invalid_comparison_linkage",
        "critic_handoff_failed",
    ]


def test_invalid_critic_output_gets_one_repair_before_release(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The graph retains Agent 3's bounded repair counts in safe state."""

    _install_happy_decisions(monkeypatch)
    critic_contexts: list[CriticDecisionContext] = []

    def decide(
        _adapter: OllamaCriticDecisionModel,
        context: CriticDecisionContext,
    ) -> CriticDecision:
        critic_contexts.append(context)
        if len(critic_contexts) == 1:
            raise InvalidCriticDecisionError(
                InvalidCriticDecisionReason.SCHEMA_VIOLATION
            )
        return CriticDecision(
            outcome=CriticDisposition.GROUNDED,
            reason_code=CriticReasonCode.EVIDENCE_CONSISTENT,
        )

    monkeypatch.setattr(OllamaCriticDecisionModel, "decide", decide)
    orchestrator = IncomeReviewOrchestrator.local(tmp_path / "critic_repair")

    outcome = orchestrator.review(_bundle(synthetic_pay_stub_text, synthetic_w2_text))

    assert outcome.status is WorkflowStatus.RELEASED
    assert outcome.release_allowed is True
    assert outcome.final_reason_code is (
        IncomeReviewReasonCode.CONSISTENT_INCOME_GROUNDED
    )
    assert outcome.critic_attempt_count == 2
    assert outcome.critic_repair_count == 1
    assert critic_contexts[1].repair_reason is CriticRepairReason.INVALID_MODEL_OUTPUT


def test_repeated_critic_contradiction_routes_to_human_review(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two contradictory Critic decisions are discarded without release."""

    _install_happy_decisions(monkeypatch)
    critic_contexts: list[CriticDecisionContext] = []

    def contradict(
        _adapter: OllamaCriticDecisionModel,
        context: CriticDecisionContext,
    ) -> CriticDecision:
        critic_contexts.append(context)
        return CriticDecision(
            outcome=CriticDisposition.ESCALATE,
            reason_code=CriticReasonCode.INCOME_INCONSISTENT,
        )

    monkeypatch.setattr(OllamaCriticDecisionModel, "decide", contradict)
    orchestrator = IncomeReviewOrchestrator.local(tmp_path / "critic_exhausted")

    outcome = orchestrator.review(_bundle(synthetic_pay_stub_text, synthetic_w2_text))

    assert len(critic_contexts) == 2
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.release_allowed is False
    assert outcome.final_reason_code is IncomeReviewReasonCode.PROCESSING_FAILURE
    assert outcome.failure_code is IncomeReviewFailureCode.CRITIC_FAILED
    assert outcome.critic_status is CriticStatus.FAILED
    assert outcome.critic_failure_code is CriticFailureCode.REPAIR_EXHAUSTED
    assert outcome.critic_attempt_count == 2
    assert outcome.critic_repair_count == 1


def test_critic_operational_failure_routes_to_human_without_repair(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A Critic transport failure is terminal and cannot trigger output repair."""

    verification_contexts = _install_happy_decisions(monkeypatch)
    critic_contexts: list[CriticDecisionContext] = []
    private_sentinel = "PRIVATE-CRITIC-TRANSPORT-DETAIL"

    def fail_decision(
        _adapter: OllamaCriticDecisionModel,
        context: CriticDecisionContext,
    ) -> CriticDecision:
        critic_contexts.append(context)
        raise RuntimeError(private_sentinel)

    monkeypatch.setattr(OllamaCriticDecisionModel, "decide", fail_decision)
    runtime = tmp_path / "critic_operational_failure"
    trace_sink = _CapturingTraceSink()
    orchestrator = IncomeReviewOrchestrator.local(
        runtime,
        trace_sink=trace_sink,
    )

    outcome = orchestrator.review(_bundle(synthetic_pay_stub_text, synthetic_w2_text))

    assert len(verification_contexts) == 4
    assert len(critic_contexts) == 1
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.release_allowed is False
    assert outcome.final_reason_code is IncomeReviewReasonCode.PROCESSING_FAILURE
    assert outcome.failure_code is IncomeReviewFailureCode.CRITIC_FAILED
    assert outcome.critic_status is CriticStatus.FAILED
    assert outcome.critic_failure_code is CriticFailureCode.MODEL_DECISION_FAILED
    assert outcome.critic_attempt_count == 1
    assert outcome.critic_repair_count == 0
    assert private_sentinel not in outcome.model_dump_json()
    assert private_sentinel not in (
        runtime / "audit" / "events.jsonl"
    ).read_text(encoding="utf-8")
    critic_span = next(
        span
        for span in trace_sink.spans
        if span.stage is TraceStage.AGENT_3_CRITIC
    )
    assert critic_span.outcome is TraceSpanOutcome.FAILED
    assert critic_span.causes[0].reason_code == "model_decision_failed"
    assert private_sentinel not in "".join(
        span.model_dump_json() for span in trace_sink.spans
    )


def test_final_gate_rejects_mismatched_private_handoff_artifact(
    tmp_path: Path,
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The gate independently rejects a changed Agent 2 → Critic contract."""

    _install_happy_decisions(monkeypatch)
    critic_contexts = _install_compatible_critic_decisions(monkeypatch)
    original_assemble = CriticInputAssembler.assemble
    assembly_count = 0

    def changing_handoff(result: IncomeVerificationResult):
        nonlocal assembly_count
        assembly_count += 1
        request = original_assemble(result)
        if assembly_count == 2:
            return request.model_copy(
                update={"tool_call_count": request.tool_call_count - 1}
            )
        return request

    monkeypatch.setattr(CriticInputAssembler, "assemble", changing_handoff)
    trace_sink = _CapturingTraceSink()
    orchestrator = IncomeReviewOrchestrator.local(
        tmp_path / "final_gate_tamper",
        trace_sink=trace_sink,
    )

    outcome = orchestrator.review(_bundle(synthetic_pay_stub_text, synthetic_w2_text))

    assert assembly_count == 2
    assert len(critic_contexts) == 1
    assert outcome.status is WorkflowStatus.HUMAN_REVIEW
    assert outcome.release_allowed is False
    assert outcome.final_reason_code is IncomeReviewReasonCode.PROCESSING_FAILURE
    assert outcome.failure_code is IncomeReviewFailureCode.FINAL_GATE_FAILED
    final_gate_span = next(
        span
        for span in trace_sink.spans
        if span.stage is TraceStage.FINAL_GATE
    )
    assert final_gate_span.outcome is TraceSpanOutcome.FAILED
    assert final_gate_span.causes[0].reason_code == "final_gate_failed"


def test_public_outcome_rejects_domain_reason_status_contradiction() -> None:
    """Closed output fields cannot describe mutually inconsistent findings."""

    with pytest.raises(ValueError):
        IncomeReviewOutcome(
            status=WorkflowStatus.HUMAN_REVIEW,
            release_allowed=False,
            final_reason_code=IncomeReviewReasonCode.INCOME_INCONSISTENT,
            document_types=(DocumentType.PAY_STUB, DocumentType.TAX_FORM),
            verification_status=VerificationStatus.NOT_COMPARABLE,
            critic_status=CriticStatus.COMPLETED,
            critic_disposition=CriticDisposition.ESCALATE,
            critic_reason_code=CriticReasonCode.INCOME_NOT_COMPARABLE,
            critic_attempt_count=1,
        )


def test_outer_graph_state_contains_only_callback_safe_control_fields() -> None:
    """Sensitive cross-agent objects remain absent from the LangGraph schema."""

    state_fields = set(IncomeReviewState.__annotations__)
    assert state_fields == {
        "run_token",
        "document_types",
        "extractions_ready",
        "verification_request_ready",
        "verification_result_ready",
        "ready_for_critic",
        "critic_request_ready",
        "critic_result_ready",
        "ready_for_final_gate",
        "failure_code",
        "upstream_failure_codes",
        "handoff_failure_code",
        "verification_status",
        "verification_failure_code",
        "critic_handoff_failure_code",
        "critic_status",
        "critic_disposition",
        "critic_reason_code",
        "critic_failure_code",
        "tool_call_count",
        "invalid_decision_count",
        "critic_attempt_count",
        "critic_repair_count",
        "review_status",
        "release_allowed",
        "final_reason_code",
    }
    assert not state_fields & {
        "document_text",
        "documents",
        "validated_extractions",
        "verification_request",
        "verification_result",
        "critic_request",
        "critic_result",
        "normalized_income",
        "provenance",
        "correlation_id",
        "document_id",
    }
