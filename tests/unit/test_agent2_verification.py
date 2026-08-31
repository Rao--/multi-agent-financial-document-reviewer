from __future__ import annotations

import json
from collections.abc import Callable
from decimal import Decimal
from typing import Any

import pytest

from financial_reviewer.agents.agent2_verification import (
    AGENT2_GRAPH_RECURSION_LIMIT,
    DecisionAction,
    IncomeBasis,
    IncomeEvidence,
    IncomePeriod,
    IncomeVerificationAgent,
    IncomeVerificationRequest,
    InvalidToolDecisionError,
    InvalidToolDecisionReason,
    MAX_VERIFICATION_MODEL_DECISIONS,
    MAX_VERIFICATION_TOOL_CALLS,
    OllamaIncomeToolDecisionModel,
    TransformationRule,
    VerificationDecisionContext,
    VerificationFailureCode,
    VerificationInputError,
    VerificationReason,
    VerificationStatus,
    VerificationToolDecision,
    VerificationToolName,
    VerificationWorkflowState,
)
from financial_reviewer.foundation.config import LocalModelSettings
from financial_reviewer.foundation.schemas import DocumentType, SourceProvenance
from financial_reviewer.local.model import OllamaModel


PAYSTUB_DOCUMENT_REF = "doc_" + ("1" * 32)
W2_DOCUMENT_REF = "doc_" + ("2" * 32)
SECOND_PAYSTUB_DOCUMENT_REF = "doc_" + ("3" * 32)
PAYSTUB_EVIDENCE_REF = "evidence_" + ("a" * 32)
W2_EVIDENCE_REF = "evidence_" + ("b" * 32)
SECOND_PAYSTUB_EVIDENCE_REF = "evidence_" + ("c" * 32)


def _provenance(document_ref: str, marker: str) -> SourceProvenance:
    """Create synthetic provenance without real document content or PII."""

    return SourceProvenance(
        document_id=document_ref,
        page_number=1,
        line_start=1,
        line_end=1,
        char_start=0,
        char_end=10,
        evidence_sha256=marker * 64,
        confidence=1.0,
    )


def _income_evidence(
    *,
    evidence_ref: str,
    document_ref: str,
    document_type: DocumentType,
    amount: str,
    period: IncomePeriod,
    basis: IncomeBasis = IncomeBasis.GROSS,
    calendar_year: int = 2025,
    marker: str,
) -> IncomeEvidence:
    """Build one validated synthetic Agent 2 input fact."""

    return IncomeEvidence(
        evidence_ref=evidence_ref,
        document_ref=document_ref,
        document_type=document_type,
        amount=Decimal(amount),
        period=period,
        income_basis=basis,
        calendar_year=calendar_year,
        provenance=(_provenance(document_ref, marker),),
    )


def _request(
    *,
    paystub_amount: str = "6000.00",
    w2_amount: str = "72000.00",
    paystub_basis: IncomeBasis = IncomeBasis.GROSS,
    w2_basis: IncomeBasis = IncomeBasis.GROSS,
    paystub_year: int = 2025,
    w2_year: int = 2025,
) -> IncomeVerificationRequest:
    """Create the supported synthetic paystub/W-2 verification request."""

    return IncomeVerificationRequest(
        evidence=(
            _income_evidence(
                evidence_ref=PAYSTUB_EVIDENCE_REF,
                document_ref=PAYSTUB_DOCUMENT_REF,
                document_type=DocumentType.PAY_STUB,
                amount=paystub_amount,
                period=IncomePeriod.MONTHLY,
                basis=paystub_basis,
                calendar_year=paystub_year,
                marker="a",
            ),
            _income_evidence(
                evidence_ref=W2_EVIDENCE_REF,
                document_ref=W2_DOCUMENT_REF,
                document_type=DocumentType.TAX_FORM,
                amount=w2_amount,
                period=IncomePeriod.ANNUAL,
                basis=w2_basis,
                calendar_year=w2_year,
                marker="b",
            ),
        )
    )


DecisionStep = (
    VerificationToolDecision
    | Exception
    | Callable[[VerificationDecisionContext], VerificationToolDecision]
    | Any
)


class _ScriptedDecisionModel:
    """Test double that records every safe context supplied by Agent 2."""

    def __init__(self, steps: list[DecisionStep]) -> None:
        self.steps = list(steps)
        self.contexts: list[VerificationDecisionContext] = []

    def decide(self, context: VerificationDecisionContext) -> VerificationToolDecision:
        self.contexts.append(context)
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        if callable(step):
            return step(context)
        return step


class _RawDecision:
    """Represent malformed model JSON that bypassed normal object construction."""

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def model_dump(self, **_: object) -> dict[str, object]:
        return self.payload


def _normalize_paystub() -> VerificationToolDecision:
    return VerificationToolDecision(
        action=DecisionAction.CALL_TOOL,
        tool_name=VerificationToolName.NORMALIZE_PAYSTUB_INCOME,
        evidence_ref=PAYSTUB_EVIDENCE_REF,
    )


def _normalize_w2() -> VerificationToolDecision:
    return VerificationToolDecision(
        action=DecisionAction.CALL_TOOL,
        tool_name=VerificationToolName.NORMALIZE_W2_INCOME,
        evidence_ref=W2_EVIDENCE_REF,
    )


def _compare(context: VerificationDecisionContext) -> VerificationToolDecision:
    refs = tuple(item.normalized_ref for item in context.normalized_income)
    assert len(refs) == 2
    return VerificationToolDecision(
        action=DecisionAction.CALL_TOOL,
        tool_name=VerificationToolName.COMPARE_INCOME_SOURCES,
        left_normalized_ref=refs[0],
        right_normalized_ref=refs[1],
    )


def _complete() -> VerificationToolDecision:
    return VerificationToolDecision(action=DecisionAction.COMPLETE)


def _happy_steps() -> list[DecisionStep]:
    return [_normalize_paystub(), _normalize_w2(), _compare, _complete()]


def test_agent2_calls_three_explicit_tools_and_preserves_provenance() -> None:
    model = _ScriptedDecisionModel(_happy_steps())

    result = IncomeVerificationAgent(model).verify(_request())

    assert result.status is VerificationStatus.CONSISTENT
    assert result.evidence_complete is True
    assert result.tool_call_count == 3
    assert result.invalid_decision_count == 0
    assert len(result.normalized_income) == 2
    assert {item.monthly_amount for item in result.normalized_income} == {
        Decimal("6000.00")
    }
    assert {item.transformation_rule for item in result.normalized_income} == {
        TransformationRule.PAYSTUB_MONTHLY_V1,
        TransformationRule.W2_ANNUAL_TO_MONTHLY_V1,
    }
    assert all(item.provenance for item in result.normalized_income)
    assert result.comparisons[0].amount_difference == Decimal("0.00")
    assert result.comparisons[0].reason_code is VerificationReason.EXACT_MATCH

    serialized_contexts = "\n".join(context.model_dump_json() for context in model.contexts)
    assert '"amount"' not in serialized_contexts
    assert '"provenance"' not in serialized_contexts
    assert "6000.00" not in serialized_contexts
    assert "72000.00" not in serialized_contexts


def test_agent2_exposes_five_node_graph_and_keeps_state_callback_safe() -> None:
    """The agentic loop is a real graph without financial data in graph state."""

    agent = IncomeVerificationAgent(_ScriptedDecisionModel(_happy_steps()))

    result = agent.verify(_request())

    assert result.status is VerificationStatus.CONSISTENT
    assert set(agent.workflow_node_names) == {
        "validate_request",
        "model_decision",
        "decision_guard",
        "execute_tool",
        "finalize",
    }
    assert agent._artifacts == {}
    assert AGENT2_GRAPH_RECURSION_LIMIT == 20
    assert set(VerificationWorkflowState.__annotations__) == {
        "run_token",
        "request_validated",
        "model_decision_count",
        "tool_call_count",
        "invalid_decision_count",
        "decision_ready",
        "result_ready",
        "next_step",
        "failure_code",
    }
    forbidden_state_fields = {
        "amount",
        "provenance",
        "request",
        "normalized_income",
        "comparison",
        "prompt",
        "model_output",
        "observations",
        "decision",
    }
    assert forbidden_state_fields.isdisjoint(VerificationWorkflowState.__annotations__)
    mermaid = agent._graph.get_graph().draw_mermaid()
    assert "model_decision --> decision_guard" in mermaid
    assert "decision_guard -.-> model_decision" in mermaid
    assert "decision_guard -.-> execute_tool" in mermaid
    assert "execute_tool -.-> model_decision" in mermaid


def test_agent2_detects_exact_policy_mismatch_with_deterministic_delta() -> None:
    result = IncomeVerificationAgent(_ScriptedDecisionModel(_happy_steps())).verify(
        _request(w2_amount="69600.00")
    )

    assert result.status is VerificationStatus.INCONSISTENT
    comparison = result.comparisons[0]
    assert comparison.amount_difference == Decimal("200.00")
    assert comparison.percentage_difference == Decimal("3.33")
    assert comparison.reason_code is VerificationReason.INCOME_VALUES_INCONSISTENT


@pytest.mark.parametrize(
    ("verification_request", "reason"),
    [
        (
            _request(w2_basis=IncomeBasis.TAXABLE),
            VerificationReason.INCOME_BASIS_NOT_COMPARABLE,
        ),
        (
            _request(w2_year=2024),
            VerificationReason.INCOME_PERIOD_NOT_COMPARABLE,
        ),
    ],
)
def test_agent2_refuses_incompatible_sources(
    verification_request: IncomeVerificationRequest,
    reason: VerificationReason,
) -> None:
    result = IncomeVerificationAgent(_ScriptedDecisionModel(_happy_steps())).verify(
        verification_request
    )

    assert result.status is VerificationStatus.NOT_COMPARABLE
    assert result.comparisons[0].reason_code is reason
    assert result.comparisons[0].amount_difference is None
    assert result.comparisons[0].percentage_difference is None
    assert result.unsupported_reasons == (reason,)


def test_agent2_marks_unsupported_document_pair_without_tool_calls() -> None:
    request = IncomeVerificationRequest(
        evidence=(
            _request().evidence[0],
            _income_evidence(
                evidence_ref=SECOND_PAYSTUB_EVIDENCE_REF,
                document_ref=SECOND_PAYSTUB_DOCUMENT_REF,
                document_type=DocumentType.PAY_STUB,
                amount="6000.00",
                period=IncomePeriod.MONTHLY,
                marker="c",
            ),
        )
    )

    result = IncomeVerificationAgent(
        _ScriptedDecisionModel([_complete()])
    ).verify(request)

    assert result.status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert result.tool_call_count == 0
    assert result.unsupported_reasons == (
        VerificationReason.UNSUPPORTED_DOCUMENT_COMBINATION,
    )


def test_agent2_allows_one_rejected_tool_call_then_recovers() -> None:
    invalid_call = VerificationToolDecision(
        action=DecisionAction.CALL_TOOL,
        tool_name=VerificationToolName.NORMALIZE_W2_INCOME,
        evidence_ref=PAYSTUB_EVIDENCE_REF,
    )
    model = _ScriptedDecisionModel([invalid_call, *_happy_steps()])

    result = IncomeVerificationAgent(model).verify(_request())

    assert result.status is VerificationStatus.CONSISTENT
    assert result.tool_call_count == MAX_VERIFICATION_TOOL_CALLS
    assert result.invalid_decision_count == 1
    assert len(model.contexts) == MAX_VERIFICATION_MODEL_DECISIONS


def test_agent2_retries_one_malformed_model_decision_only() -> None:
    malformed = _RawDecision(
        {"action": "call_tool", "tool_name": "delete_document"}
    )
    result = IncomeVerificationAgent(
        _ScriptedDecisionModel([malformed, *_happy_steps()])
    ).verify(_request())

    assert result.status is VerificationStatus.CONSISTENT
    assert result.invalid_decision_count == 1
    assert result.tool_call_count == 3


def test_agent2_fails_closed_after_second_invalid_decision() -> None:
    malformed = _RawDecision(
        {"action": "call_tool", "tool_name": "delete_document"}
    )
    result = IncomeVerificationAgent(
        _ScriptedDecisionModel([malformed, malformed])
    ).verify(_request())

    assert result.status is VerificationStatus.FAILED
    assert result.failure_code is VerificationFailureCode.INVALID_MODEL_DECISION
    assert result.evidence_complete is False
    assert result.normalized_income == ()
    assert result.comparisons == ()


def test_agent2_enforces_tool_call_ceiling() -> None:
    repeated_calls = [_normalize_paystub()] * MAX_VERIFICATION_MODEL_DECISIONS
    model = _ScriptedDecisionModel(repeated_calls)

    result = IncomeVerificationAgent(model).verify(_request())

    assert result.status is VerificationStatus.FAILED
    assert result.failure_code is VerificationFailureCode.TOOL_CALL_LIMIT_REACHED
    assert result.tool_call_count == MAX_VERIFICATION_TOOL_CALLS
    assert len(model.contexts) == MAX_VERIFICATION_MODEL_DECISIONS


def test_agent2_sanitizes_model_failure_and_releases_no_partial_values() -> None:
    private_sentinel = "PRIVATE-MODEL-FAILURE-SENTINEL"
    result = IncomeVerificationAgent(
        _ScriptedDecisionModel([RuntimeError(private_sentinel)])
    ).verify(_request())

    assert result.status is VerificationStatus.FAILED
    assert result.failure_code is VerificationFailureCode.MODEL_DECISION_FAILED
    assert private_sentinel not in repr(result)
    assert result.normalized_income == ()


def test_invalid_constructed_request_never_reaches_decision_model() -> None:
    model = _ScriptedDecisionModel([_complete()])
    invalid = IncomeVerificationRequest.model_construct(
        evidence=(_request().evidence[0],)
    )

    with pytest.raises(VerificationInputError):
        IncomeVerificationAgent(model).verify(invalid)

    assert model.contexts == []


def _decision_from_safe_prompt(prompt: str) -> str:
    """Return the next synthetic model decision from amount-free prompt state."""

    marker = "Safe decision context JSON: "
    context = json.loads(prompt.split(marker, maxsplit=1)[1])
    normalized = context["normalized_income"]
    observations = context["observations"]
    normalized_sources = {item["source_evidence_ref"] for item in normalized}

    if len(normalized) < 2:
        evidence = next(
            item
            for item in context["evidence"]
            if item["evidence_ref"] not in normalized_sources
        )
        tool_name = (
            "normalize_paystub_income"
            if evidence["document_type"] == "pay_stub"
            else "normalize_w2_income"
        )
        return json.dumps(
            {
                "decision": {
                    "action": tool_name,
                    "evidence_ref": evidence["evidence_ref"],
                }
            }
        )

    comparison_seen = any(
        item.get("comparison_result") is not None for item in observations
    )
    if not comparison_seen:
        return json.dumps(
            {
                "decision": {
                    "action": "compare_income_sources",
                    "left_normalized_ref": normalized[0]["normalized_ref"],
                    "right_normalized_ref": normalized[1]["normalized_ref"],
                }
            }
        )
    return json.dumps({"decision": {"action": "complete"}})


def test_agent2_ollama_adapter_drives_full_tool_loop_without_financial_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_prompts: list[str] = []

    def generate_structured(
        _model: OllamaModel,
        task_name: str,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> str:
        assert task_name == "agent2_income_verification"
        assert response_schema["additionalProperties"] is False
        captured_prompts.append(prompt)
        return _decision_from_safe_prompt(prompt)

    monkeypatch.setattr(OllamaModel, "generate_structured", generate_structured)
    adapter = OllamaIncomeToolDecisionModel(OllamaModel(LocalModelSettings()))

    result = IncomeVerificationAgent(adapter).verify(_request())

    assert result.status is VerificationStatus.CONSISTENT
    assert result.tool_call_count == 3
    assert len(captured_prompts) == 4
    joined = "\n".join(captured_prompts)
    assert "DO:" in joined
    assert "DO NOT:" in joined
    assert "include no other fields" in joined
    assert '"amount"' not in joined
    assert '"provenance"' not in joined
    assert "6000.00" not in joined
    assert "72000.00" not in joined


def test_agent2_retries_one_invalid_ollama_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def generate_structured(
        _model: OllamaModel,
        task_name: str,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return '{"decision":{"action":"delete_document"}}'
        return _decision_from_safe_prompt(prompt)

    monkeypatch.setattr(OllamaModel, "generate_structured", generate_structured)
    adapter = OllamaIncomeToolDecisionModel(OllamaModel(LocalModelSettings()))

    result = IncomeVerificationAgent(adapter).verify(_request())

    assert result.status is VerificationStatus.CONSISTENT
    assert result.invalid_decision_count == 1
    assert result.tool_call_count == 3
    assert calls == MAX_VERIFICATION_MODEL_DECISIONS


def test_ollama_adapter_rejects_duplicate_decision_keys_without_echoing_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "PRIVATE-DUPLICATE-OUTPUT-SENTINEL"

    def generate_structured(
        _model: OllamaModel,
        task_name: str,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> str:
        return (
            '{"decision":{"action":"complete","action":"complete","unexpected":"'
            + sentinel
            + '"}}'
        )

    monkeypatch.setattr(OllamaModel, "generate_structured", generate_structured)
    adapter = OllamaIncomeToolDecisionModel(OllamaModel(LocalModelSettings()))
    context = IncomeVerificationAgent._decision_context(_request(), {}, [], 0)

    with pytest.raises(InvalidToolDecisionError) as captured:
        adapter.decide(context)

    assert sentinel not in str(captured.value)
    assert captured.value.reason is InvalidToolDecisionReason.DUPLICATE_JSON_KEY
    assert captured.value.invalid_fields == ()


def test_agent2_ollama_adapter_rejects_unapproved_transport() -> None:
    class UnapprovedTransport:
        def generate_structured(
            self,
            task_name: str,
            prompt: str,
            response_schema: dict[str, Any],
        ) -> str:
            return "{}"

    with pytest.raises(TypeError):
        OllamaIncomeToolDecisionModel(UnapprovedTransport())  # type: ignore[arg-type]


def test_ollama_adapter_reports_only_safe_argument_shape_on_schema_violation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def generate_structured(
        _model: OllamaModel,
        task_name: str,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> str:
        return json.dumps(
            {
                "decision": {
                    "action": "complete",
                    "left_normalized_ref": "norm_" + ("1" * 32),
                    "right_normalized_ref": "norm_" + ("2" * 32),
                }
            }
        )

    monkeypatch.setattr(OllamaModel, "generate_structured", generate_structured)
    adapter = OllamaIncomeToolDecisionModel(OllamaModel(LocalModelSettings()))
    context = IncomeVerificationAgent._decision_context(_request(), {}, [], 0)

    with pytest.raises(InvalidToolDecisionError) as captured:
        adapter.decide(context)

    assert captured.value.reason is InvalidToolDecisionReason.SCHEMA_VIOLATION
    assert captured.value.proposed_action == "complete"
    assert captured.value.proposed_tool_name is None
    assert captured.value.non_null_argument_fields == (
        "left_normalized_ref",
        "right_normalized_ref",
    )
