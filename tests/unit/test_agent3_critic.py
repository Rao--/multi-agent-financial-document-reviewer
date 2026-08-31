"""Unit tests for the bounded, fail-closed Critic Agent."""

from __future__ import annotations

import json
from typing import Any

import pytest

from financial_reviewer.agents.agent2_verification import (
    ComparisonResult,
    IncomeBasis,
    TransformationRule,
    VerificationFailureCode,
    VerificationReason,
    VerificationStatus,
)
from financial_reviewer.agents.agent3_critic import (
    MAX_CRITIC_MODEL_ATTEMPTS,
    CriticComparisonSummary,
    CriticDecision,
    CriticDecisionContext,
    CriticDisposition,
    CriticFailureCode,
    CriticInputError,
    CriticReasonCode,
    CriticRepairReason,
    CriticReviewRequest,
    CriticSourceSummary,
    CriticStatus,
    IncomeReviewCriticAgent,
    InvalidCriticDecisionError,
    InvalidCriticDecisionReason,
    OllamaCriticDecisionModel,
)
from financial_reviewer.foundation.config import LocalModelSettings
from financial_reviewer.foundation.schemas import DocumentType
from financial_reviewer.local.model import OllamaModel


def _sources(
    *,
    pay_stub_year: int = 2025,
    w2_year: int = 2025,
) -> tuple[CriticSourceSummary, CriticSourceSummary]:
    """Build amount-free summaries for the supported source pair."""

    return (
        CriticSourceSummary(
            document_type=DocumentType.PAY_STUB,
            income_basis=IncomeBasis.GROSS,
            calendar_year=pay_stub_year,
            transformation_rule=TransformationRule.PAYSTUB_MONTHLY_V1,
            provenance_pointer_count=1,
        ),
        CriticSourceSummary(
            document_type=DocumentType.TAX_FORM,
            income_basis=IncomeBasis.GROSS,
            calendar_year=w2_year,
            transformation_rule=TransformationRule.W2_ANNUAL_TO_MONTHLY_V1,
            provenance_pointer_count=1,
        ),
    )


def _request(status: VerificationStatus) -> CriticReviewRequest:
    """Create one internally consistent safe summary for each Agent 2 status."""

    if status is VerificationStatus.CONSISTENT:
        comparison = CriticComparisonSummary(
            result=ComparisonResult.CONSISTENT,
            reason_code=VerificationReason.EXACT_MATCH,
        )
    elif status is VerificationStatus.INCONSISTENT:
        comparison = CriticComparisonSummary(
            result=ComparisonResult.INCONSISTENT,
            reason_code=VerificationReason.INCOME_VALUES_INCONSISTENT,
        )
    elif status is VerificationStatus.NOT_COMPARABLE:
        comparison = CriticComparisonSummary(
            result=ComparisonResult.NOT_COMPARABLE,
            reason_code=VerificationReason.INCOME_PERIOD_NOT_COMPARABLE,
        )
    else:
        comparison = None

    if status is VerificationStatus.FAILED:
        return CriticReviewRequest(
            verification_status=status,
            evidence_complete=False,
            verification_failure_code=VerificationFailureCode.MODEL_DECISION_FAILED,
            tool_call_count=0,
            invalid_decision_count=0,
        )
    return CriticReviewRequest(
        verification_status=status,
        evidence_complete=status is not VerificationStatus.INSUFFICIENT_EVIDENCE,
        sources=() if status is VerificationStatus.INSUFFICIENT_EVIDENCE else _sources(),
        comparison=comparison,
        tool_call_count=3 if comparison is not None else 0,
        invalid_decision_count=0,
    )


class _ScriptedCriticModel:
    """Record safe contexts and return scripted decisions or failures."""

    def __init__(self, steps: list[object]) -> None:
        self.steps = list(steps)
        self.contexts: list[CriticDecisionContext] = []

    def decide(self, context: CriticDecisionContext) -> CriticDecision:
        self.contexts.append(context)
        if not self.steps:
            raise AssertionError("unexpected critic model call")
        step = self.steps.pop(0)
        if isinstance(step, Exception):
            raise step
        return step  # type: ignore[return-value]


@pytest.mark.parametrize(
    ("verification_status", "outcome", "reason"),
    [
        (
            VerificationStatus.CONSISTENT,
            CriticDisposition.GROUNDED,
            CriticReasonCode.EVIDENCE_CONSISTENT,
        ),
        (
            VerificationStatus.INCONSISTENT,
            CriticDisposition.ESCALATE,
            CriticReasonCode.INCOME_INCONSISTENT,
        ),
        (
            VerificationStatus.NOT_COMPARABLE,
            CriticDisposition.ESCALATE,
            CriticReasonCode.INCOME_NOT_COMPARABLE,
        ),
        (
            VerificationStatus.INSUFFICIENT_EVIDENCE,
            CriticDisposition.REFUSE,
            CriticReasonCode.INSUFFICIENT_EVIDENCE,
        ),
    ],
)
def test_critic_accepts_only_compatible_decision_pairs(
    verification_status: VerificationStatus,
    outcome: CriticDisposition,
    reason: CriticReasonCode,
) -> None:
    """Each approved Agent 2 status maps to one exact critic recommendation."""

    model = _ScriptedCriticModel(
        [CriticDecision(outcome=outcome, reason_code=reason)]
    )

    result = IncomeReviewCriticAgent(model).critique(_request(verification_status))

    assert result.status is CriticStatus.COMPLETED
    assert result.decision == CriticDecision(outcome=outcome, reason_code=reason)
    assert result.attempt_count == 1
    assert result.repair_count == 0
    assert result.failure_code is None
    assert model.contexts[0].repair_reason is None


def test_grounded_inconsistent_decision_gets_one_repair() -> None:
    """Prompt wording cannot clear a deterministic income inconsistency."""

    model = _ScriptedCriticModel(
        [
            CriticDecision(
                outcome=CriticDisposition.GROUNDED,
                reason_code=CriticReasonCode.EVIDENCE_CONSISTENT,
            ),
            CriticDecision(
                outcome=CriticDisposition.ESCALATE,
                reason_code=CriticReasonCode.INCOME_INCONSISTENT,
            ),
        ]
    )

    result = IncomeReviewCriticAgent(model).critique(
        _request(VerificationStatus.INCONSISTENT)
    )

    assert result.status is CriticStatus.COMPLETED
    assert result.decision is not None
    assert result.decision.outcome is CriticDisposition.ESCALATE
    assert result.attempt_count == 2
    assert result.repair_count == 1
    assert model.contexts[1].repair_reason is (
        CriticRepairReason.CONTRADICTORY_DECISION
    )


def test_repeated_contradiction_exhausts_repair_and_returns_no_decision() -> None:
    """A second incompatible response is discarded instead of released."""

    contradiction = CriticDecision(
        outcome=CriticDisposition.GROUNDED,
        reason_code=CriticReasonCode.EVIDENCE_CONSISTENT,
    )
    model = _ScriptedCriticModel([contradiction, contradiction])

    result = IncomeReviewCriticAgent(model).critique(
        _request(VerificationStatus.INCONSISTENT)
    )

    assert result.status is CriticStatus.FAILED
    assert result.decision is None
    assert result.failure_code is CriticFailureCode.REPAIR_EXHAUSTED
    assert result.attempt_count == MAX_CRITIC_MODEL_ATTEMPTS
    assert result.repair_count == 1


def test_invalid_model_object_gets_one_repair() -> None:
    """Shape validation occurs again after the decision-model boundary."""

    model = _ScriptedCriticModel(
        [
            object(),
            CriticDecision(
                outcome=CriticDisposition.GROUNDED,
                reason_code=CriticReasonCode.EVIDENCE_CONSISTENT,
            ),
        ]
    )

    result = IncomeReviewCriticAgent(model).critique(
        _request(VerificationStatus.CONSISTENT)
    )

    assert result.status is CriticStatus.COMPLETED
    assert result.attempt_count == 2
    assert model.contexts[1].repair_reason is CriticRepairReason.INVALID_MODEL_OUTPUT


def test_model_operational_failure_does_not_trigger_output_repair() -> None:
    """The repair budget is for bad decisions, not transport failures."""

    model = _ScriptedCriticModel([RuntimeError("private transport detail")])

    result = IncomeReviewCriticAgent(model).critique(
        _request(VerificationStatus.CONSISTENT)
    )

    assert result.status is CriticStatus.FAILED
    assert result.failure_code is CriticFailureCode.MODEL_DECISION_FAILED
    assert result.attempt_count == 1
    assert result.repair_count == 0
    assert len(model.contexts) == 1


def test_failed_upstream_verification_never_calls_critic_model() -> None:
    """Operational Agent 2 failure cannot be transformed into a critic opinion."""

    model = _ScriptedCriticModel([])

    result = IncomeReviewCriticAgent(model).critique(
        _request(VerificationStatus.FAILED)
    )

    assert model.contexts == []
    assert result.status is CriticStatus.FAILED
    assert result.failure_code is CriticFailureCode.UPSTREAM_VERIFICATION_FAILED
    assert result.attempt_count == 0


def test_constructed_invalid_input_is_rejected_before_model() -> None:
    """A caller cannot bypass request consistency with model_construct."""

    model = _ScriptedCriticModel([])
    invalid = CriticReviewRequest.model_construct(
        verification_status=VerificationStatus.CONSISTENT,
        evidence_complete=False,
        sources=(),
        comparison=None,
        verification_failure_code=None,
        tool_call_count=0,
        invalid_decision_count=0,
    )

    with pytest.raises(CriticInputError):
        IncomeReviewCriticAgent(model).critique(invalid)

    assert model.contexts == []


def test_ollama_adapter_uses_distinct_safe_initial_and_repair_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both local calls contain only the approved amount-free critic contract."""

    calls: list[tuple[str, str]] = []

    def generate_structured(
        _model: OllamaModel,
        task_name: str,
        prompt: str,
        response_schema: dict[str, Any],
    ) -> str:
        assert response_schema["additionalProperties"] is False
        calls.append((task_name, prompt))
        return json.dumps(
            {
                "decision": {
                    "outcome": "grounded",
                    "reason_code": "evidence_consistent",
                }
            }
        )

    monkeypatch.setattr(OllamaModel, "generate_structured", generate_structured)
    adapter = OllamaCriticDecisionModel(OllamaModel(LocalModelSettings()))
    request = _request(VerificationStatus.CONSISTENT)

    initial = adapter.decide(
        CriticDecisionContext(request=request, attempt_number=1)
    )
    repaired = adapter.decide(
        CriticDecisionContext(
            request=request,
            attempt_number=2,
            repair_reason=CriticRepairReason.CONTRADICTORY_DECISION,
        )
    )

    assert initial == repaired
    assert calls[0][0] == "agent3_income_critic_initial"
    assert calls[1][0] == "agent3_income_critic_repair"
    assert "DO:" in calls[0][1]
    assert "DO NOT:" in calls[0][1]
    assert "only repair attempt" in calls[1][1]
    assert "contradictory_decision" in calls[1][1]
    rendered_prompts = "\n".join(prompt for _, prompt in calls)
    for forbidden in (
        "monthly_amount",
        "amount_difference",
        "percentage_difference",
        "document_ref",
        "evidence_ref",
        "normalized_ref",
        "provenance\"",
        "6250.00",
        "75000.00",
    ):
        assert forbidden not in rendered_prompts


def test_ollama_adapter_rejects_duplicate_keys_without_echoing_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raw invalid model output never crosses the sanitized adapter error."""

    sentinel = "PRIVATE-CRITIC-OUTPUT-SENTINEL"

    def generate_structured(
        _model: OllamaModel,
        _task_name: str,
        _prompt: str,
        _response_schema: dict[str, Any],
    ) -> str:
        return (
            '{"decision":{"outcome":"grounded","outcome":"escalate",'
            '"reason_code":"evidence_consistent","private":"'
            + sentinel
            + '"}}'
        )

    monkeypatch.setattr(OllamaModel, "generate_structured", generate_structured)
    adapter = OllamaCriticDecisionModel(OllamaModel(LocalModelSettings()))
    context = CriticDecisionContext(
        request=_request(VerificationStatus.CONSISTENT),
        attempt_number=1,
    )

    with pytest.raises(InvalidCriticDecisionError) as captured:
        adapter.decide(context)

    assert captured.value.reason is InvalidCriticDecisionReason.DUPLICATE_JSON_KEY
    assert sentinel not in str(captured.value)


def test_ollama_critic_adapter_rejects_unapproved_transport() -> None:
    """Production Agent 3 cannot be constructed around an arbitrary model SDK."""

    class UnapprovedTransport:
        def generate_structured(
            self,
            task_name: str,
            prompt: str,
            response_schema: dict[str, Any],
        ) -> str:
            return "{}"

    with pytest.raises(TypeError):
        OllamaCriticDecisionModel(UnapprovedTransport())  # type: ignore[arg-type]


def test_critic_reports_whether_it_uses_the_approved_local_adapter() -> None:
    """S3 orchestration can reject test seams at the production boundary."""

    scripted = IncomeReviewCriticAgent(
        _ScriptedCriticModel(
            [
                CriticDecision(
                    outcome=CriticDisposition.GROUNDED,
                    reason_code=CriticReasonCode.EVIDENCE_CONSISTENT,
                )
            ]
        )
    )
    local = IncomeReviewCriticAgent(
        OllamaCriticDecisionModel(OllamaModel(LocalModelSettings()))
    )

    assert scripted.uses_approved_local_adapter is False
    assert local.uses_approved_local_adapter is True
