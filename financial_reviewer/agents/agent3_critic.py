"""Implement the bounded local Critic Agent for income verification.

Why this file exists:
    Agent 2 produces a typed deterministic comparison, but an independent critic
    should challenge whether the proposed disposition agrees with that evidence.

What it owns:
    A PII-free critic input contract, closed decision/reason vocabulary, local
    Ollama structured-output adapter, one repair attempt, deterministic decision
    compatibility checks, and fail-closed result contracts.

What it does not own:
    It does not receive raw documents, names, identifiers, financial amounts, or
    source quotes. It has no tools, does not repeat Agent 1/2 work, does not route
    LangGraph, and cannot release a financial decision.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import Field, StrictBool, StrictInt, ValidationError, model_validator

from financial_reviewer.agents.agent2_verification import (
    COMPARISON_POLICY_ID,
    MAX_INVALID_DECISIONS,
    MAX_VERIFICATION_TOOL_CALLS,
    ComparisonResult,
    IncomeBasis,
    TransformationRule,
    VerificationFailureCode,
    VerificationReason,
    VerificationStatus,
)
from financial_reviewer.foundation.schemas import DocumentType, StrictModel
from financial_reviewer.local.model import LocalModelError, OllamaModel


MAX_CRITIC_MODEL_ATTEMPTS = 2
MAX_CRITIC_REPAIR_ATTEMPTS = 1
CRITIC_PROMPT_VERSION = "agent3-income-critic-v1"
_INITIAL_TASK_NAME = "agent3_income_critic_initial"
_REPAIR_TASK_NAME = "agent3_income_critic_repair"


class CriticDisposition(str, Enum):
    """Closed recommendations the Critic Agent may propose."""

    GROUNDED = "grounded"
    REFUSE = "refuse"
    ESCALATE = "escalate"


class CriticReasonCode(str, Enum):
    """Closed explanations paired with critic recommendations."""

    EVIDENCE_CONSISTENT = "evidence_consistent"
    INCOME_INCONSISTENT = "income_inconsistent"
    INCOME_NOT_COMPARABLE = "income_not_comparable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class CriticRepairReason(str, Enum):
    """PII-free feedback allowed into the single repair prompt."""

    INVALID_MODEL_OUTPUT = "invalid_model_output"
    CONTRADICTORY_DECISION = "contradictory_decision"


class CriticStatus(str, Enum):
    """Whether Agent 3 produced one deterministically accepted decision."""

    COMPLETED = "completed"
    FAILED = "failed"


class CriticFailureCode(str, Enum):
    """Operational failures separated from financial-review recommendations."""

    UPSTREAM_VERIFICATION_FAILED = "upstream_verification_failed"
    MODEL_DECISION_FAILED = "model_decision_failed"
    REPAIR_EXHAUSTED = "repair_exhausted"


class InvalidCriticDecisionReason(str, Enum):
    """Sanitized parsing failures raised by the local adapter."""

    INVALID_CONTEXT = "invalid_context"
    MALFORMED_JSON = "malformed_json"
    DUPLICATE_JSON_KEY = "duplicate_json_key"
    SCHEMA_VIOLATION = "schema_violation"


class CriticSourceSummary(StrictModel):
    """Amount-free description of one normalized, provenance-backed source."""

    document_type: DocumentType
    income_basis: IncomeBasis
    calendar_year: Annotated[StrictInt, Field(ge=2000, le=2100)]
    transformation_rule: TransformationRule
    provenance_pointer_count: Annotated[StrictInt, Field(ge=1, le=100)]


class CriticComparisonSummary(StrictModel):
    """Amount-free deterministic comparison supplied by Agent 2."""

    result: ComparisonResult
    reason_code: VerificationReason
    policy_id: Literal["synthetic_exact_v1"] = COMPARISON_POLICY_ID

    @model_validator(mode="after")
    def validate_result_reason_pair(self) -> "CriticComparisonSummary":
        """Require Agent 2's comparison result and reason to agree."""

        allowed = {
            ComparisonResult.CONSISTENT: {VerificationReason.EXACT_MATCH},
            ComparisonResult.INCONSISTENT: {
                VerificationReason.INCOME_VALUES_INCONSISTENT
            },
            ComparisonResult.NOT_COMPARABLE: {
                VerificationReason.INCOME_BASIS_NOT_COMPARABLE,
                VerificationReason.INCOME_PERIOD_NOT_COMPARABLE,
            },
        }
        if self.reason_code not in allowed[self.result]:
            raise ValueError("comparison result and reason are inconsistent")
        return self


class CriticReviewRequest(StrictModel):
    """Complete PII-free input contract accepted by the Critic Agent."""

    verification_status: VerificationStatus
    evidence_complete: StrictBool
    sources: Annotated[tuple[CriticSourceSummary, ...], Field(max_length=2)] = ()
    comparison: CriticComparisonSummary | None = None
    verification_failure_code: VerificationFailureCode | None = None
    tool_call_count: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_VERIFICATION_TOOL_CALLS),
    ]
    invalid_decision_count: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_INVALID_DECISIONS + 1),
    ]

    @model_validator(mode="after")
    def validate_upstream_contract(self) -> "CriticReviewRequest":
        """Reject contradictory Agent 2 summaries before any critic model call."""

        comparison_status = {
            VerificationStatus.CONSISTENT: ComparisonResult.CONSISTENT,
            VerificationStatus.INCONSISTENT: ComparisonResult.INCONSISTENT,
            VerificationStatus.NOT_COMPARABLE: ComparisonResult.NOT_COMPARABLE,
        }
        if self.verification_status is VerificationStatus.FAILED:
            if (
                self.verification_failure_code is None
                or self.evidence_complete
                or self.comparison is not None
            ):
                raise ValueError("failed verification summary is inconsistent")
            return self
        if self.verification_failure_code is not None:
            raise ValueError("domain verification cannot carry a failure code")
        if self.verification_status in comparison_status:
            if (
                not self.evidence_complete
                or len(self.sources) != 2
                or self.comparison is None
                or self.comparison.result
                is not comparison_status[self.verification_status]
                or {item.document_type for item in self.sources}
                != {DocumentType.PAY_STUB, DocumentType.TAX_FORM}
            ):
                raise ValueError("comparison verification summary is inconsistent")
        elif self.verification_status is VerificationStatus.INSUFFICIENT_EVIDENCE:
            if self.evidence_complete or self.comparison is not None:
                raise ValueError("insufficient-evidence summary is inconsistent")
        return self


class CriticDecision(StrictModel):
    """One structured recommendation proposed by the local critic model."""

    outcome: CriticDisposition
    reason_code: CriticReasonCode


class CriticDecisionContext(StrictModel):
    """Safe context supplied to either the initial or repair model call."""

    request: CriticReviewRequest
    attempt_number: Annotated[
        StrictInt,
        Field(ge=1, le=MAX_CRITIC_MODEL_ATTEMPTS),
    ]
    repair_reason: CriticRepairReason | None = None

    @model_validator(mode="after")
    def validate_attempt_kind(self) -> "CriticDecisionContext":
        """Allow repair feedback only on the second and final attempt."""

        if (self.attempt_number == 1) != (self.repair_reason is None):
            raise ValueError("critic attempt and repair reason are inconsistent")
        return self


class CriticResult(StrictModel):
    """Validated Agent 3 result retained privately until final orchestration."""

    status: CriticStatus
    decision: CriticDecision | None = None
    attempt_count: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_CRITIC_MODEL_ATTEMPTS),
    ]
    repair_count: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_CRITIC_REPAIR_ATTEMPTS),
    ]
    failure_code: CriticFailureCode | None = None

    @model_validator(mode="after")
    def validate_result_contract(self) -> "CriticResult":
        """Prevent completed and failed critic states from being conflated."""

        if self.status is CriticStatus.COMPLETED:
            if self.decision is None or self.failure_code is not None or self.attempt_count < 1:
                raise ValueError("completed critic result is inconsistent")
        elif self.decision is not None or self.failure_code is None:
            raise ValueError("failed critic result is inconsistent")
        if self.repair_count != max(0, self.attempt_count - 1):
            raise ValueError("critic attempt and repair counts are inconsistent")
        return self


class _CriticDecisionEnvelope(StrictModel):
    """Exact top-level schema requested from local Ollama."""

    decision: CriticDecision


class InvalidCriticDecisionError(RuntimeError):
    """Sanitized rejection that never retains raw local-model output."""

    def __init__(
        self,
        reason: InvalidCriticDecisionReason,
        invalid_fields: tuple[str, ...] = (),
    ) -> None:
        """Expose only closed parser diagnostics."""

        self.reason = reason
        self.invalid_fields = invalid_fields
        super().__init__("The local model returned an invalid Critic Agent decision")


class CriticInputError(ValueError):
    """Sanitized input rejection raised before any critic model call."""


@runtime_checkable
class CriticDecisionModel(Protocol):
    """Injected decision boundary implemented by local Ollama in production."""

    def decide(self, context: CriticDecisionContext) -> CriticDecision:
        """Return one structured initial or repaired critic recommendation."""


class OllamaCriticDecisionModel:
    """Translate safe critic context into one local structured decision."""

    __slots__ = ("_model",)

    def __init__(self, model: OllamaModel) -> None:
        """Require the repository's immutable loopback-only Ollama adapter."""

        if type(model) is not OllamaModel:
            raise TypeError("Critic Agent requires the approved local Ollama adapter")
        self._model = model

    def decide(self, context: CriticDecisionContext) -> CriticDecision:
        """Call the explicit initial or repair prompt and parse safe JSON."""

        try:
            safe_context = CriticDecisionContext.model_validate(
                context.model_dump(mode="python", warnings="none")
            )
        except (AttributeError, TypeError, ValueError, ValidationError):
            raise InvalidCriticDecisionError(
                InvalidCriticDecisionReason.INVALID_CONTEXT
            ) from None

        if safe_context.repair_reason is None:
            task_name = _INITIAL_TASK_NAME
            prompt = self._build_initial_prompt(safe_context)
        else:
            task_name = _REPAIR_TASK_NAME
            prompt = self._build_repair_prompt(safe_context)
        try:
            raw_output = self._model.generate_structured(
                task_name,
                prompt,
                _CriticDecisionEnvelope.model_json_schema(),
            )
        except LocalModelError:
            raise

        def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
            """Reject ambiguous objects without keeping the last repeated key."""

            parsed: dict[str, object] = {}
            for key, value in pairs:
                if key in parsed:
                    raise InvalidCriticDecisionError(
                        InvalidCriticDecisionReason.DUPLICATE_JSON_KEY
                    )
                parsed[key] = value
            return parsed

        try:
            payload = json.loads(raw_output, object_pairs_hook=reject_duplicate_keys)
        except InvalidCriticDecisionError:
            raise
        except json.JSONDecodeError:
            raise InvalidCriticDecisionError(
                InvalidCriticDecisionReason.MALFORMED_JSON
            ) from None
        try:
            envelope = _CriticDecisionEnvelope.model_validate(payload)
        except (TypeError, ValueError, ValidationError) as error:
            invalid_fields: tuple[str, ...] = ()
            if isinstance(error, ValidationError):
                invalid_fields = tuple(
                    sorted(
                        {
                            str(item["loc"][-1]) if item["loc"] else "decision"
                            for item in error.errors(include_input=False)
                        }
                    )
                )
            raise InvalidCriticDecisionError(
                InvalidCriticDecisionReason.SCHEMA_VIOLATION,
                invalid_fields,
            ) from None
        return envelope.decision

    @staticmethod
    def _build_initial_prompt(context: CriticDecisionContext) -> str:
        """Build the first explicit DO/DON'T prompt from amount-free state."""

        request_json = context.request.model_dump_json()
        return (
            "You are the local Income Review Critic Agent.\n"
            f"Prompt version: {CRITIC_PROMPT_VERSION}\n"
            "Review the validated, amount-free Verification Agent summary.\n"
            "DO:\n"
            "- Return grounded/evidence_consistent only for consistent evidence.\n"
            "- Return escalate/income_inconsistent for inconsistent evidence.\n"
            "- Return escalate/income_not_comparable for not-comparable evidence.\n"
            "- Return refuse/insufficient_evidence for insufficient evidence.\n"
            "DO NOT:\n"
            "- Recalculate income, invent evidence, or override the verification status.\n"
            "- Return grounded for inconsistent, not-comparable, or insufficient evidence.\n"
            "- Include explanations, identifiers, undeclared fields, or markdown.\n"
            "Return only JSON matching the supplied response schema.\n"
            f"Safe critic request JSON: {request_json}"
        )

    @staticmethod
    def _build_repair_prompt(context: CriticDecisionContext) -> str:
        """Build the single repair prompt with only a closed rejection reason."""

        request_json = context.request.model_dump_json()
        return (
            "You are repairing one rejected local Critic Agent decision.\n"
            f"Prompt version: {CRITIC_PROMPT_VERSION}\n"
            f"Closed rejection reason: {context.repair_reason.value}\n"
            "DO return exactly the disposition/reason pair required by the supplied "
            "verification status.\n"
            "DO NOT repeat the rejected contradiction, invent evidence, add fields, "
            "include prose, or change the verification status.\n"
            "This is the only repair attempt. Return only JSON matching the schema.\n"
            f"Safe critic request JSON: {request_json}"
        )


_COMPATIBLE_DECISIONS: dict[
    VerificationStatus,
    tuple[CriticDisposition, CriticReasonCode],
] = {
    VerificationStatus.CONSISTENT: (
        CriticDisposition.GROUNDED,
        CriticReasonCode.EVIDENCE_CONSISTENT,
    ),
    VerificationStatus.INCONSISTENT: (
        CriticDisposition.ESCALATE,
        CriticReasonCode.INCOME_INCONSISTENT,
    ),
    VerificationStatus.NOT_COMPARABLE: (
        CriticDisposition.ESCALATE,
        CriticReasonCode.INCOME_NOT_COMPARABLE,
    ),
    VerificationStatus.INSUFFICIENT_EVIDENCE: (
        CriticDisposition.REFUSE,
        CriticReasonCode.INSUFFICIENT_EVIDENCE,
    ),
}


class IncomeReviewCriticAgent:
    """Run at most one initial and one repaired local critic decision."""

    __slots__ = ("_decision_model",)

    def __init__(self, decision_model: CriticDecisionModel) -> None:
        """Bind a decision model without granting tools or release authority."""

        if not isinstance(decision_model, CriticDecisionModel):
            raise TypeError("critic decision model must satisfy CriticDecisionModel")
        self._decision_model = decision_model

    @property
    def uses_approved_local_adapter(self) -> bool:
        """Report whether production is bound to sealed local Ollama."""

        return type(self._decision_model) is OllamaCriticDecisionModel

    def critique(self, request: CriticReviewRequest) -> CriticResult:
        """Validate input, enforce one repair, and return no unguarded decision."""

        safe_request = self._revalidate_request(request)
        if safe_request.verification_status is VerificationStatus.FAILED:
            return CriticResult(
                status=CriticStatus.FAILED,
                attempt_count=0,
                repair_count=0,
                failure_code=CriticFailureCode.UPSTREAM_VERIFICATION_FAILED,
            )

        repair_reason: CriticRepairReason | None = None
        for attempt_number in range(1, MAX_CRITIC_MODEL_ATTEMPTS + 1):
            context = CriticDecisionContext(
                request=safe_request,
                attempt_number=attempt_number,
                repair_reason=repair_reason,
            )
            try:
                proposed = self._decision_model.decide(context)
            except InvalidCriticDecisionError:
                repair_reason = CriticRepairReason.INVALID_MODEL_OUTPUT
                continue
            except Exception:
                return CriticResult(
                    status=CriticStatus.FAILED,
                    attempt_count=attempt_number,
                    repair_count=attempt_number - 1,
                    failure_code=CriticFailureCode.MODEL_DECISION_FAILED,
                )
            try:
                decision = CriticDecision.model_validate(
                    proposed.model_dump(mode="python", warnings="none")
                )
            except (AttributeError, TypeError, ValueError, ValidationError):
                repair_reason = CriticRepairReason.INVALID_MODEL_OUTPUT
                continue
            if self._is_compatible(safe_request.verification_status, decision):
                return CriticResult(
                    status=CriticStatus.COMPLETED,
                    decision=decision,
                    attempt_count=attempt_number,
                    repair_count=attempt_number - 1,
                )
            repair_reason = CriticRepairReason.CONTRADICTORY_DECISION

        return CriticResult(
            status=CriticStatus.FAILED,
            attempt_count=MAX_CRITIC_MODEL_ATTEMPTS,
            repair_count=MAX_CRITIC_REPAIR_ATTEMPTS,
            failure_code=CriticFailureCode.REPAIR_EXHAUSTED,
        )

    @staticmethod
    def _revalidate_request(request: CriticReviewRequest) -> CriticReviewRequest:
        """Reject constructed or mutated summaries before consulting the model."""

        try:
            return CriticReviewRequest.model_validate(
                request.model_dump(mode="python", warnings="none")
            )
        except (AttributeError, TypeError, ValueError, ValidationError):
            raise CriticInputError("critic input is invalid") from None

    @staticmethod
    def _is_compatible(
        verification_status: VerificationStatus,
        decision: CriticDecision,
    ) -> bool:
        """Apply the authoritative decision table after every model response."""

        expected = _COMPATIBLE_DECISIONS.get(verification_status)
        return expected == (decision.outcome, decision.reason_code)


@runtime_checkable
class CriticAgent(Protocol):
    """Orchestrator-facing protocol implemented by the bounded critic."""

    def critique(self, request: CriticReviewRequest) -> CriticResult:
        """Return one validated critic result without release authority."""


__all__ = [
    "CRITIC_PROMPT_VERSION",
    "MAX_CRITIC_MODEL_ATTEMPTS",
    "MAX_CRITIC_REPAIR_ATTEMPTS",
    "CriticAgent",
    "CriticComparisonSummary",
    "CriticDecision",
    "CriticDecisionContext",
    "CriticDecisionModel",
    "CriticDisposition",
    "CriticFailureCode",
    "CriticInputError",
    "CriticReasonCode",
    "CriticRepairReason",
    "CriticResult",
    "CriticReviewRequest",
    "CriticSourceSummary",
    "CriticStatus",
    "IncomeReviewCriticAgent",
    "InvalidCriticDecisionError",
    "InvalidCriticDecisionReason",
    "OllamaCriticDecisionModel",
]
