"""Verify income across validated document evidence with bounded local tools.

Why this file exists:
    Agent 2 owns one narrow responsibility: normalize comparable income facts
    and determine whether the normalized values agree. It does not extract
    document text, assess broader financial risk, orchestrate other agents, or
    authorize release.

How the agent is intentionally bounded:
    A decision model sees only opaque references and non-sensitive metadata.
    It may select three allowlisted deterministic tools. Monetary values and
    source provenance stay in this agent's private execution context. The
    agent permits at most four tool calls and one invalid-decision retry.

Current integration status:
    The public ``verify`` boundary is connected to the Financial Review
    Orchestrator. Internally, a five-node LangGraph owns Agent 2's bounded
    model-decision and tool-execution loop.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Annotated, Literal, Protocol, Sequence, TypedDict, runtime_checkable
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from pydantic import (
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)

from financial_reviewer.foundation.schemas import (
    DocumentType,
    SourceProvenance,
    StrictModel,
    ValidatedExtraction,
)
from financial_reviewer.local.model import LocalModelError, OllamaModel


# The happy path uses three tools: normalize the paystub, normalize the W-2,
# and compare them. The fourth slot permits one recoverable rejected tool call.
MAX_VERIFICATION_TOOL_CALLS = 4
# COMPLETE is a model decision but not a tool call.
MAX_VERIFICATION_MODEL_DECISIONS = 5
# Only one malformed, premature, or rejected decision may be retried.
MAX_INVALID_DECISIONS = 1
# The longest legal path executes 16 application nodes: request validation,
# five model decisions, five decision guards, four tools, and finalization.
# This separate ceiling protects against an accidental future graph cycle; it
# does not increase the model, tool, or repair budgets above.
AGENT2_GRAPH_RECURSION_LIMIT = 20
# No business tolerance is invented for the first synthetic increment.
COMPARISON_POLICY_ID = "synthetic_exact_v1"
AGENT2_DECISION_PROMPT_VERSION = "agent2-tool-selection-v1"
_OLLAMA_TASK_NAME = "agent2_income_verification"
_CENT = Decimal("0.01")
_PERCENT = Decimal("0.01")


class IncomeBasis(str, Enum):
    """Supported meanings of an income amount."""

    GROSS = "gross"
    TAXABLE = "taxable"


class IncomePeriod(str, Enum):
    """Source period supported by the first paystub/W-2 slice."""

    MONTHLY = "monthly"
    ANNUAL = "annual"


class VerificationToolName(str, Enum):
    """Closed set of local tools Agent 2 may request."""

    NORMALIZE_PAYSTUB_INCOME = "normalize_paystub_income"
    NORMALIZE_W2_INCOME = "normalize_w2_income"
    COMPARE_INCOME_SOURCES = "compare_income_sources"


class VerificationStatus(str, Enum):
    """Domain result or operational failure returned by Agent 2."""

    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    NOT_COMPARABLE = "not_comparable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    FAILED = "failed"


class ComparisonResult(str, Enum):
    """Deterministic result produced by the comparison tool."""

    CONSISTENT = "consistent"
    INCONSISTENT = "inconsistent"
    NOT_COMPARABLE = "not_comparable"


class VerificationReason(str, Enum):
    """Closed explanations for deterministic verification outcomes."""

    EXACT_MATCH = "exact_match"
    INCOME_VALUES_INCONSISTENT = "income_values_inconsistent"
    INCOME_BASIS_NOT_COMPARABLE = "income_basis_not_comparable"
    INCOME_PERIOD_NOT_COMPARABLE = "income_period_not_comparable"
    UNSUPPORTED_DOCUMENT_COMBINATION = "unsupported_document_combination"


class VerificationFailureCode(str, Enum):
    """PII-free operational failures separate from financial risk findings."""

    MODEL_DECISION_FAILED = "model_decision_failed"
    INVALID_MODEL_DECISION = "invalid_model_decision"
    TOOL_CALL_LIMIT_REACHED = "tool_call_limit_reached"


class TransformationRule(str, Enum):
    """Versioned formula recorded for every normalized amount."""

    PAYSTUB_MONTHLY_V1 = "paystub_monthly_v1"
    W2_ANNUAL_TO_MONTHLY_V1 = "w2_annual_to_monthly_v1"


class ToolObservationStatus(str, Enum):
    """Sanitized status returned after a tool request."""

    SUCCEEDED = "succeeded"
    REJECTED = "rejected"


class ToolObservationCode(str, Enum):
    """Non-sensitive feedback allowed into the next model decision."""

    TOOL_COMPLETED = "tool_completed"
    INVALID_DECISION = "invalid_decision"
    INVALID_TOOL_ARGUMENTS = "invalid_tool_arguments"
    PREMATURE_COMPLETION = "premature_completion"


class InvalidToolDecisionReason(str, Enum):
    """Closed diagnostics for rejected local-model decision output."""

    INVALID_CONTEXT = "invalid_context"
    MALFORMED_JSON = "malformed_json"
    DUPLICATE_JSON_KEY = "duplicate_json_key"
    SCHEMA_VIOLATION = "schema_violation"


class DecisionAction(str, Enum):
    """The model may request a tool or declare its work complete."""

    CALL_TOOL = "call_tool"
    COMPLETE = "complete"


class VerificationToolDefinition(StrictModel):
    """Safe declaration presented to a future local tool-selection model."""

    name: VerificationToolName
    description: StrictStr
    required_arguments: tuple[StrictStr, ...]


# Declarations describe callable capabilities; the executor still authorizes
# every selected name and resolves every opaque reference.
VERIFICATION_TOOL_DECLARATIONS: tuple[VerificationToolDefinition, ...] = (
    VerificationToolDefinition(
        name=VerificationToolName.NORMALIZE_PAYSTUB_INCOME,
        description="Normalize one validated monthly paystub income fact.",
        required_arguments=("evidence_ref",),
    ),
    VerificationToolDefinition(
        name=VerificationToolName.NORMALIZE_W2_INCOME,
        description="Convert one validated annual W-2 income fact to monthly income.",
        required_arguments=("evidence_ref",),
    ),
    VerificationToolDefinition(
        name=VerificationToolName.COMPARE_INCOME_SOURCES,
        description="Compare two normalized income facts under the exact policy.",
        required_arguments=("left_normalized_ref", "right_normalized_ref"),
    ),
)


class IncomeEvidence(StrictModel):
    """One validated Agent 1 income fact accepted by Agent 2.

    Amount and provenance are private business data. They never appear in the
    decision context or sanitized tool observations.
    """

    evidence_ref: Annotated[StrictStr, Field(pattern=r"^evidence_[0-9a-f]{32}$")]
    document_ref: Annotated[StrictStr, Field(pattern=r"^doc_[0-9a-f]{32}$")]
    document_type: DocumentType
    amount: Annotated[Decimal, Field(ge=Decimal("0"), repr=False)]
    period: IncomePeriod
    income_basis: IncomeBasis
    calendar_year: Annotated[StrictInt, Field(ge=2000, le=2100)]
    provenance: Annotated[tuple[SourceProvenance, ...], Field(min_length=1)] = Field(
        repr=False
    )

    @model_validator(mode="after")
    def validate_provenance_document(self) -> "IncomeEvidence":
        """Require every provenance pointer to target this evidence's document."""

        if any(source.document_id != self.document_ref for source in self.provenance):
            raise ValueError("income evidence provenance targets another document")
        return self


class IncomeVerificationRequest(StrictModel):
    """First-slice request containing exactly two validated income sources."""

    evidence: Annotated[tuple[IncomeEvidence, ...], Field(min_length=2, max_length=2)]

    @model_validator(mode="after")
    def validate_unique_references(self) -> "IncomeVerificationRequest":
        """Reject duplicate evidence identifiers before consulting the model."""

        references = [item.evidence_ref for item in self.evidence]
        if len(set(references)) != len(references):
            raise ValueError("income evidence references must be unique")
        return self


class EvidenceDescriptor(StrictModel):
    """Non-sensitive evidence metadata visible to the decision model."""

    evidence_ref: StrictStr
    document_type: DocumentType
    period: IncomePeriod
    income_basis: IncomeBasis
    calendar_year: StrictInt


class NormalizedIncomeDescriptor(StrictModel):
    """Opaque normalized-result metadata visible to the decision model."""

    normalized_ref: StrictStr
    source_evidence_ref: StrictStr
    document_type: DocumentType


class VerificationToolObservation(StrictModel):
    """PII-free feedback from the guarded tool executor."""

    status: ToolObservationStatus
    code: ToolObservationCode
    tool_name: VerificationToolName | None = None
    produced_ref: StrictStr | None = None
    comparison_result: ComparisonResult | None = None


class VerificationDecisionContext(StrictModel):
    """Complete safe context supplied to each model decision."""

    tools: tuple[VerificationToolDefinition, ...]
    evidence: tuple[EvidenceDescriptor, ...]
    normalized_income: tuple[NormalizedIncomeDescriptor, ...]
    observations: tuple[VerificationToolObservation, ...]
    remaining_tool_calls: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_VERIFICATION_TOOL_CALLS),
    ]


class VerificationToolDecision(StrictModel):
    """Structured decision proposed by the local model.

    Only opaque references are accepted. The executor resolves them against
    private maps and never trusts model-supplied financial values.
    """

    action: DecisionAction
    tool_name: VerificationToolName | None = None
    evidence_ref: StrictStr | None = None
    left_normalized_ref: StrictStr | None = None
    right_normalized_ref: StrictStr | None = None

    @model_validator(mode="after")
    def validate_action_arguments(self) -> "VerificationToolDecision":
        """Require exactly the arguments owned by the selected action/tool."""

        if self.action is DecisionAction.COMPLETE:
            if any(
                value is not None
                for value in (
                    self.tool_name,
                    self.evidence_ref,
                    self.left_normalized_ref,
                    self.right_normalized_ref,
                )
            ):
                raise ValueError("complete decisions cannot contain tool arguments")
            return self

        if self.tool_name is None:
            raise ValueError("tool call requires an allowlisted tool name")
        if self.tool_name in {
            VerificationToolName.NORMALIZE_PAYSTUB_INCOME,
            VerificationToolName.NORMALIZE_W2_INCOME,
        }:
            if self.evidence_ref is None or any(
                value is not None
                for value in (self.left_normalized_ref, self.right_normalized_ref)
            ):
                raise ValueError("normalization tool arguments are invalid")
        elif self.tool_name is VerificationToolName.COMPARE_INCOME_SOURCES:
            if (
                self.evidence_ref is not None
                or self.left_normalized_ref is None
                or self.right_normalized_ref is None
                or self.left_normalized_ref == self.right_normalized_ref
            ):
                raise ValueError("comparison tool arguments are invalid")
        return self


@runtime_checkable
class ToolDecisionModel(Protocol):
    """Injected boundary implemented by local Ollama during integration."""

    def decide(self, context: VerificationDecisionContext) -> VerificationToolDecision:
        """Return one structured tool call or COMPLETE decision."""


class InvalidToolDecisionError(RuntimeError):
    """Sanitized signal that local-model output was not a valid tool decision."""

    def __init__(
        self,
        reason: InvalidToolDecisionReason,
        invalid_fields: tuple[str, ...] = (),
        proposed_action: str | None = None,
        proposed_tool_name: str | None = None,
        non_null_argument_fields: tuple[str, ...] = (),
    ) -> None:
        super().__init__("The local model returned an invalid Agent 2 decision")
        self.reason = reason
        self.invalid_fields = invalid_fields
        self.proposed_action = proposed_action
        self.proposed_tool_name = proposed_tool_name
        self.non_null_argument_fields = non_null_argument_fields


class _DuplicateModelOutputKeyError(ValueError):
    """Internal parser signal that never carries the duplicate value."""


class _NormalizePaystubModelDecision(StrictModel):
    """Model-facing schema requiring a paystub evidence reference."""

    action: Literal["normalize_paystub_income"]
    evidence_ref: StrictStr


class _NormalizeW2ModelDecision(StrictModel):
    """Model-facing schema requiring a W-2 evidence reference."""

    action: Literal["normalize_w2_income"]
    evidence_ref: StrictStr


class _CompareIncomeModelDecision(StrictModel):
    """Model-facing schema requiring two distinct normalized references."""

    action: Literal["compare_income_sources"]
    left_normalized_ref: StrictStr
    right_normalized_ref: StrictStr

    @model_validator(mode="after")
    def validate_distinct_references(self) -> "_CompareIncomeModelDecision":
        """Reject a comparison of one normalized result with itself."""

        if self.left_normalized_ref == self.right_normalized_ref:
            raise ValueError("comparison references must be distinct")
        return self


class _CompleteModelDecision(StrictModel):
    """Model-facing terminal decision that accepts no tool arguments."""

    action: Literal["complete"]


_ModelDecision = Annotated[
    _NormalizePaystubModelDecision
    | _NormalizeW2ModelDecision
    | _CompareIncomeModelDecision
    | _CompleteModelDecision,
    Field(discriminator="action"),
]


class _ModelDecisionEnvelope(StrictModel):
    """Exact schema-constrained response requested from local Ollama."""

    decision: _ModelDecision


class OllamaIncomeToolDecisionModel:
    """Translate safe Agent 2 state into one local Qwen tool decision.

    This adapter owns the Agent 2 prompt and response parsing. It cannot execute
    tools directly; ``IncomeVerificationAgent`` remains the authorization and
    execution boundary.
    """

    __slots__ = ("_model",)

    def __init__(self, model: OllamaModel) -> None:
        """Require the repository's sealed loopback-only Ollama transport."""

        if type(model) is not OllamaModel:
            raise TypeError("Agent 2 requires the approved local Ollama model adapter")
        self._model = model

    def decide(self, context: VerificationDecisionContext) -> VerificationToolDecision:
        """Request and validate one structured decision without logging content."""

        try:
            safe_context = VerificationDecisionContext.model_validate(
                context.model_dump(mode="python", warnings="none")
            )
        except (AttributeError, TypeError, ValueError, ValidationError):
            raise InvalidToolDecisionError(
                InvalidToolDecisionReason.INVALID_CONTEXT
            ) from None

        prompt = self._build_prompt(safe_context)
        try:
            raw_output = self._model.generate_structured(
                _OLLAMA_TASK_NAME,
                prompt,
                _ModelDecisionEnvelope.model_json_schema(),
            )
        except LocalModelError:
            raise

        def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
            """Reject ambiguous model JSON without retaining its contents."""

            parsed: dict[str, object] = {}
            for key, value in pairs:
                if key in parsed:
                    raise _DuplicateModelOutputKeyError("duplicate model-output key")
                parsed[key] = value
            return parsed

        try:
            payload = json.loads(raw_output, object_pairs_hook=reject_duplicate_keys)
        except json.JSONDecodeError:
            raise InvalidToolDecisionError(
                InvalidToolDecisionReason.MALFORMED_JSON
            ) from None
        except _DuplicateModelOutputKeyError:
            raise InvalidToolDecisionError(
                InvalidToolDecisionReason.DUPLICATE_JSON_KEY
            ) from None
        try:
            envelope = _ModelDecisionEnvelope.model_validate(payload)
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
            payload_dict = payload if isinstance(payload, dict) else {}
            raw_decision = payload_dict.get("decision")
            decision_dict = raw_decision if isinstance(raw_decision, dict) else {}
            raw_action = decision_dict.get("action")
            proposed_action = (
                DecisionAction.COMPLETE.value
                if raw_action == "complete"
                else DecisionAction.CALL_TOOL.value
                if raw_action in {item.value for item in VerificationToolName}
                else "unrecognized"
            )
            proposed_tool_name = (
                raw_action
                if raw_action in {item.value for item in VerificationToolName}
                else None
            )
            argument_names = (
                "evidence_ref",
                "left_normalized_ref",
                "right_normalized_ref",
            )
            non_null_argument_fields = tuple(
                name for name in argument_names if decision_dict.get(name) is not None
            )
            raise InvalidToolDecisionError(
                InvalidToolDecisionReason.SCHEMA_VIOLATION,
                invalid_fields,
                proposed_action=proposed_action,
                proposed_tool_name=proposed_tool_name,
                non_null_argument_fields=non_null_argument_fields,
            ) from None

        decision = envelope.decision
        if isinstance(decision, _NormalizePaystubModelDecision):
            return VerificationToolDecision(
                action=DecisionAction.CALL_TOOL,
                tool_name=VerificationToolName.NORMALIZE_PAYSTUB_INCOME,
                evidence_ref=decision.evidence_ref,
            )
        if isinstance(decision, _NormalizeW2ModelDecision):
            return VerificationToolDecision(
                action=DecisionAction.CALL_TOOL,
                tool_name=VerificationToolName.NORMALIZE_W2_INCOME,
                evidence_ref=decision.evidence_ref,
            )
        if isinstance(decision, _CompareIncomeModelDecision):
            return VerificationToolDecision(
                action=DecisionAction.CALL_TOOL,
                tool_name=VerificationToolName.COMPARE_INCOME_SOURCES,
                left_normalized_ref=decision.left_normalized_ref,
                right_normalized_ref=decision.right_normalized_ref,
            )
        return VerificationToolDecision(action=DecisionAction.COMPLETE)

    @staticmethod
    def _build_prompt(context: VerificationDecisionContext) -> str:
        """Build the explicit DO/DON'T contract around amount-free JSON state."""

        context_json = context.model_dump_json()
        return (
            "You are the local Income Verification Agent tool selector.\n"
            f"Prompt version: {AGENT2_DECISION_PROMPT_VERSION}\n"
            "Return one top-level decision object and choose exactly one next action "
            "from the supplied structured context.\n"
            "DO:\n"
            "- Call normalize_paystub_income for an unnormalized pay_stub reference.\n"
            "- Call normalize_w2_income for an unnormalized tax_form reference; "
            "tax_form represents the synthetic W-2 in this slice.\n"
            "- After both sources are normalized, call compare_income_sources with "
            "their exact normalized_ref values.\n"
            "- Return complete only after a comparison observation exists, or when "
            "the evidence does not contain one pay_stub and one tax_form.\n"
            "- When decision.action is complete, include no other fields inside "
            "the decision object.\n"
            "DO NOT:\n"
            "- Calculate, estimate, or provide financial values.\n"
            "- Invent or alter evidence_ref or normalized_ref values.\n"
            "- Select an undeclared tool or include undeclared fields.\n"
            "- Repeat a successful tool call.\n"
            "Return only JSON matching the supplied response schema.\n"
            f"Safe decision context JSON: {context_json}"
        )


class NormalizedIncome(StrictModel):
    """Private monthly income with complete derivation provenance."""

    normalized_ref: Annotated[StrictStr, Field(pattern=r"^norm_[0-9a-f]{32}$")]
    document_ref: StrictStr
    document_type: DocumentType
    monthly_amount: Decimal = Field(repr=False)
    income_basis: IncomeBasis
    calendar_year: StrictInt
    source_evidence_ref: StrictStr
    provenance: Annotated[tuple[SourceProvenance, ...], Field(min_length=1)] = Field(
        repr=False
    )
    transformation_rule: TransformationRule


class IncomeComparison(StrictModel):
    """Private deterministic comparison derived from normalized facts."""

    left_income_ref: StrictStr
    right_income_ref: StrictStr
    amount_difference: Decimal | None = Field(repr=False)
    percentage_difference: Decimal | None = Field(repr=False)
    result: ComparisonResult
    reason_code: VerificationReason
    policy_id: Literal["synthetic_exact_v1"] = COMPARISON_POLICY_ID


class IncomeVerificationResult(StrictModel):
    """Typed Agent 2 artifact held privately until graph integration."""

    normalized_income: tuple[NormalizedIncome, ...] = Field(repr=False)
    comparisons: tuple[IncomeComparison, ...] = Field(repr=False)
    status: VerificationStatus
    evidence_complete: StrictBool
    unsupported_reasons: tuple[VerificationReason, ...]
    tool_call_count: Annotated[StrictInt, Field(ge=0, le=MAX_VERIFICATION_TOOL_CALLS)]
    invalid_decision_count: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_INVALID_DECISIONS + 1),
    ]
    failure_code: VerificationFailureCode | None = None

    @model_validator(mode="after")
    def validate_result_consistency(self) -> "IncomeVerificationResult":
        """Prevent contradictory status, comparison, and failure combinations."""

        if self.status is VerificationStatus.FAILED:
            if self.failure_code is None or self.evidence_complete:
                raise ValueError("failed verification requires a failure code")
        elif self.failure_code is not None:
            raise ValueError("domain verification result cannot carry a failure code")

        comparison_statuses = {
            VerificationStatus.CONSISTENT,
            VerificationStatus.INCONSISTENT,
            VerificationStatus.NOT_COMPARABLE,
        }
        if self.status in comparison_statuses and len(self.comparisons) != 1:
            raise ValueError("comparison status requires exactly one comparison")
        if self.status is VerificationStatus.INSUFFICIENT_EVIDENCE and self.comparisons:
            raise ValueError("insufficient evidence cannot carry a comparison")
        return self


class VerificationInputError(ValueError):
    """Sanitized input rejection raised before the decision model is called."""


class _ToolExecutionError(RuntimeError):
    """Internal sanitized signal for a rejected model-selected tool call."""


class VerificationWorkflowState(TypedDict, total=False):
    """Callback-safe control state for Agent 2's internal LangGraph.

    Financial amounts, provenance, decisions, observations, and results are
    deliberately absent. Nodes resolve those objects from a private run-scoped
    artifact using only the random ``run_token`` carried here.
    """

    run_token: str
    request_validated: bool
    model_decision_count: int
    tool_call_count: int
    invalid_decision_count: int
    decision_ready: bool
    result_ready: bool
    next_step: Literal["model_decision", "execute_tool", "finalize"]
    failure_code: VerificationFailureCode | None


class _CallbackSafeVerificationState(StrictModel):
    """Runtime validator for every field allowed into Agent 2's graph state."""

    run_token: StrictStr = Field(pattern=r"^run_[0-9a-f]{32}$")
    request_validated: StrictBool = False
    model_decision_count: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_VERIFICATION_MODEL_DECISIONS),
    ] = 0
    tool_call_count: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_VERIFICATION_TOOL_CALLS),
    ] = 0
    invalid_decision_count: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_INVALID_DECISIONS + 1),
    ] = 0
    decision_ready: StrictBool = False
    result_ready: StrictBool = False
    next_step: Literal["model_decision", "execute_tool", "finalize"] = (
        "model_decision"
    )
    failure_code: VerificationFailureCode | None = None


@dataclass
class _VerificationRunArtifacts:
    """Private financial data and decisions for exactly one Agent 2 run."""

    request: object = field(repr=False)
    validated_request: IncomeVerificationRequest | None = field(default=None, repr=False)
    private_evidence: dict[str, IncomeEvidence] = field(default_factory=dict, repr=False)
    normalized: dict[str, NormalizedIncome] = field(default_factory=dict, repr=False)
    comparisons: list[IncomeComparison] = field(default_factory=list, repr=False)
    observations: list[VerificationToolObservation] = field(
        default_factory=list,
        repr=False,
    )
    proposed_decision: object | None = field(default=None, repr=False)
    decision_error: Literal["invalid", "operational"] | None = None
    input_error: bool = False
    result: IncomeVerificationResult | None = field(default=None, repr=False)


class IncomeVerificationAgent:
    """Run the bounded observe-select-tool loop as an internal LangGraph."""

    __slots__ = ("_artifact_lock", "_artifacts", "_decision_model", "_graph")

    def __init__(self, decision_model: ToolDecisionModel) -> None:
        """Bind one decision model without granting it direct tool access."""

        if not isinstance(decision_model, ToolDecisionModel):
            raise TypeError("decision model must implement the ToolDecisionModel protocol")
        self._decision_model = decision_model
        self._artifact_lock = threading.RLock()
        self._artifacts: dict[str, _VerificationRunArtifacts] = {}
        self._graph = self._build_graph()

    @property
    def uses_approved_local_adapter(self) -> bool:
        """Report whether production orchestration is bound to local Ollama.

        Standalone unit tests may inject a scripted decision model. The public
        multi-agent orchestrator rejects that test seam and accepts only the
        sealed loopback ``OllamaIncomeToolDecisionModel`` adapter.
        """

        return type(self._decision_model) is OllamaIncomeToolDecisionModel

    @property
    def workflow_node_names(self) -> tuple[str, ...]:
        """Expose Agent 2's application nodes for diagrams and regression tests."""

        nodes = self._graph.get_graph().nodes
        return tuple(
            name
            for name in nodes
            if name not in {START, END, "__start__", "__end__"}
        )

    def verify(self, request: IncomeVerificationRequest) -> IncomeVerificationResult:
        """Invoke the graph and delete every private artifact on all exit paths."""

        run_token = f"run_{uuid4().hex}"
        with self._artifact_lock:
            self._artifacts[run_token] = _VerificationRunArtifacts(request=request)
        initial = _CallbackSafeVerificationState(run_token=run_token).model_dump(
            mode="python",
            warnings="none",
        )
        try:
            final_state = self._graph.invoke(
                initial,
                config={
                    "callbacks": [],
                    "recursion_limit": AGENT2_GRAPH_RECURSION_LIMIT,
                },
            )
            safe_final = _CallbackSafeVerificationState.model_validate(final_state)
            artifact = self._artifact(safe_final.model_dump(mode="python"))
            if artifact.input_error:
                raise VerificationInputError(
                    "income verification input is invalid"
                ) from None
            if not safe_final.result_ready or artifact.result is None:
                raise RuntimeError("Agent 2 graph did not produce a terminal result")
            return IncomeVerificationResult.model_validate(
                artifact.result.model_dump(mode="python", warnings="none")
            )
        finally:
            with self._artifact_lock:
                self._artifacts.pop(run_token, None)

    def _build_graph(self):
        """Compile the five-node decision/tool graph with explicit loop routes."""

        builder = StateGraph(VerificationWorkflowState)
        builder.add_node("validate_request", self._validate_request_node)
        builder.add_node("model_decision", self._model_decision_node)
        builder.add_node("decision_guard", self._decision_guard_node)
        builder.add_node("execute_tool", self._execute_tool_node)
        builder.add_node("finalize", self._finalize_node)
        builder.add_edge(START, "validate_request")
        builder.add_conditional_edges(
            "validate_request",
            self._route_after_validation,
            {"model_decision": "model_decision", "finalize": "finalize"},
        )
        builder.add_edge("model_decision", "decision_guard")
        builder.add_conditional_edges(
            "decision_guard",
            self._route_after_decision_guard,
            {
                "model_decision": "model_decision",
                "execute_tool": "execute_tool",
                "finalize": "finalize",
            },
        )
        builder.add_conditional_edges(
            "execute_tool",
            self._route_after_tool,
            {"model_decision": "model_decision", "finalize": "finalize"},
        )
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=False, name="income_verification_agent")

    def _artifact(
        self,
        state: VerificationWorkflowState,
    ) -> _VerificationRunArtifacts:
        """Resolve private artifacts by a random token carrying no business data."""

        run_token = state.get("run_token")
        if not isinstance(run_token, str):
            raise RuntimeError("missing Agent 2 run token")
        with self._artifact_lock:
            artifact = self._artifacts.get(run_token)
        if artifact is None:
            raise RuntimeError("missing Agent 2 private artifacts")
        return artifact

    def _validate_request_node(
        self,
        state: VerificationWorkflowState,
    ) -> VerificationWorkflowState:
        """Revalidate the typed handoff before the first local-model decision."""

        artifact = self._artifact(state)
        try:
            request = self._revalidate_request(artifact.request)  # type: ignore[arg-type]
        except VerificationInputError:
            artifact.input_error = True
            return {
                "request_validated": False,
                "next_step": "finalize",
            }
        artifact.validated_request = request
        artifact.private_evidence = {
            item.evidence_ref: item for item in request.evidence
        }
        return {
            "request_validated": True,
            "next_step": "model_decision",
        }

    def _model_decision_node(
        self,
        state: VerificationWorkflowState,
    ) -> VerificationWorkflowState:
        """Request one typed local-model action without exposing private values."""

        artifact = self._artifact(state)
        request = artifact.validated_request
        if request is None:
            raise RuntimeError("Agent 2 request was not validated")
        decision_count = state.get("model_decision_count", 0) + 1
        if decision_count > MAX_VERIFICATION_MODEL_DECISIONS:
            raise RuntimeError("Agent 2 model-decision ceiling was bypassed")
        context = self._decision_context(
            request,
            artifact.normalized,
            artifact.observations,
            state.get("tool_call_count", 0),
        )
        artifact.proposed_decision = None
        artifact.decision_error = None
        try:
            artifact.proposed_decision = self._decision_model.decide(context)
        except InvalidToolDecisionError:
            artifact.decision_error = "invalid"
        except Exception:
            artifact.decision_error = "operational"
        return {
            "model_decision_count": decision_count,
            "decision_ready": artifact.decision_error is None,
        }

    def _decision_guard_node(
        self,
        state: VerificationWorkflowState,
    ) -> VerificationWorkflowState:
        """Validate a proposed action and select tool, retry, or terminal routing."""

        artifact = self._artifact(state)
        if artifact.decision_error == "operational":
            return self._failure_update(
                artifact,
                VerificationFailureCode.MODEL_DECISION_FAILED,
                state,
            )
        if artifact.decision_error == "invalid":
            return self._rejection_update(
                artifact,
                state,
                ToolObservationCode.INVALID_DECISION,
            )
        try:
            proposed = artifact.proposed_decision
            decision = VerificationToolDecision.model_validate(
                proposed.model_dump(mode="python", warnings="none")  # type: ignore[union-attr]
            )
        except Exception:
            return self._rejection_update(
                artifact,
                state,
                ToolObservationCode.INVALID_DECISION,
            )
        artifact.proposed_decision = decision

        request = artifact.validated_request
        if request is None:
            raise RuntimeError("Agent 2 request was not validated")
        if decision.action is DecisionAction.COMPLETE:
            if artifact.comparisons or not self._has_supported_document_pair(request):
                artifact.result = self._complete_result(
                    artifact.normalized,
                    artifact.comparisons,
                    state.get("tool_call_count", 0),
                    state.get("invalid_decision_count", 0),
                )
                return {
                    "decision_ready": False,
                    "next_step": "finalize",
                    "failure_code": None,
                }
            return self._rejection_update(
                artifact,
                state,
                ToolObservationCode.PREMATURE_COMPLETION,
            )

        if state.get("tool_call_count", 0) >= MAX_VERIFICATION_TOOL_CALLS:
            return self._failure_update(
                artifact,
                VerificationFailureCode.TOOL_CALL_LIMIT_REACHED,
                state,
            )
        return {
            "decision_ready": True,
            "next_step": "execute_tool",
            "failure_code": None,
        }

    def _execute_tool_node(
        self,
        state: VerificationWorkflowState,
    ) -> VerificationWorkflowState:
        """Authorize and execute one deterministic tool against private evidence."""

        artifact = self._artifact(state)
        try:
            decision = VerificationToolDecision.model_validate(
                artifact.proposed_decision.model_dump(  # type: ignore[union-attr]
                    mode="python",
                    warnings="none",
                )
            )
        except Exception:
            return self._rejection_update(
                artifact,
                state,
                ToolObservationCode.INVALID_DECISION,
            )

        tool_call_count = state.get("tool_call_count", 0) + 1
        try:
            observation = self._execute_tool(
                decision,
                artifact.private_evidence,
                artifact.normalized,
                artifact.comparisons,
            )
        except _ToolExecutionError:
            return self._rejection_update(
                artifact,
                state,
                ToolObservationCode.INVALID_TOOL_ARGUMENTS,
                tool_name=decision.tool_name,
                tool_call_count=tool_call_count,
            )

        artifact.observations.append(observation)
        artifact.proposed_decision = None
        if state.get("model_decision_count", 0) >= MAX_VERIFICATION_MODEL_DECISIONS:
            return self._failure_update(
                artifact,
                VerificationFailureCode.TOOL_CALL_LIMIT_REACHED,
                state,
                tool_call_count=tool_call_count,
            )
        return {
            "tool_call_count": tool_call_count,
            "decision_ready": False,
            "next_step": "model_decision",
            "failure_code": None,
        }

    def _finalize_node(
        self,
        state: VerificationWorkflowState,
    ) -> VerificationWorkflowState:
        """Revalidate the private terminal result or preserve input rejection."""

        artifact = self._artifact(state)
        artifact.proposed_decision = None
        artifact.decision_error = None
        if artifact.input_error:
            return {
                "decision_ready": False,
                "result_ready": False,
                "next_step": "finalize",
            }
        if artifact.result is None:
            artifact.result = self._failed_result(
                VerificationFailureCode.MODEL_DECISION_FAILED,
                state.get("tool_call_count", 0),
                state.get("invalid_decision_count", 0),
            )
        try:
            artifact.result = IncomeVerificationResult.model_validate(
                artifact.result.model_dump(mode="python", warnings="none")
            )
        except Exception:
            artifact.result = self._failed_result(
                VerificationFailureCode.MODEL_DECISION_FAILED,
                state.get("tool_call_count", 0),
                state.get("invalid_decision_count", 0),
            )
        return {
            "tool_call_count": artifact.result.tool_call_count,
            "invalid_decision_count": artifact.result.invalid_decision_count,
            "decision_ready": False,
            "result_ready": True,
            "next_step": "finalize",
            "failure_code": artifact.result.failure_code,
        }

    def _rejection_update(
        self,
        artifact: _VerificationRunArtifacts,
        state: VerificationWorkflowState,
        code: ToolObservationCode,
        *,
        tool_name: VerificationToolName | None = None,
        tool_call_count: int | None = None,
    ) -> VerificationWorkflowState:
        """Record one closed rejection and enforce retry and decision ceilings."""

        calls = (
            state.get("tool_call_count", 0)
            if tool_call_count is None
            else tool_call_count
        )
        invalid_count = state.get("invalid_decision_count", 0) + 1
        artifact.observations.append(
            VerificationToolObservation(
                status=ToolObservationStatus.REJECTED,
                code=code,
                tool_name=tool_name,
            )
        )
        artifact.proposed_decision = None
        artifact.decision_error = None
        if invalid_count > MAX_INVALID_DECISIONS:
            return self._failure_update(
                artifact,
                VerificationFailureCode.INVALID_MODEL_DECISION,
                state,
                tool_call_count=calls,
                invalid_decision_count=invalid_count,
            )
        if state.get("model_decision_count", 0) >= MAX_VERIFICATION_MODEL_DECISIONS:
            return self._failure_update(
                artifact,
                VerificationFailureCode.TOOL_CALL_LIMIT_REACHED,
                state,
                tool_call_count=calls,
                invalid_decision_count=invalid_count,
            )
        return {
            "tool_call_count": calls,
            "invalid_decision_count": invalid_count,
            "decision_ready": False,
            "next_step": "model_decision",
            "failure_code": None,
        }

    def _failure_update(
        self,
        artifact: _VerificationRunArtifacts,
        failure_code: VerificationFailureCode,
        state: VerificationWorkflowState,
        *,
        tool_call_count: int | None = None,
        invalid_decision_count: int | None = None,
    ) -> VerificationWorkflowState:
        """Create one PII-free terminal failure and route to finalization."""

        calls = (
            state.get("tool_call_count", 0)
            if tool_call_count is None
            else tool_call_count
        )
        invalid = (
            state.get("invalid_decision_count", 0)
            if invalid_decision_count is None
            else invalid_decision_count
        )
        artifact.proposed_decision = None
        artifact.decision_error = None
        artifact.result = self._failed_result(failure_code, calls, invalid)
        return {
            "tool_call_count": calls,
            "invalid_decision_count": invalid,
            "decision_ready": False,
            "next_step": "finalize",
            "failure_code": failure_code,
        }

    @staticmethod
    def _route_after_validation(
        state: VerificationWorkflowState,
    ) -> Literal["model_decision", "finalize"]:
        """Prevent any model call when the Agent 1 handoff is invalid."""

        return (
            "model_decision"
            if state.get("request_validated") is True
            else "finalize"
        )

    @staticmethod
    def _route_after_decision_guard(
        state: VerificationWorkflowState,
    ) -> Literal["model_decision", "execute_tool", "finalize"]:
        """Route one guarded model proposal to retry, tool execution, or exit."""

        step = state.get("next_step", "finalize")
        if step not in {"model_decision", "execute_tool", "finalize"}:
            return "finalize"
        return step

    @staticmethod
    def _route_after_tool(
        state: VerificationWorkflowState,
    ) -> Literal["model_decision", "finalize"]:
        """Continue the agentic loop only when another decision is permitted."""

        return (
            "model_decision"
            if state.get("next_step") == "model_decision"
            else "finalize"
        )

    @staticmethod
    def _revalidate_request(
        request: IncomeVerificationRequest,
    ) -> IncomeVerificationRequest:
        """Reject constructed or mutated input before invoking the model."""

        try:
            return IncomeVerificationRequest.model_validate(
                request.model_dump(mode="python", warnings="none")
            )
        except (AttributeError, TypeError, ValueError, ValidationError):
            raise VerificationInputError("income verification input is invalid") from None

    @staticmethod
    def _decision_context(
        request: IncomeVerificationRequest,
        normalized: dict[str, NormalizedIncome],
        observations: list[VerificationToolObservation],
        tool_call_count: int,
    ) -> VerificationDecisionContext:
        """Build the amount-free context visible to the decision model."""

        return VerificationDecisionContext(
            tools=VERIFICATION_TOOL_DECLARATIONS,
            evidence=tuple(
                EvidenceDescriptor(
                    evidence_ref=item.evidence_ref,
                    document_type=item.document_type,
                    period=item.period,
                    income_basis=item.income_basis,
                    calendar_year=item.calendar_year,
                )
                for item in request.evidence
            ),
            normalized_income=tuple(
                NormalizedIncomeDescriptor(
                    normalized_ref=item.normalized_ref,
                    source_evidence_ref=item.source_evidence_ref,
                    document_type=item.document_type,
                )
                for item in normalized.values()
            ),
            observations=tuple(observations),
            remaining_tool_calls=MAX_VERIFICATION_TOOL_CALLS - tool_call_count,
        )

    @staticmethod
    def _execute_tool(
        decision: VerificationToolDecision,
        private_evidence: dict[str, IncomeEvidence],
        normalized: dict[str, NormalizedIncome],
        comparisons: list[IncomeComparison],
    ) -> VerificationToolObservation:
        """Authorize, resolve, and execute one model-selected local tool."""

        if decision.tool_name is VerificationToolName.NORMALIZE_PAYSTUB_INCOME:
            result = _normalize_paystub_income(decision.evidence_ref, private_evidence)
            normalized[result.normalized_ref] = result
            return VerificationToolObservation(
                status=ToolObservationStatus.SUCCEEDED,
                code=ToolObservationCode.TOOL_COMPLETED,
                tool_name=decision.tool_name,
                produced_ref=result.normalized_ref,
            )
        if decision.tool_name is VerificationToolName.NORMALIZE_W2_INCOME:
            result = _normalize_w2_income(decision.evidence_ref, private_evidence)
            normalized[result.normalized_ref] = result
            return VerificationToolObservation(
                status=ToolObservationStatus.SUCCEEDED,
                code=ToolObservationCode.TOOL_COMPLETED,
                tool_name=decision.tool_name,
                produced_ref=result.normalized_ref,
            )
        if decision.tool_name is VerificationToolName.COMPARE_INCOME_SOURCES:
            comparison = _compare_income_sources(
                decision.left_normalized_ref,
                decision.right_normalized_ref,
                normalized,
            )
            comparisons[:] = [comparison]
            return VerificationToolObservation(
                status=ToolObservationStatus.SUCCEEDED,
                code=ToolObservationCode.TOOL_COMPLETED,
                tool_name=decision.tool_name,
                comparison_result=comparison.result,
            )
        raise _ToolExecutionError("tool is not allowlisted")

    @staticmethod
    def _has_supported_document_pair(request: IncomeVerificationRequest) -> bool:
        """Return whether the first slice has one paystub and one W-2 source."""

        types = {item.document_type for item in request.evidence}
        return types == {DocumentType.PAY_STUB, DocumentType.TAX_FORM}

    @staticmethod
    def _complete_result(
        normalized: dict[str, NormalizedIncome],
        comparisons: list[IncomeComparison],
        tool_call_count: int,
        invalid_decision_count: int,
    ) -> IncomeVerificationResult:
        """Derive the final status from deterministic artifacts, never model text."""

        if not comparisons:
            return IncomeVerificationResult(
                normalized_income=tuple(normalized.values()),
                comparisons=(),
                status=VerificationStatus.INSUFFICIENT_EVIDENCE,
                evidence_complete=False,
                unsupported_reasons=(
                    VerificationReason.UNSUPPORTED_DOCUMENT_COMBINATION,
                ),
                tool_call_count=tool_call_count,
                invalid_decision_count=invalid_decision_count,
            )

        comparison = comparisons[0]
        status = VerificationStatus(comparison.result.value)
        unsupported_reasons: tuple[VerificationReason, ...] = ()
        if status is VerificationStatus.NOT_COMPARABLE:
            unsupported_reasons = (comparison.reason_code,)
        return IncomeVerificationResult(
            normalized_income=tuple(normalized.values()),
            comparisons=(comparison,),
            status=status,
            evidence_complete=True,
            unsupported_reasons=unsupported_reasons,
            tool_call_count=tool_call_count,
            invalid_decision_count=invalid_decision_count,
        )

    @staticmethod
    def _failed_result(
        failure_code: VerificationFailureCode,
        tool_call_count: int,
        invalid_decision_count: int,
    ) -> IncomeVerificationResult:
        """Return a PII-free failure with no partial financial values."""

        return IncomeVerificationResult(
            normalized_income=(),
            comparisons=(),
            status=VerificationStatus.FAILED,
            evidence_complete=False,
            unsupported_reasons=(),
            tool_call_count=tool_call_count,
            invalid_decision_count=invalid_decision_count,
            failure_code=failure_code,
        )


def _normalize_paystub_income(
    evidence_ref: str | None,
    private_evidence: dict[str, IncomeEvidence],
) -> NormalizedIncome:
    """Normalize one validated monthly paystub fact without model arithmetic."""

    evidence = private_evidence.get(evidence_ref or "")
    if (
        evidence is None
        or evidence.document_type is not DocumentType.PAY_STUB
        or evidence.period is not IncomePeriod.MONTHLY
    ):
        raise _ToolExecutionError("paystub normalization arguments are invalid")
    return _normalized_income(
        evidence,
        evidence.amount,
        TransformationRule.PAYSTUB_MONTHLY_V1,
    )


def _normalize_w2_income(
    evidence_ref: str | None,
    private_evidence: dict[str, IncomeEvidence],
) -> NormalizedIncome:
    """Convert one validated annual W-2 fact to monthly Decimal income."""

    evidence = private_evidence.get(evidence_ref or "")
    if (
        evidence is None
        or evidence.document_type is not DocumentType.TAX_FORM
        or evidence.period is not IncomePeriod.ANNUAL
    ):
        raise _ToolExecutionError("W-2 normalization arguments are invalid")
    return _normalized_income(
        evidence,
        evidence.amount / Decimal("12"),
        TransformationRule.W2_ANNUAL_TO_MONTHLY_V1,
    )


def _normalized_income(
    evidence: IncomeEvidence,
    monthly_amount: Decimal,
    rule: TransformationRule,
) -> NormalizedIncome:
    """Create an opaque reference and cent-rounded deterministic derivation."""

    digest = hashlib.sha256(
        f"{evidence.evidence_ref}:{rule.value}".encode("ascii")
    ).hexdigest()[:32]
    return NormalizedIncome(
        normalized_ref=f"norm_{digest}",
        document_ref=evidence.document_ref,
        document_type=evidence.document_type,
        monthly_amount=monthly_amount.quantize(_CENT, rounding=ROUND_HALF_UP),
        income_basis=evidence.income_basis,
        calendar_year=evidence.calendar_year,
        source_evidence_ref=evidence.evidence_ref,
        provenance=evidence.provenance,
        transformation_rule=rule,
    )


def _compare_income_sources(
    left_ref: str | None,
    right_ref: str | None,
    normalized: dict[str, NormalizedIncome],
) -> IncomeComparison:
    """Compare two trusted normalized facts under the exact policy."""

    left = normalized.get(left_ref or "")
    right = normalized.get(right_ref or "")
    if left is None or right is None or left.normalized_ref == right.normalized_ref:
        raise _ToolExecutionError("comparison references are invalid")

    if left.income_basis is not right.income_basis:
        difference = None
        percentage = None
        result = ComparisonResult.NOT_COMPARABLE
        reason = VerificationReason.INCOME_BASIS_NOT_COMPARABLE
    elif left.calendar_year != right.calendar_year:
        difference = None
        percentage = None
        result = ComparisonResult.NOT_COMPARABLE
        reason = VerificationReason.INCOME_PERIOD_NOT_COMPARABLE
    else:
        difference = abs(left.monthly_amount - right.monthly_amount).quantize(
            _CENT,
            rounding=ROUND_HALF_UP,
        )
        maximum = max(left.monthly_amount, right.monthly_amount)
        percentage = (
            Decimal("0.00")
            if maximum == 0
            else ((difference / maximum) * Decimal("100")).quantize(
                _PERCENT,
                rounding=ROUND_HALF_UP,
            )
        )
        if difference == 0:
            result = ComparisonResult.CONSISTENT
            reason = VerificationReason.EXACT_MATCH
        else:
            result = ComparisonResult.INCONSISTENT
            reason = VerificationReason.INCOME_VALUES_INCONSISTENT

    return IncomeComparison(
        left_income_ref=left.normalized_ref,
        right_income_ref=right.normalized_ref,
        amount_difference=difference,
        percentage_difference=percentage,
        result=result,
        reason_code=reason,
    )


class EvidenceLinkedFinding(StrictModel):
    """Compatibility contract retained until the workflow handoff is upgraded.

    The Milestone 1 workflow and the unimplemented Agent 3 placeholder import
    this type. Keeping it here prevents this standalone Agent 2 increment from
    silently changing their behavior.
    """

    finding_code: StrictStr = Field(pattern=r"^[a-z][a-z0-9_.-]{0,63}$")
    provenance: tuple[SourceProvenance, ...] = Field(min_length=1)


@runtime_checkable
class VerificationAgent(Protocol):
    """Legacy handoff protocol preserved until orchestrator integration."""

    def verify(
        self,
        extraction: ValidatedExtraction,
    ) -> Sequence[EvidenceLinkedFinding]:
        """Return evidence-linked findings under the original placeholder API."""

        ...


__all__ = [
    "AGENT2_GRAPH_RECURSION_LIMIT",
    "AGENT2_DECISION_PROMPT_VERSION",
    "COMPARISON_POLICY_ID",
    "DecisionAction",
    "EvidenceLinkedFinding",
    "IncomeBasis",
    "IncomeEvidence",
    "IncomePeriod",
    "IncomeVerificationAgent",
    "IncomeVerificationRequest",
    "IncomeVerificationResult",
    "InvalidToolDecisionError",
    "InvalidToolDecisionReason",
    "MAX_INVALID_DECISIONS",
    "MAX_VERIFICATION_MODEL_DECISIONS",
    "MAX_VERIFICATION_TOOL_CALLS",
    "OllamaIncomeToolDecisionModel",
    "ToolDecisionModel",
    "VerificationDecisionContext",
    "VerificationFailureCode",
    "VerificationInputError",
    "VerificationAgent",
    "VerificationReason",
    "VerificationStatus",
    "VerificationToolDecision",
    "VerificationToolName",
    "VerificationWorkflowState",
    "VERIFICATION_TOOL_DECLARATIONS",
]
