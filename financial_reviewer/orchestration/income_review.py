"""Orchestrate three agents behind a deterministic final release gate.

Why this file exists:
    ``workflow.DocumentExtractionReviewer`` securely reviews one document. The financial
    income use case needs a higher-level graph that reviews exactly two local
    documents, assembles guarded handoffs, and invokes Agents 2 and 3.

What this increment owns:
    Six nodes—document extraction, verification-input assembly, verification,
    critic-input assembly, criticism, and a deterministic final gate—plus
    callback-safe state, private run artifacts, strict conditional routing, and
    one vendor-neutral sanitized parent/child trace spanning the complete
    review.

What it deliberately does not own:
    No credit/claim approval, public extraction, bank payroll interpretation,
    PDF/OCR, generic auto-instrumentation, or Traceboard-specific graph state.
    ``released`` means only that the validated review result may proceed
    downstream.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal, Mapping, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from pydantic import ConfigDict, Field, StrictBool, StrictInt, model_validator

from financial_reviewer.agents.agent2_verification import (
    MAX_INVALID_DECISIONS,
    MAX_VERIFICATION_TOOL_CALLS,
    IncomeVerificationAgent,
    IncomeVerificationRequest,
    IncomeVerificationResult,
    OllamaIncomeToolDecisionModel,
    VerificationFailureCode,
    VerificationStatus,
)
from financial_reviewer.agents.agent3_critic import (
    MAX_CRITIC_MODEL_ATTEMPTS,
    MAX_CRITIC_REPAIR_ATTEMPTS,
    CriticDisposition,
    CriticFailureCode,
    CriticReasonCode,
    CriticResult,
    CriticReviewRequest,
    CriticStatus,
    IncomeReviewCriticAgent,
    OllamaCriticDecisionModel,
)
from financial_reviewer.foundation.config import (
    LocalModelSettings,
    UnsafeRuntimeConfigurationError,
    ensure_cloud_tracing_disabled,
)
from financial_reviewer.foundation.handoffs import (
    CriticHandoffError,
    CriticHandoffFailureCode,
    CriticInputAssembler,
    VerificationHandoffError,
    VerificationHandoffFailureCode,
    VerificationInputAssembler,
)
from financial_reviewer.foundation.intake import DocumentSubmission
from financial_reviewer.foundation.schemas import (
    DocumentType,
    FailureCode,
    StrictModel,
    ValidatedExtraction,
    WorkflowStatus,
)
from financial_reviewer.local.model import OllamaModel
from financial_reviewer.local.telemetry import (
    NoOpReviewTraceSink,
    ReviewTraceSession,
    ReviewTraceSink,
    ReviewTraceSpan,
    SanitizedTraceCause,
    TraceCauseKind,
    TraceSpanOutcome,
    TraceStage,
)
from financial_reviewer.workflow import DocumentExtractionReviewer


# The graph has six application nodes and no graph-level loop. This separate
# ceiling protects orchestration from an accidental future cycle; both agents'
# internal model/tool bounds remain independently enforced.
INCOME_REVIEW_GRAPH_RECURSION_LIMIT = 9


class IncomeReviewFailureCode(str, Enum):
    """Closed orchestration stages that can block multi-agent progression."""

    INVALID_BUNDLE = "invalid_bundle"
    EXTRACTION_FAILED = "extraction_failed"
    HANDOFF_FAILED = "handoff_failed"
    VERIFICATION_FAILED = "verification_failed"
    CRITIC_HANDOFF_FAILED = "critic_handoff_failed"
    CRITIC_FAILED = "critic_failed"
    FINAL_GATE_FAILED = "final_gate_failed"
    UNSAFE_CONFIGURATION = "unsafe_configuration"
    INTERNAL_FAILURE = "internal_failure"


class IncomeReviewReasonCode(str, Enum):
    """Closed final reasons kept separate from technical failure lineage."""

    CONSISTENT_INCOME_GROUNDED = "consistent_income_grounded"
    INCOME_INCONSISTENT = "income_inconsistent"
    INCOME_NOT_COMPARABLE = "income_not_comparable"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PROCESSING_FAILURE = "processing_failure"


# This table—not either agent—owns the final release policy. Every accepted
# domain status has exactly one compatible Critic decision and terminal result.
_FINAL_DECISION_POLICY: dict[
    VerificationStatus,
    tuple[
        CriticDisposition,
        CriticReasonCode,
        WorkflowStatus,
        IncomeReviewReasonCode,
    ],
] = {
    VerificationStatus.CONSISTENT: (
        CriticDisposition.GROUNDED,
        CriticReasonCode.EVIDENCE_CONSISTENT,
        WorkflowStatus.RELEASED,
        IncomeReviewReasonCode.CONSISTENT_INCOME_GROUNDED,
    ),
    VerificationStatus.INCONSISTENT: (
        CriticDisposition.ESCALATE,
        CriticReasonCode.INCOME_INCONSISTENT,
        WorkflowStatus.HUMAN_REVIEW,
        IncomeReviewReasonCode.INCOME_INCONSISTENT,
    ),
    VerificationStatus.NOT_COMPARABLE: (
        CriticDisposition.ESCALATE,
        CriticReasonCode.INCOME_NOT_COMPARABLE,
        WorkflowStatus.HUMAN_REVIEW,
        IncomeReviewReasonCode.INCOME_NOT_COMPARABLE,
    ),
    VerificationStatus.INSUFFICIENT_EVIDENCE: (
        CriticDisposition.REFUSE,
        CriticReasonCode.INSUFFICIENT_EVIDENCE,
        WorkflowStatus.HUMAN_REVIEW,
        IncomeReviewReasonCode.INSUFFICIENT_EVIDENCE,
    ),
}


class IncomeReviewBundle(StrictModel):
    """Exactly two synthetic local submissions for cross-document review."""

    documents: tuple[DocumentSubmission, DocumentSubmission] = Field(repr=False)


class IncomeReviewOutcome(StrictModel):
    """PII-free final review result; release is not financial approval."""

    status: WorkflowStatus
    release_allowed: StrictBool
    final_reason_code: IncomeReviewReasonCode
    document_types: Annotated[tuple[DocumentType, ...], Field(max_length=2)] = ()
    verification_status: VerificationStatus | None = None
    failure_code: IncomeReviewFailureCode | None = None
    upstream_failure_codes: tuple[FailureCode, ...] = ()
    handoff_failure_code: VerificationHandoffFailureCode | None = None
    verification_failure_code: VerificationFailureCode | None = None
    critic_handoff_failure_code: CriticHandoffFailureCode | None = None
    critic_status: CriticStatus | None = None
    critic_disposition: CriticDisposition | None = None
    critic_reason_code: CriticReasonCode | None = None
    critic_failure_code: CriticFailureCode | None = None
    tool_call_count: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_VERIFICATION_TOOL_CALLS),
    ] = 0
    invalid_decision_count: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_INVALID_DECISIONS + 1),
    ] = 0
    critic_attempt_count: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_CRITIC_MODEL_ATTEMPTS),
    ] = 0
    critic_repair_count: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_CRITIC_REPAIR_ATTEMPTS),
    ] = 0

    @model_validator(mode="after")
    def validate_terminal_contract(self) -> "IncomeReviewOutcome":
        """Require released, domain-review, and failed outcomes to stay distinct."""

        if self.status is WorkflowStatus.RELEASED:
            if (
                not self.release_allowed
                or self.final_reason_code
                is not IncomeReviewReasonCode.CONSISTENT_INCOME_GROUNDED
                or self.failure_code is not None
                or self.upstream_failure_codes
                or self.handoff_failure_code is not None
                or self.verification_failure_code is not None
                or self.critic_handoff_failure_code is not None
                or self.critic_failure_code is not None
                or self.verification_status in {None, VerificationStatus.FAILED}
                or self.verification_status is not VerificationStatus.CONSISTENT
                or self.critic_status is not CriticStatus.COMPLETED
                or self.critic_disposition is not CriticDisposition.GROUNDED
                or self.critic_reason_code
                is not CriticReasonCode.EVIDENCE_CONSISTENT
                or set(self.document_types)
                != {DocumentType.PAY_STUB, DocumentType.TAX_FORM}
            ):
                raise ValueError("released income-review outcome is inconsistent")
        elif self.release_allowed:
            raise ValueError("human-review outcome cannot permit release")
        elif self.final_reason_code is IncomeReviewReasonCode.PROCESSING_FAILURE:
            if self.failure_code is None:
                raise ValueError("processing failure requires failure lineage")
        else:
            expected_domain_result = {
                IncomeReviewReasonCode.INCOME_INCONSISTENT: (
                    VerificationStatus.INCONSISTENT,
                    CriticDisposition.ESCALATE,
                    CriticReasonCode.INCOME_INCONSISTENT,
                ),
                IncomeReviewReasonCode.INCOME_NOT_COMPARABLE: (
                    VerificationStatus.NOT_COMPARABLE,
                    CriticDisposition.ESCALATE,
                    CriticReasonCode.INCOME_NOT_COMPARABLE,
                ),
                IncomeReviewReasonCode.INSUFFICIENT_EVIDENCE: (
                    VerificationStatus.INSUFFICIENT_EVIDENCE,
                    CriticDisposition.REFUSE,
                    CriticReasonCode.INSUFFICIENT_EVIDENCE,
                ),
            }.get(self.final_reason_code)
            if (
                expected_domain_result is None
                or self.failure_code is not None
                or self.upstream_failure_codes
                or self.handoff_failure_code is not None
                or self.verification_failure_code is not None
                or self.critic_handoff_failure_code is not None
                or self.critic_failure_code is not None
                or self.critic_status is not CriticStatus.COMPLETED
                or (
                    self.verification_status,
                    self.critic_disposition,
                    self.critic_reason_code,
                )
                != expected_domain_result
                or set(self.document_types)
                != {DocumentType.PAY_STUB, DocumentType.TAX_FORM}
            ):
                raise ValueError("domain human-review outcome is inconsistent")
        return self


class IncomeReviewState(TypedDict, total=False):
    """Only PII-safe routing data visible to LangGraph and callbacks."""

    run_token: str
    document_types: tuple[DocumentType, ...]
    extractions_ready: bool
    verification_request_ready: bool
    verification_result_ready: bool
    ready_for_critic: bool
    critic_request_ready: bool
    critic_result_ready: bool
    ready_for_final_gate: bool
    failure_code: IncomeReviewFailureCode | None
    upstream_failure_codes: tuple[FailureCode, ...]
    handoff_failure_code: VerificationHandoffFailureCode | None
    verification_status: VerificationStatus | None
    verification_failure_code: VerificationFailureCode | None
    critic_handoff_failure_code: CriticHandoffFailureCode | None
    critic_status: CriticStatus | None
    critic_disposition: CriticDisposition | None
    critic_reason_code: CriticReasonCode | None
    critic_failure_code: CriticFailureCode | None
    tool_call_count: int
    invalid_decision_count: int
    critic_attempt_count: int
    critic_repair_count: int
    review_status: WorkflowStatus | None
    release_allowed: bool
    final_reason_code: IncomeReviewReasonCode | None


class _CallbackSafeState(StrictModel):
    """Runtime validator for the exact state allowed to enter LangGraph."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_token: str
    document_types: tuple[DocumentType, ...] = ()
    extractions_ready: StrictBool = False
    verification_request_ready: StrictBool = False
    verification_result_ready: StrictBool = False
    ready_for_critic: StrictBool = False
    critic_request_ready: StrictBool = False
    critic_result_ready: StrictBool = False
    ready_for_final_gate: StrictBool = False
    failure_code: IncomeReviewFailureCode | None = None
    upstream_failure_codes: tuple[FailureCode, ...] = ()
    handoff_failure_code: VerificationHandoffFailureCode | None = None
    verification_status: VerificationStatus | None = None
    verification_failure_code: VerificationFailureCode | None = None
    critic_handoff_failure_code: CriticHandoffFailureCode | None = None
    critic_status: CriticStatus | None = None
    critic_disposition: CriticDisposition | None = None
    critic_reason_code: CriticReasonCode | None = None
    critic_failure_code: CriticFailureCode | None = None
    tool_call_count: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_VERIFICATION_TOOL_CALLS),
    ] = 0
    invalid_decision_count: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_INVALID_DECISIONS + 1),
    ] = 0
    critic_attempt_count: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_CRITIC_MODEL_ATTEMPTS),
    ] = 0
    critic_repair_count: Annotated[
        StrictInt,
        Field(ge=0, le=MAX_CRITIC_REPAIR_ATTEMPTS),
    ] = 0
    review_status: WorkflowStatus | None = None
    release_allowed: StrictBool = False
    final_reason_code: IncomeReviewReasonCode | None = None


@dataclass
class _IncomeReviewArtifacts:
    """Sensitive objects shared privately for one graph invocation."""

    bundle: IncomeReviewBundle = field(repr=False)
    trace_session: ReviewTraceSession = field(repr=False)
    review_trace_span: ReviewTraceSpan = field(repr=False)
    active_node_span: ReviewTraceSpan | None = field(default=None, repr=False)
    extractions: tuple[ValidatedExtraction, ...] = field(default=(), repr=False)
    verification_request: IncomeVerificationRequest | None = field(default=None, repr=False)
    verification_result: IncomeVerificationResult | None = field(default=None, repr=False)
    critic_request: CriticReviewRequest | None = field(default=None, repr=False)
    critic_result: CriticResult | None = field(default=None, repr=False)


class _IncomeReviewGraph:
    """Private LangGraph implementation for the gated three-agent review flow."""

    def __init__(
        self,
        extractor: DocumentExtractionReviewer,
        verification_agent: IncomeVerificationAgent,
        critic_agent: IncomeReviewCriticAgent,
    ) -> None:
        """Bind already validated local components and compile the graph once."""

        self._extractor = extractor
        self._verification_agent = verification_agent
        self._critic_agent = critic_agent
        self._artifacts: dict[str, _IncomeReviewArtifacts] = {}
        self._artifact_lock = threading.RLock()
        self._graph = self._build_graph()

    @property
    def node_names(self) -> tuple[str, ...]:
        """Expose application node names for architecture and regression tests."""

        nodes = self._graph.get_graph().nodes
        return tuple(
            name
            for name in nodes
            if name not in {START, END, "__start__", "__end__"}
        )

    def invoke(
        self,
        bundle: IncomeReviewBundle,
        *,
        trace_session: ReviewTraceSession,
        review_trace_span: ReviewTraceSpan,
    ) -> IncomeReviewOutcome:
        """Run one bundle and delete every private artifact on all exit paths."""

        run_token = f"run_{uuid4().hex}"
        artifact = _IncomeReviewArtifacts(
            bundle=bundle,
            trace_session=trace_session,
            review_trace_span=review_trace_span,
        )
        with self._artifact_lock:
            self._artifacts[run_token] = artifact
        initial = _CallbackSafeState(run_token=run_token).model_dump(
            mode="python",
            warnings="none",
        )
        try:
            final_state = self._graph.invoke(
                initial,
                config={
                    "callbacks": [],
                    "recursion_limit": INCOME_REVIEW_GRAPH_RECURSION_LIMIT,
                },
            )
            return self._to_outcome(final_state)
        except Exception:
            return _human_review_outcome(IncomeReviewFailureCode.INTERNAL_FAILURE)
        finally:
            with self._artifact_lock:
                self._artifacts.pop(run_token, None)

    def _build_graph(self):
        """Compile the completed S3 graph with fail-closed conditional edges."""

        builder = StateGraph(IncomeReviewState)
        builder.add_node(
            "extract_documents",
            self._traced_node(TraceStage.EXTRACT_DOCUMENTS, self._extract_documents),
        )
        builder.add_node(
            "verification_input_assemble",
            self._traced_node(
                TraceStage.VERIFICATION_INPUT_ASSEMBLE,
                self._assemble_input,
            ),
        )
        builder.add_node(
            "verification",
            self._traced_node(TraceStage.AGENT_2_VERIFICATION, self._verify),
        )
        builder.add_node(
            "critic_input_assemble",
            self._traced_node(
                TraceStage.CRITIC_INPUT_ASSEMBLE,
                self._assemble_critic_input,
            ),
        )
        builder.add_node(
            "critic",
            self._traced_node(TraceStage.AGENT_3_CRITIC, self._critique),
        )
        builder.add_node(
            "final_gate",
            self._traced_node(TraceStage.FINAL_GATE, self._final_gate),
        )
        builder.add_edge(START, "extract_documents")
        builder.add_conditional_edges(
            "extract_documents",
            self._route_after_extraction,
            {
                "verification_input_assemble": "verification_input_assemble",
                "final_gate": "final_gate",
            },
        )
        builder.add_conditional_edges(
            "verification_input_assemble",
            self._route_after_assembly,
            {"verification": "verification", "final_gate": "final_gate"},
        )
        builder.add_conditional_edges(
            "verification",
            self._route_after_verification,
            {
                "critic_input_assemble": "critic_input_assemble",
                "final_gate": "final_gate",
            },
        )
        builder.add_conditional_edges(
            "critic_input_assemble",
            self._route_after_critic_assembly,
            {"critic": "critic", "final_gate": "final_gate"},
        )
        builder.add_edge("critic", "final_gate")
        builder.add_edge("final_gate", END)
        return builder.compile(checkpointer=False, name="financial_income_review_s3_3")

    def _traced_node(
        self,
        stage: TraceStage,
        operation: Callable[[IncomeReviewState], IncomeReviewState],
    ) -> Callable[[IncomeReviewState], IncomeReviewState]:
        """Wrap one outer node with a sanitized child span.

        The active handle is retained only in the private run artifact so the
        Extractor can attach its existing detailed spans beneath the
        ``extract_documents`` span. No trace identifier or handle enters graph
        state, model context, audit metadata, or public output.
        """

        def traced(state: IncomeReviewState) -> IncomeReviewState:
            artifact = self._artifact(state)
            if artifact.active_node_span is not None:
                raise RuntimeError("another trace node is already active")
            span = artifact.review_trace_span.start_child(stage)
            artifact.active_node_span = span
            try:
                try:
                    update = operation(state)
                except Exception:
                    cause = SanitizedTraceCause(
                        origin_stage=stage,
                        kind=TraceCauseKind.TECHNICAL_FAILURE,
                        reason_code="internal_failure",
                    )
                    span.finish(TraceSpanOutcome.FAILED, causes=(cause,))
                    raise
                effective_state: IncomeReviewState = {**state, **update}
                outcome, causes = self._trace_result(stage, effective_state)
                span.finish(outcome, causes=causes)
                return update
            finally:
                artifact.active_node_span = None

        return traced

    @staticmethod
    def _trace_result(
        stage: TraceStage,
        state: IncomeReviewState,
    ) -> tuple[TraceSpanOutcome, tuple[SanitizedTraceCause, ...]]:
        """Map safe node state to one closed operational trace result."""

        failure_code = state.get("failure_code")
        if failure_code is not None:
            reason_code = failure_code.value
            if (
                stage is TraceStage.VERIFICATION_INPUT_ASSEMBLE
                and state.get("handoff_failure_code") is not None
            ):
                reason_code = state["handoff_failure_code"].value
            elif (
                stage is TraceStage.AGENT_2_VERIFICATION
                and state.get("verification_failure_code") is not None
            ):
                reason_code = state["verification_failure_code"].value
            elif (
                stage is TraceStage.CRITIC_INPUT_ASSEMBLE
                and state.get("critic_handoff_failure_code") is not None
            ):
                reason_code = state["critic_handoff_failure_code"].value
            elif (
                stage is TraceStage.AGENT_3_CRITIC
                and state.get("critic_failure_code") is not None
            ):
                reason_code = state["critic_failure_code"].value
            validation_stages = {
                TraceStage.VERIFICATION_INPUT_ASSEMBLE,
                TraceStage.CRITIC_INPUT_ASSEMBLE,
                TraceStage.FINAL_GATE,
            }
            cause = SanitizedTraceCause(
                origin_stage=stage,
                kind=(
                    TraceCauseKind.VALIDATION_FAILURE
                    if stage in validation_stages
                    else TraceCauseKind.TECHNICAL_FAILURE
                ),
                reason_code=reason_code,
            )
            return TraceSpanOutcome.FAILED, (cause,)
        if (
            stage is TraceStage.FINAL_GATE
            and state.get("review_status") is WorkflowStatus.HUMAN_REVIEW
        ):
            final_reason = state.get("final_reason_code")
            if final_reason is None:
                return TraceSpanOutcome.HUMAN_REVIEW, ()
            cause = SanitizedTraceCause(
                origin_stage=stage,
                kind=TraceCauseKind.BUSINESS_FINDING,
                reason_code=final_reason.value,
            )
            return TraceSpanOutcome.HUMAN_REVIEW, (cause,)
        return TraceSpanOutcome.SUCCEEDED, ()

    def _artifact(self, state: IncomeReviewState) -> _IncomeReviewArtifacts:
        """Resolve private data by random run token, never by business ID."""

        run_token = state.get("run_token")
        if not isinstance(run_token, str):
            raise RuntimeError("missing run token")
        with self._artifact_lock:
            artifact = self._artifacts.get(run_token)
        if artifact is None:
            raise RuntimeError("missing run artifacts")
        return artifact

    def _extract_documents(self, state: IncomeReviewState) -> IncomeReviewState:
        """Run the existing secure Agent 1 review independently for both inputs."""

        try:
            artifact = self._artifact(state)
            trace_parent = artifact.active_node_span
            if trace_parent is None:
                raise RuntimeError("missing extraction trace parent")
            outcomes = tuple(
                self._extractor.review_with_trace_parent(
                    document,
                    trace_session=artifact.trace_session,
                    parent_span=trace_parent,
                )
                for document in artifact.bundle.documents
            )
        except Exception:
            return {
                "extractions_ready": False,
                "failure_code": IncomeReviewFailureCode.EXTRACTION_FAILED,
                "upstream_failure_codes": (FailureCode.INTERNAL_FAILURE,),
            }

        document_types = tuple(outcome.document_type for outcome in outcomes)
        upstream_failures = tuple(
            outcome.failure_code
            for outcome in outcomes
            if outcome.status is not WorkflowStatus.RELEASED
            and outcome.failure_code is not None
        )
        extractions = tuple(
            outcome.validated_extraction
            for outcome in outcomes
            if outcome.status is WorkflowStatus.RELEASED
            and outcome.validated_extraction is not None
        )
        if upstream_failures or len(extractions) != 2:
            artifact.extractions = ()
            return {
                "document_types": document_types,
                "extractions_ready": False,
                "failure_code": IncomeReviewFailureCode.EXTRACTION_FAILED,
                "upstream_failure_codes": upstream_failures
                or (FailureCode.INTERNAL_FAILURE,),
            }
        artifact.extractions = extractions
        return {
            "document_types": document_types,
            "extractions_ready": True,
            "failure_code": None,
            "upstream_failure_codes": (),
        }

    def _assemble_input(self, state: IncomeReviewState) -> IncomeReviewState:
        """Create Agent 2's typed request from two private guarded extractions."""

        try:
            artifact = self._artifact(state)
            request = VerificationInputAssembler.assemble(artifact.extractions)
        except VerificationHandoffError as error:
            return {
                "verification_request_ready": False,
                "failure_code": IncomeReviewFailureCode.HANDOFF_FAILED,
                "handoff_failure_code": error.code,
            }
        except Exception:
            return {
                "verification_request_ready": False,
                "failure_code": IncomeReviewFailureCode.INTERNAL_FAILURE,
            }
        artifact.verification_request = request
        return {
            "verification_request_ready": True,
            "failure_code": None,
            "handoff_failure_code": None,
        }

    def _verify(self, state: IncomeReviewState) -> IncomeReviewState:
        """Invoke the bounded Verification Agent and revalidate its result."""

        try:
            artifact = self._artifact(state)
            request = artifact.verification_request
            result = self._verification_agent.verify(request)  # type: ignore[arg-type]
            result = IncomeVerificationResult.model_validate(
                result.model_dump(mode="python", warnings="none")
            )
        except Exception:
            return {
                "verification_result_ready": False,
                "ready_for_critic": False,
                "failure_code": IncomeReviewFailureCode.VERIFICATION_FAILED,
            }
        artifact.verification_result = result
        if result.status is VerificationStatus.FAILED:
            return {
                "verification_result_ready": False,
                "ready_for_critic": False,
                "verification_status": result.status,
                "verification_failure_code": result.failure_code,
                "tool_call_count": result.tool_call_count,
                "invalid_decision_count": result.invalid_decision_count,
                "failure_code": IncomeReviewFailureCode.VERIFICATION_FAILED,
            }
        return {
            "verification_result_ready": True,
            "ready_for_critic": True,
            "verification_status": result.status,
            "verification_failure_code": None,
            "tool_call_count": result.tool_call_count,
            "invalid_decision_count": result.invalid_decision_count,
            "failure_code": None,
        }

    def _assemble_critic_input(self, state: IncomeReviewState) -> IncomeReviewState:
        """Convert Agent 2's private result into an amount-free Critic request."""

        try:
            artifact = self._artifact(state)
            request = CriticInputAssembler.assemble(artifact.verification_result)  # type: ignore[arg-type]
        except CriticHandoffError as error:
            return {
                "critic_request_ready": False,
                "failure_code": IncomeReviewFailureCode.CRITIC_HANDOFF_FAILED,
                "critic_handoff_failure_code": error.code,
            }
        except Exception:
            return {
                "critic_request_ready": False,
                "failure_code": IncomeReviewFailureCode.INTERNAL_FAILURE,
            }
        artifact.critic_request = request
        return {
            "critic_request_ready": True,
            "failure_code": None,
            "critic_handoff_failure_code": None,
        }

    def _critique(self, state: IncomeReviewState) -> IncomeReviewState:
        """Invoke the bounded Critic Agent and revalidate its private result."""

        try:
            artifact = self._artifact(state)
            result = self._critic_agent.critique(artifact.critic_request)  # type: ignore[arg-type]
            result = CriticResult.model_validate(
                result.model_dump(mode="python", warnings="none")
            )
        except Exception:
            return {
                "critic_result_ready": False,
                "ready_for_final_gate": False,
                "failure_code": IncomeReviewFailureCode.CRITIC_FAILED,
            }
        artifact.critic_result = result
        if result.status is CriticStatus.FAILED:
            return {
                "critic_result_ready": False,
                "ready_for_final_gate": False,
                "critic_status": result.status,
                "critic_failure_code": result.failure_code,
                "critic_attempt_count": result.attempt_count,
                "critic_repair_count": result.repair_count,
                "failure_code": IncomeReviewFailureCode.CRITIC_FAILED,
            }
        decision = result.decision
        if decision is None:
            return {
                "critic_result_ready": False,
                "ready_for_final_gate": False,
                "critic_status": CriticStatus.FAILED,
                "failure_code": IncomeReviewFailureCode.CRITIC_FAILED,
            }
        return {
            "critic_result_ready": True,
            "ready_for_final_gate": False,
            "critic_status": result.status,
            "critic_disposition": decision.outcome,
            "critic_reason_code": decision.reason_code,
            "critic_failure_code": None,
            "critic_attempt_count": result.attempt_count,
            "critic_repair_count": result.repair_count,
            "failure_code": None,
        }

    def _final_gate(self, state: IncomeReviewState) -> IncomeReviewState:
        """Revalidate the artifact chain and apply the code-owned release table.

        This method calls no agent, model, tool, or network adapter. Rebuilding
        the amount-free Critic request verifies linkage without recalculating any
        financial value or repeating Agent 2's deterministic comparison.
        """

        if state.get("failure_code") is not None:
            return {
                "ready_for_final_gate": False,
                "review_status": WorkflowStatus.HUMAN_REVIEW,
                "release_allowed": False,
                "final_reason_code": IncomeReviewReasonCode.PROCESSING_FAILURE,
            }
        if not all(
            state.get(flag) is True
            for flag in (
                "extractions_ready",
                "verification_request_ready",
                "verification_result_ready",
                "ready_for_critic",
                "critic_request_ready",
                "critic_result_ready",
            )
        ):
            return self._final_gate_failure()

        try:
            artifact = self._artifact(state)
            verification_result = IncomeVerificationResult.model_validate(
                artifact.verification_result.model_dump(  # type: ignore[union-attr]
                    mode="python",
                    warnings="none",
                )
            )
            critic_request = CriticReviewRequest.model_validate(
                artifact.critic_request.model_dump(  # type: ignore[union-attr]
                    mode="python",
                    warnings="none",
                )
            )
            expected_critic_request = CriticInputAssembler.assemble(
                verification_result
            )
            critic_result = CriticResult.model_validate(
                artifact.critic_result.model_dump(  # type: ignore[union-attr]
                    mode="python",
                    warnings="none",
                )
            )
        except Exception:
            return self._final_gate_failure()

        decision = critic_result.decision
        if (
            critic_request != expected_critic_request
            or critic_result.status is not CriticStatus.COMPLETED
            or decision is None
            or not self._state_matches_completed_artifacts(
                state,
                verification_result,
                critic_result,
            )
        ):
            return self._final_gate_failure()

        policy = _FINAL_DECISION_POLICY.get(verification_result.status)
        if policy is None:
            return self._final_gate_failure()
        expected_disposition, expected_reason, final_status, final_reason = policy
        if (
            decision.outcome is not expected_disposition
            or decision.reason_code is not expected_reason
        ):
            return self._final_gate_failure()
        return {
            "ready_for_final_gate": True,
            "review_status": final_status,
            "release_allowed": final_status is WorkflowStatus.RELEASED,
            "final_reason_code": final_reason,
            "failure_code": None,
        }

    @staticmethod
    def _state_matches_completed_artifacts(
        state: IncomeReviewState,
        verification_result: IncomeVerificationResult,
        critic_result: CriticResult,
    ) -> bool:
        """Require callback-safe state to describe the same private artifacts."""

        decision = critic_result.decision
        return (
            decision is not None
            and set(state.get("document_types", ()))
            == {DocumentType.PAY_STUB, DocumentType.TAX_FORM}
            and state.get("verification_status") is verification_result.status
            and state.get("tool_call_count") == verification_result.tool_call_count
            and state.get("invalid_decision_count")
            == verification_result.invalid_decision_count
            and state.get("critic_status") is critic_result.status
            and state.get("critic_disposition") is decision.outcome
            and state.get("critic_reason_code") is decision.reason_code
            and state.get("critic_attempt_count") == critic_result.attempt_count
            and state.get("critic_repair_count") == critic_result.repair_count
        )

    @staticmethod
    def _final_gate_failure() -> IncomeReviewState:
        """Return one sanitized fail-closed state for an invalid artifact chain."""

        return {
            "ready_for_final_gate": False,
            "review_status": WorkflowStatus.HUMAN_REVIEW,
            "release_allowed": False,
            "final_reason_code": IncomeReviewReasonCode.PROCESSING_FAILURE,
            "failure_code": IncomeReviewFailureCode.FINAL_GATE_FAILED,
        }

    @staticmethod
    def _route_after_extraction(
        state: IncomeReviewState,
    ) -> Literal["verification_input_assemble", "final_gate"]:
        """Block the handoff unless two evidence-guarded extractions exist."""

        return (
            "verification_input_assemble"
            if state.get("extractions_ready") and state.get("failure_code") is None
            else "final_gate"
        )

    @staticmethod
    def _route_after_assembly(
        state: IncomeReviewState,
    ) -> Literal["verification", "final_gate"]:
        """Block Agent 2 unless the strict handoff request is ready."""

        return (
            "verification"
            if state.get("verification_request_ready")
            and state.get("failure_code") is None
            else "final_gate"
        )

    @staticmethod
    def _route_after_verification(
        state: IncomeReviewState,
    ) -> Literal["critic_input_assemble", "final_gate"]:
        """Skip the Critic handoff after an operational Agent 2 failure."""

        return (
            "critic_input_assemble"
            if state.get("verification_result_ready")
            and state.get("ready_for_critic")
            and state.get("failure_code") is None
            else "final_gate"
        )

    @staticmethod
    def _route_after_critic_assembly(
        state: IncomeReviewState,
    ) -> Literal["critic", "final_gate"]:
        """Block Agent 3 unless its reduced, validated request is ready."""

        return (
            "critic"
            if state.get("critic_request_ready")
            and state.get("failure_code") is None
            else "final_gate"
        )

    @staticmethod
    def _to_outcome(state: Mapping[str, object]) -> IncomeReviewOutcome:
        """Convert terminal safe state into the public final review envelope."""

        return IncomeReviewOutcome(
            status=state.get("review_status"),
            release_allowed=state.get("release_allowed", False),
            final_reason_code=state.get("final_reason_code"),
            document_types=tuple(state.get("document_types", ())),
            verification_status=state.get("verification_status"),
            failure_code=state.get("failure_code"),
            upstream_failure_codes=tuple(state.get("upstream_failure_codes", ())),
            handoff_failure_code=state.get("handoff_failure_code"),
            verification_failure_code=state.get("verification_failure_code"),
            critic_handoff_failure_code=state.get("critic_handoff_failure_code"),
            critic_status=state.get("critic_status"),
            critic_disposition=state.get("critic_disposition"),
            critic_reason_code=state.get("critic_reason_code"),
            critic_failure_code=state.get("critic_failure_code"),
            tool_call_count=state.get("tool_call_count", 0),
            invalid_decision_count=state.get("invalid_decision_count", 0),
            critic_attempt_count=state.get("critic_attempt_count", 0),
            critic_repair_count=state.get("critic_repair_count", 0),
        )


class IncomeReviewOrchestrator:
    """Public local-only entry point for the complete three-agent review flow."""

    def __init__(
        self,
        *,
        extractor: DocumentExtractionReviewer,
        verification_agent: IncomeVerificationAgent,
        critic_agent: IncomeReviewCriticAgent,
        trace_sink: ReviewTraceSink | None = None,
    ) -> None:
        """Reject alternate production components at the orchestration boundary."""

        if type(extractor) is not DocumentExtractionReviewer:
            raise TypeError("income review requires the approved Extractor Agent facade")
        if type(verification_agent) is not IncomeVerificationAgent:
            raise TypeError("income review requires the approved Verification Agent")
        if not verification_agent.uses_approved_local_adapter:
            raise TypeError("income review requires the local Ollama decision adapter")
        if type(critic_agent) is not IncomeReviewCriticAgent:
            raise TypeError("income review requires the approved Critic Agent")
        if not critic_agent.uses_approved_local_adapter:
            raise TypeError("income review requires the local Ollama Critic adapter")
        if trace_sink is not None and not isinstance(trace_sink, ReviewTraceSink):
            raise TypeError("income review requires the safe trace sink contract")
        self._trace_sink = trace_sink or NoOpReviewTraceSink()
        self._workflow = _IncomeReviewGraph(
            extractor,
            verification_agent,
            critic_agent,
        )

    @classmethod
    def local(
        cls,
        root: Path,
        *,
        settings: LocalModelSettings | None = None,
        trace_sink: ReviewTraceSink | None = None,
    ) -> "IncomeReviewOrchestrator":
        """Construct all agents around one sealed loopback-only Ollama adapter."""

        safe_settings = settings or LocalModelSettings()
        model = OllamaModel(safe_settings)
        extractor = DocumentExtractionReviewer.local(
            root,
            settings=safe_settings,
            model=model,
        )
        verification_agent = IncomeVerificationAgent(
            OllamaIncomeToolDecisionModel(model)
        )
        critic_agent = IncomeReviewCriticAgent(OllamaCriticDecisionModel(model))
        return cls(
            extractor=extractor,
            verification_agent=verification_agent,
            critic_agent=critic_agent,
            trace_sink=trace_sink,
        )

    @property
    def workflow_node_names(self) -> tuple[str, ...]:
        """Expose the six application nodes for navigation and architecture tests."""

        return self._workflow.node_names

    def review(
        self,
        bundle: IncomeReviewBundle | Mapping[str, object],
    ) -> IncomeReviewOutcome:
        """Trace one complete bundle and return a gated result, not approval."""

        try:
            ensure_cloud_tracing_disabled()
        except UnsafeRuntimeConfigurationError:
            return _human_review_outcome(
                IncomeReviewFailureCode.UNSAFE_CONFIGURATION
            )
        trace_session = ReviewTraceSession(self._trace_sink)
        review_span = trace_session.start_root()
        validation_span = review_span.start_child(
            TraceStage.INCOME_REVIEW_INPUT_VALIDATION
        )
        try:
            validated = IncomeReviewBundle.model_validate(bundle)
        except Exception:
            cause = SanitizedTraceCause(
                origin_stage=TraceStage.INCOME_REVIEW_INPUT_VALIDATION,
                kind=TraceCauseKind.VALIDATION_FAILURE,
                reason_code="invalid_bundle",
            )
            validation_span.finish(TraceSpanOutcome.REJECTED, causes=(cause,))
            outcome = _human_review_outcome(IncomeReviewFailureCode.INVALID_BUNDLE)
            review_span.finish(
                TraceSpanOutcome.HUMAN_REVIEW,
                causes=trace_session.causes,
            )
            return outcome
        validation_span.finish(TraceSpanOutcome.ACCEPTED)
        outcome = self._workflow.invoke(
            validated,
            trace_session=trace_session,
            review_trace_span=review_span,
        )
        if outcome.failure_code is not None and not trace_session.causes:
            cause = SanitizedTraceCause(
                origin_stage=TraceStage.FINANCIAL_REVIEW,
                kind=TraceCauseKind.TECHNICAL_FAILURE,
                reason_code=outcome.failure_code.value,
            )
            causes = (cause,)
        else:
            causes = trace_session.causes
        review_span.finish(
            (
                TraceSpanOutcome.SUCCEEDED
                if outcome.status is WorkflowStatus.RELEASED
                else TraceSpanOutcome.HUMAN_REVIEW
            ),
            causes=causes,
        )
        return outcome


def _human_review_outcome(
    failure_code: IncomeReviewFailureCode,
) -> IncomeReviewOutcome:
    """Build the minimum sanitized terminal outcome for pre-graph failures."""

    return IncomeReviewOutcome(
        status=WorkflowStatus.HUMAN_REVIEW,
        release_allowed=False,
        final_reason_code=IncomeReviewReasonCode.PROCESSING_FAILURE,
        failure_code=failure_code,
    )


__all__ = [
    "INCOME_REVIEW_GRAPH_RECURSION_LIMIT",
    "IncomeReviewBundle",
    "IncomeReviewFailureCode",
    "IncomeReviewOutcome",
    "IncomeReviewOrchestrator",
    "IncomeReviewReasonCode",
    "IncomeReviewState",
]
