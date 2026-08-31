"""Orchestrate one local financial-document review from intake to outcome.

This is the main navigation file for Milestone 1.  External callers enter
through :meth:`DocumentExtractionReviewer.review`; they do not call graph nodes directly.
The reviewer validates and stores the submission before invoking the private
Agent 1 graph. The graph then classifies, deterministically extracts one of the
approved pay-stub, W-2, or bank-statement templates, schema-validates,
evidence-validates, and finalizes.

Two data channels are intentionally kept separate:

``WorkflowState``
    PII-safe control signals used by LangGraph nodes and routing functions.
``_RunArtifacts``
    Document text and extraction objects held locally outside callback-visible
    graph state. Only PII-free unresolved field names/reason codes are shared.

The review trace is a third, control-only execution channel. One random root
trace begins before intake; private span handles are passed through
``_RunArtifacts`` so Agent 1 nodes can form a parent/child hierarchy without
placing trace IDs, document identifiers, or business content in graph state.

Every terminal path fails closed.  A caller receives a released extraction
only when the state authorizes release, the guarded artifact exists, and the
decisive local audit write succeeds.  Agent 2 and Agent 3 appear only as future
data contracts; this graph contains no nodes that implement their behavior.
"""

from __future__ import annotations

import hashlib
import os
import stat
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError

from financial_reviewer.agents.agent1_extraction import (
    DeterministicExtractionError,
    ExtractionValidationError,
    classify_document,
    enforce_evidence_guard,
    extract_bank_statement_deterministically,
    extract_pay_stub_deterministically,
    extract_tax_form_deterministically,
)
from financial_reviewer.foundation.config import (
    LocalModelSettings,
    UnsafeRuntimeConfigurationError,
    ensure_cloud_tracing_disabled,
)
from financial_reviewer.foundation.intake import DocumentSubmission, IntakeError, SecureIntake
from financial_reviewer.agents.agent2_verification import EvidenceLinkedFinding
from financial_reviewer.agents.agent3_critic import CriticDecision
from financial_reviewer.local.model import OllamaModel
from financial_reviewer.local.observability import (
    AuditRecord,
    AuditWriteError,
    LocalJsonlAuditStore,
    PIISafeStructuredLogger,
    SafeEventMetadata,
    SafeLogEvent,
    new_correlation_id,
    new_idempotency_key,
)
from financial_reviewer.foundation.schemas import (
    BankStatementExtraction,
    DocumentClassification,
    DocumentType,
    DeterministicUnresolvedField,
    FailureCode,
    PayStubExtraction,
    ReviewOutcome,
    SCHEMA_VERSION,
    UnsupportedField,
    TaxFormExtraction,
    ValidatedExtraction,
    WorkflowStatus,
    iter_extracted_fields,
)
from financial_reviewer.local.storage import LocalDocumentStore, StorageError, StoredDocument
from financial_reviewer.local.telemetry import (
    DisabledLangSmithTelemetrySink,
    NoOpReviewTraceSink,
    ReviewTraceSession,
    ReviewTraceSink,
    ReviewTraceSpan,
    SanitizedTraceCause,
    SanitizedTelemetryEvent,
    TelemetryEventType,
    TelemetrySink,
    TraceCauseKind,
    TraceSpanOutcome,
    TraceStage,
)


# Recorded in audit metadata so a past decision can be tied to the exact graph
# contract that produced it. Renaming this value is a compatibility decision,
# not merely a display-name change.
WORKFLOW_VERSION = "financial-reviewer-agent1-det-v2"
# LangGraph's independent safety ceiling for node executions/supersteps. This is
# deliberately above the longest valid path but still stops accidental cycles.
GRAPH_RECURSION_LIMIT = 8

# Schema validation independently checks that the extractor returned the
# envelope promised by the classified document type.
_DETERMINISTIC_EXTRACTION_TYPES = {
    DocumentType.BANK_STATEMENT: BankStatementExtraction,
    DocumentType.PAY_STUB: PayStubExtraction,
    DocumentType.TAX_FORM: TaxFormExtraction,
}


class WorkflowState(TypedDict, total=False):
    """PII-safe control signals shared between LangGraph nodes.

    Nodes return partial dictionaries and LangGraph merges those updates into
    this state.  Readiness flags describe which sensitive object exists in the
    separate ``_RunArtifacts`` store; the object itself never appears here.
    Routing methods inspect these flags to choose the next node.
    """

    # Random lookup key for the run's private _RunArtifacts entry.
    run_token: str
    # Classification result used to choose a document-specific schema.
    document_type: DocumentType
    classification: DocumentClassification | None
    # Stage markers used by conditional graph edges.
    deterministic_candidate_ready: bool
    # Safe field names and closed reason codes; values and source text stay private.
    unresolved_fields: tuple[DeterministicUnresolvedField, ...]
    validated_extraction_ready: bool
    schema_valid: bool
    evidence_valid: bool
    # Final release gate; false unless the evidence guard explicitly approves.
    release_allowed: bool
    failure_code: FailureCode | None
    # Audit failure disables every downstream stage and release.
    audit_healthy: bool
    # Reserved contracts only.  Milestone 1 has no Agent 2/Agent 3 nodes.
    verification_findings_ready: bool
    critic_decision_ready: bool


class _CallbackSafeInput(BaseModel):
    """Validate the exact control-state shape allowed to enter LangGraph.

    This second runtime boundary complements the ``WorkflowState`` type hint.
    It rejects extra keys and strict-type mismatches so a caller cannot smuggle
    document text or another sensitive value into graph state where callbacks
    might observe it.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    document_type: DocumentType = DocumentType.UNKNOWN
    classification: DocumentClassification | None = None
    deterministic_candidate_ready: StrictBool = False
    unresolved_fields: tuple[DeterministicUnresolvedField, ...] = ()
    validated_extraction_ready: StrictBool = False
    schema_valid: StrictBool = False
    evidence_valid: StrictBool = False
    release_allowed: StrictBool = False
    failure_code: FailureCode | None = None
    audit_healthy: StrictBool = True
    verification_findings_ready: StrictBool = False
    critic_decision_ready: StrictBool = False


class BoundaryState(TypedDict, total=False):
    """Opaque identifiers available at the reviewer/audit boundary only.

    These values are needed to correlate local audit events before and after
    graph execution.  They are deliberately not part of ``WorkflowState``.
    """

    correlation_id: str = field(repr=False)
    idempotency_key: str = field(repr=False)
    document_id: str = field(repr=False)
    document_type: DocumentType


@dataclass
class _RunArtifacts:
    """Sensitive working data for one graph invocation.

    ``_Agent1Graph.invoke`` creates one instance under a random ``run_token``.
    Nodes retrieve it through ``_artifact`` while sharing only safe flags via
    ``WorkflowState``.  The entry is removed in a ``finally`` block, including
    failed runs, so artifacts cannot leak into a later invocation.
    """

    correlation_id: str
    idempotency_key: str
    document_id: str
    document_text: str = field(repr=False)
    trace_session: ReviewTraceSession = field(repr=False)
    agent_1_trace_span: ReviewTraceSpan = field(repr=False)
    unresolved_fields: tuple[DeterministicUnresolvedField, ...] = field(
        default=(),
        repr=False,
    )
    candidate_extraction: ValidatedExtraction | None = field(default=None, repr=False)
    validated_extraction: ValidatedExtraction | None = field(default=None, repr=False)
    verification_findings: tuple[EvidenceLinkedFinding, ...] | None = field(
        default=None,
        repr=False,
    )
    critic_decision: CriticDecision | None = field(default=None, repr=False)


# Translate internal failure enums into the closed, PII-safe vocabulary that
# the logging and audit schemas permit.
_OBSERVABILITY_ERROR_CODES: dict[FailureCode, str] = {
    FailureCode.INVALID_INPUT: "INVALID_INPUT",
    FailureCode.UNSUPPORTED_DOCUMENT_TYPE: "UNSUPPORTED_DOCUMENT_TYPE",
    FailureCode.LOCAL_MODEL_ERROR: "LOCAL_MODEL_ERROR",
    FailureCode.INVALID_MODEL_OUTPUT: "INVALID_MODEL_OUTPUT",
    FailureCode.SCHEMA_VALIDATION_FAILED: "SCHEMA_VALIDATION_FAILED",
    FailureCode.EVIDENCE_VALIDATION_FAILED: "EVIDENCE_VALIDATION_FAILED",
    FailureCode.UNSUPPORTED_REQUIRED_FIELD: "UNSUPPORTED_REQUIRED_FIELD",
    FailureCode.LOCAL_STORAGE_ERROR: "LOCAL_STORAGE_ERROR",
    FailureCode.AUDIT_WRITE_FAILED: "AUDIT_WRITE_FAILED",
    FailureCode.UNSAFE_CONFIGURATION: "UNSAFE_CONFIGURATION",
    FailureCode.INTERNAL_FAILURE: "INTERNAL_FAILURE",
}

_TECHNICAL_TRACE_FAILURES = frozenset(
    {
        FailureCode.LOCAL_MODEL_ERROR,
        FailureCode.LOCAL_STORAGE_ERROR,
        FailureCode.AUDIT_WRITE_FAILED,
        FailureCode.UNSAFE_CONFIGURATION,
        FailureCode.INTERNAL_FAILURE,
    }
)
_VALIDATION_TRACE_FAILURES = frozenset(
    {
        FailureCode.INVALID_INPUT,
        FailureCode.INVALID_MODEL_OUTPUT,
        FailureCode.SCHEMA_VALIDATION_FAILED,
        FailureCode.EVIDENCE_VALIDATION_FAILED,
    }
)


def _trace_cause(
    stage: TraceStage,
    failure_code: FailureCode,
) -> SanitizedTraceCause:
    """Translate one internal failure into the closed trace-causality schema."""

    if failure_code in _TECHNICAL_TRACE_FAILURES:
        kind = TraceCauseKind.TECHNICAL_FAILURE
    elif failure_code in _VALIDATION_TRACE_FAILURES:
        kind = TraceCauseKind.VALIDATION_FAILURE
    else:
        kind = TraceCauseKind.POLICY_BLOCK
    return SanitizedTraceCause(
        origin_stage=stage,
        kind=kind,
        reason_code=failure_code.value,
    )


def _trace_outcome_for_failure(failure_code: FailureCode) -> TraceSpanOutcome:
    """Select a stable operational outcome without interpreting business data."""

    if failure_code is FailureCode.INVALID_INPUT:
        return TraceSpanOutcome.REJECTED
    if failure_code is FailureCode.UNSUPPORTED_DOCUMENT_TYPE:
        return TraceSpanOutcome.UNSUPPORTED
    if failure_code in _TECHNICAL_TRACE_FAILURES:
        return TraceSpanOutcome.FAILED
    return TraceSpanOutcome.BLOCKED


def _trace_causes_with(
    existing: tuple[SanitizedTraceCause, ...],
    stage: TraceStage,
    failure_code: FailureCode,
    *,
    propagated: bool = False,
) -> tuple[SanitizedTraceCause, ...]:
    """Add one origin, or reuse a matching child cause during propagation."""

    if propagated and any(
        cause.reason_code == failure_code.value for cause in existing
    ):
        return existing
    candidate = _trace_cause(stage, failure_code)
    key = (candidate.origin_stage, candidate.kind, candidate.reason_code)
    if any(
        (cause.origin_stage, cause.kind, cause.reason_code) == key
        for cause in existing
    ):
        return existing
    return (*existing, candidate)

class _Agent1Graph:
    """Private LangGraph runtime for the implemented Agent 1 workflow.

    The class owns graph construction, node methods, conditional routing, and
    node-level audit/telemetry emission.  It does
    not own input validation or durable document storage; those happen in
    ``DocumentExtractionReviewer.review`` before this graph can run.
    """

    def __init__(
        self,
        *,
        audit_store: LocalJsonlAuditStore,
        logger: PIISafeStructuredLogger,
        telemetry: TelemetrySink,
    ) -> None:
        """Bind validated local dependencies and compile the graph once.

        ``DocumentExtractionReviewer`` is the production caller.  The in-memory artifact
        dictionary is private to this graph instance and protected because
        multiple local review calls may execute concurrently.
        """

        self._audit = audit_store
        self._logger = logger
        self._telemetry = telemetry
        self._artifact_lock = threading.RLock()
        self._artifacts: dict[str, _RunArtifacts] = {}
        self._graph = self._build_graph()

    @property
    def node_names(self) -> tuple[str, ...]:
        """Return application node names without LangGraph's START/END nodes."""

        nodes = self._graph.get_graph().nodes
        return tuple(name for name in nodes if name not in {START, END, "__start__", "__end__"})

    def invoke(
        self,
        state: WorkflowState,
        *,
        correlation_id: str,
        idempotency_key: str,
        document_id: str,
        document_text: str,
        trace_session: ReviewTraceSession,
        agent_1_trace_span: ReviewTraceSpan,
    ) -> tuple[WorkflowState, ValidatedExtraction | None]:
        """Run the compiled graph with safe state and private local artifacts.

        Called by:
            ``DocumentExtractionReviewer.review`` after intake, storage, and storage
            integrity verification succeed.

        Reads:
            Initial PII-safe ``WorkflowState`` plus opaque identifiers and
            document text supplied separately by the reviewer.

        Writes:
            A temporary ``_RunArtifacts`` entry keyed by a random run token.
            Graph nodes mutate that artifact and return safe state updates.

        Returns:
            Terminal graph state and the guarded extraction only when both
            release flags are true.  The temporary artifact entry is removed
            on every exit path.
        """

        ensure_cloud_tracing_disabled()
        safe_state = self._normalize_graph_input(state)
        run_token = f"run_{uuid4().hex}"
        try:
            with self._artifact_lock:
                if run_token in self._artifacts:
                    raise RuntimeError("duplicate active workflow identifier")
                self._artifacts[run_token] = _RunArtifacts(
                    correlation_id=correlation_id,
                    idempotency_key=idempotency_key,
                    document_id=document_id,
                    document_text=document_text,
                    trace_session=trace_session,
                    agent_1_trace_span=agent_1_trace_span,
                )
            final_state = self._graph.invoke(
                {**safe_state, "run_token": run_token},
                config={
                    "callbacks": [],
                    "recursion_limit": GRAPH_RECURSION_LIMIT,
                },
            )
            artifact = self._artifact(final_state)
            extraction = (
                artifact.validated_extraction
                if final_state.get("release_allowed")
                and final_state.get("validated_extraction_ready")
                else None
            )
            return final_state, extraction
        finally:
            with self._artifact_lock:
                self._artifacts.pop(run_token, None)

    @staticmethod
    def _normalize_graph_input(state: WorkflowState) -> WorkflowState:
        """Reject undeclared or incorrectly typed graph input fields.

        This check runs immediately before graph invocation.  It prevents
        sensitive fields such as ``document_text`` or ``raw_model_output`` from
        entering callback-visible state even if a caller bypasses type hints.
        """

        allowed_fields = set(_CallbackSafeInput.model_fields)
        if set(state) - allowed_fields:
            raise RuntimeError("workflow state contains undeclared fields")
        try:
            validated = _CallbackSafeInput.model_validate(dict(state))
        except (ValidationError, TypeError, ValueError):
            raise RuntimeError("workflow state failed safety validation") from None
        return validated.model_dump(mode="python", warnings="none")  # type: ignore[return-value]

    def _artifact(self, state: WorkflowState) -> _RunArtifacts:
        """Resolve the current run's private artifacts from its safe token.

        Every node calls this helper before reading sensitive data.  A missing,
        malformed, expired, or unknown token is treated as an internal failure
        rather than allowing a node to continue without its evidence context.
        """

        run_token = state.get("run_token")
        if not isinstance(run_token, str):
            raise RuntimeError("missing workflow identifier")
        with self._artifact_lock:
            artifact = self._artifacts.get(run_token)
        if artifact is None:
            raise RuntimeError("missing workflow artifacts")
        return artifact

    def _build_graph(self):
        """Compile the five-node Agent 1 graph and its conditional routes.

        Happy path:
            START -> classify -> deterministic_extract -> schema_validate -> evidence_guard
            -> finalize -> END.

        Terminal path:
            Unsupported document types, unresolved deterministic fields, audit
            failure, or a successful evidence decision routes to ``finalize``.
            This first slice has no model node or retry edge. Checkpointing is
            disabled because sensitive state must remain local and ephemeral.
        """

        builder = StateGraph(WorkflowState)
        builder.add_node(
            "classify",
            self._traced_node(TraceStage.CLASSIFICATION, self._classify),
        )
        builder.add_node(
            "deterministic_extract",
            self._traced_node(
                TraceStage.DETERMINISTIC_EXTRACTION,
                self._deterministic_extract,
            ),
        )
        builder.add_node(
            "schema_validate",
            self._traced_node(TraceStage.SCHEMA_VALIDATION, self._schema_validate),
        )
        builder.add_node(
            "evidence_guard",
            self._traced_node(TraceStage.EVIDENCE_GUARD, self._evidence_guard),
        )
        builder.add_node(
            "finalize",
            self._traced_node(TraceStage.AGENT_1_FINALIZE, self._finalize),
        )
        builder.add_edge(START, "classify")
        builder.add_conditional_edges(
            "classify",
            self._route_after_classification,
            {
                "deterministic_extract": "deterministic_extract",
                "finalize": "finalize",
            },
        )
        builder.add_conditional_edges(
            "deterministic_extract",
            self._route_after_deterministic_extraction,
            {
                "schema_validate": "schema_validate",
                "finalize": "finalize",
            },
        )
        builder.add_conditional_edges(
            "schema_validate",
            self._route_after_schema_validation,
            {
                "evidence_guard": "evidence_guard",
                "finalize": "finalize",
            },
        )
        builder.add_conditional_edges(
            "evidence_guard",
            self._route_after_evidence_guard,
            {"finalize": "finalize"},
        )
        builder.add_edge("finalize", END)
        return builder.compile(
            checkpointer=False,
            name="financial_document_extraction",
        )

    def _traced_node(
        self,
        stage: TraceStage,
        node: Callable[[WorkflowState], WorkflowState],
    ) -> Callable[[WorkflowState], WorkflowState]:
        """Wrap one graph node with a child span without changing node logic.

        The wrapper resolves its parent from ``_RunArtifacts`` rather than
        placing trace handles or identifiers in callback-visible graph state.
        It records only the closed stage, outcome, document classification,
        and sanitized failure cause.
        """

        def traced(state: WorkflowState) -> WorkflowState:
            artifact = self._artifact(state)
            span = artifact.agent_1_trace_span.start_child(stage)
            try:
                update = node(state)
            except Exception:
                cause = _trace_cause(stage, FailureCode.INTERNAL_FAILURE)
                span.finish(TraceSpanOutcome.FAILED, causes=(cause,))
                raise

            resulting_state: WorkflowState = {**state, **update}
            failure_code = resulting_state.get("failure_code")
            document_type = resulting_state.get(
                "document_type",
                DocumentType.UNKNOWN,
            ).value
            if failure_code is None:
                outcome = TraceSpanOutcome.SUCCEEDED
                causes: tuple[SanitizedTraceCause, ...] = ()
            else:
                outcome = (
                    TraceSpanOutcome.HUMAN_REVIEW
                    if stage is TraceStage.AGENT_1_FINALIZE
                    else _trace_outcome_for_failure(failure_code)
                )
                existing_causes = (
                    artifact.trace_session.causes
                    if stage is TraceStage.AGENT_1_FINALIZE
                    else ()
                )
                causes = _trace_causes_with(
                    existing_causes,
                    stage,
                    failure_code,
                    propagated=stage is TraceStage.AGENT_1_FINALIZE,
                )
            span.finish(
                outcome,
                document_type=document_type,
                causes=causes,
            )
            return update

        return traced

    def _classify(self, state: WorkflowState) -> WorkflowState:
        """Classify the stored synthetic document before deterministic extraction.

        Called by:
            LangGraph directly after START.

        Reads:
            ``document_text`` from ``_RunArtifacts``.

        Writes:
            ``document_type`` and ``classification`` to safe graph state.

        Routes:
            A recognized type proceeds to ``deterministic_extract``. Unknown
            content records a sanitized failure and proceeds to ``finalize``.
            The extractor then enforces the explicit three-template allowlist.
        """

        if not self._record(
            state,
            component="agent_1_extraction",
            action="classification_started",
            status="started",
        ):
            return self._audit_failure()
        try:
            document_text = self._artifact(state).document_text
        except RuntimeError:
            return self._safe_stop(FailureCode.INTERNAL_FAILURE)
        document_type, classification = classify_document(document_text)
        if document_type is DocumentType.UNKNOWN or classification is None:
            update = self._safe_stop(FailureCode.UNSUPPORTED_DOCUMENT_TYPE)
            update["document_type"] = DocumentType.UNKNOWN
            if not self._record(
                {**state, **update},
                component="agent_1_extraction",
                action="classification_completed",
                status="unsupported",
            ):
                return self._audit_failure()
            return update
        update: WorkflowState = {
            "document_type": document_type,
            "classification": classification,
            "failure_code": None,
        }
        if not self._record(
            {**state, **update},
            component="agent_1_extraction",
            action="classification_completed",
            status="succeeded",
        ):
            return self._audit_failure()
        return update

    def _deterministic_extract(self, state: WorkflowState) -> WorkflowState:
        """Extract one approved synthetic template without calling a model.

        Called by:
            The classification success route.

        Reads:
            ``document_type`` and ``classification`` from safe graph state;
            document ID and text from the private ``_RunArtifacts`` entry.

        Writes:
            Stores a typed candidate with code-owned provenance in private
            artifacts.  On failure, shares only safe field names and closed
            unresolved reasons through ``WorkflowState``.

        Routes:
            A complete candidate proceeds to ``schema_validate``.  Unresolved
            fields and document types outside the approved templates finalize
            to human review with no LLM fallback or retry.
        """

        if not self._record(
            state,
            component="agent_1_extraction",
            action="extraction_started",
            status="started",
        ):
            return self._audit_failure()
        try:
            artifact = self._artifact(state)
            document_type = state.get("document_type")
            classification = state.get("classification")
            # Resolve the entry method at call time so tests can replace one
            # document parser without affecting the other approved types.
            if document_type is DocumentType.BANK_STATEMENT:
                extractor = extract_bank_statement_deterministically
            elif document_type is DocumentType.PAY_STUB:
                extractor = extract_pay_stub_deterministically
            elif document_type is DocumentType.TAX_FORM:
                extractor = extract_tax_form_deterministically
            else:
                extractor = None
            if extractor is None or classification is None:
                failure = FailureCode.UNSUPPORTED_DOCUMENT_TYPE
                update = {
                    **self._safe_stop(failure),
                    "deterministic_candidate_ready": False,
                    "unresolved_fields": (),
                }
                artifact.candidate_extraction = None
                artifact.unresolved_fields = ()
                if not self._record(
                    {**state, **update},
                    component="agent_1_extraction",
                    action="extraction_failed",
                    status="unsupported",
                    failure_code=failure,
                ):
                    update.update(self._audit_failure())
                return update
            candidate = extractor(
                document_id=artifact.document_id,
                document_text=artifact.document_text,
                classification=classification,
            )
            # Recheck after the sensitive extraction step. A debugger or
            # concurrent process must not be able to enable cloud tracing
            # mid-run and still permit this candidate to move toward release.
            ensure_cloud_tracing_disabled()
        except UnsafeRuntimeConfigurationError:
            failure = FailureCode.UNSAFE_CONFIGURATION
            update = {
                **self._safe_stop(failure),
                "deterministic_candidate_ready": False,
                "unresolved_fields": (),
            }
            try:
                artifact = self._artifact(state)
                artifact.candidate_extraction = None
                artifact.unresolved_fields = ()
            except RuntimeError:
                pass
            if not self._record(
                {**state, **update},
                component="agent_1_extraction",
                action="extraction_failed",
                status="blocked",
                failure_code=failure,
            ):
                update.update(self._audit_failure())
            return update
        except DeterministicExtractionError as exc:
            failure = _failure_from_extraction_error(exc)
            unresolved = exc.unresolved_fields
            update: WorkflowState = {
                **self._safe_stop(failure),
                "deterministic_candidate_ready": False,
                "unresolved_fields": unresolved,
            }
            try:
                artifact = self._artifact(state)
                artifact.candidate_extraction = None
                artifact.unresolved_fields = unresolved
            except RuntimeError:
                pass
            if not self._record(
                {**state, **update},
                component="agent_1_extraction",
                action="extraction_failed",
                status="blocked",
                failure_code=failure,
                unsupported_field_count=len(unresolved),
                validation_error_count=len(unresolved),
            ):
                update.update(self._audit_failure())
            return update
        except ExtractionValidationError as exc:
            failure = _failure_from_extraction_error(exc)
            update = {
                **self._safe_stop(failure),
                "deterministic_candidate_ready": False,
                "unresolved_fields": (),
            }
            if not self._record(
                {**state, **update},
                component="agent_1_extraction",
                action="extraction_failed",
                status="blocked",
                failure_code=failure,
                validation_error_count=1,
            ):
                update.update(self._audit_failure())
            return update
        except RuntimeError:
            return self._safe_stop(FailureCode.INTERNAL_FAILURE)

        field_count, supported_count, unsupported_count, evidence_count = _field_counts(candidate)
        artifact.candidate_extraction = candidate
        artifact.unresolved_fields = ()
        update = {
            "deterministic_candidate_ready": True,
            "unresolved_fields": (),
            "failure_code": None,
            "schema_valid": False,
            "evidence_valid": False,
        }
        if not self._record(
            {**state, **update},
            component="agent_1_extraction",
            action="extraction_completed",
            status="succeeded",
            field_count=field_count,
            supported_field_count=supported_count,
            unsupported_field_count=unsupported_count,
            evidence_count=evidence_count,
        ):
            artifact.candidate_extraction = None
            return self._audit_failure()
        return update

    def _schema_validate(self, state: WorkflowState) -> WorkflowState:
        """Revalidate the deterministic candidate at an explicit graph gate.

        Called by:
            The route after ``deterministic_extract`` reports a complete candidate.

        Reads:
            Document type and classification from state; typed candidate and
            document ID from ``_RunArtifacts``.

        Writes:
            Stores a newly revalidated immutable candidate and sets
            ``schema_valid`` while preserving deterministic readiness.

        Routes:
            Success proceeds to ``evidence_guard``.  Any type, identity,
            classification, or extraction-method mismatch finalizes to human
            review without retry or model fallback.
        """

        if not self._record(
            state,
            component="schema_validator",
            action="validation_started",
            status="started",
        ):
            return self._audit_failure()
        try:
            artifact = self._artifact(state)
            candidate = artifact.candidate_extraction
            document_type = state.get("document_type")
            classification = state.get("classification")
            candidate_type = _DETERMINISTIC_EXTRACTION_TYPES.get(document_type)
            if (
                not state.get("deterministic_candidate_ready")
                or candidate_type is None
                or classification is None
                or not isinstance(candidate, candidate_type)
            ):
                raise ExtractionValidationError("schema_validation_failed")
            try:
                candidate = candidate_type.model_validate(
                    candidate.model_dump(mode="python", warnings="none")
                )
            except (ValidationError, TypeError, ValueError):
                raise ExtractionValidationError("schema_validation_failed") from None
            if (
                candidate.document_id != artifact.document_id
                or candidate.classification != classification
                or candidate.metadata.extraction_method != "deterministic_labels_v1"
                or candidate.metadata.attempt_count != 0
            ):
                raise ExtractionValidationError("schema_validation_failed")
        except ExtractionValidationError as exc:
            failure = _failure_from_extraction_error(exc)
            update = {
                **self._safe_stop(failure),
                "deterministic_candidate_ready": False,
                "schema_valid": False,
            }
            try:
                artifact = self._artifact(state)
                artifact.candidate_extraction = None
            except RuntimeError:
                pass
            if not self._record(
                {**state, **update},
                component="schema_validator",
                action="validation_completed",
                status="failed",
                failure_code=failure,
                validation_error_count=1,
            ):
                update.update(self._audit_failure())
            return update

        field_count, supported_count, unsupported_count, evidence_count = _field_counts(candidate)
        artifact.candidate_extraction = candidate
        update: WorkflowState = {
            "deterministic_candidate_ready": True,
            "schema_valid": True,
            "failure_code": None,
        }
        if not self._record(
            {**state, **update},
            component="schema_validator",
            action="validation_completed",
            status="succeeded",
            field_count=field_count,
            supported_field_count=supported_count,
            unsupported_field_count=unsupported_count,
            evidence_count=evidence_count,
        ):
            return self._audit_failure()
        return update

    def _evidence_guard(self, state: WorkflowState) -> WorkflowState:
        """Make the decisive, deterministic Agent 1 release decision.

        Called by:
            The route after a typed candidate passes schema validation.

        Reads:
            Candidate extraction and original document text from
            ``_RunArtifacts``.

        Writes:
            On success, replaces the candidate with a guarded validated
            extraction, clears the temporary candidate, and sets
            ``evidence_valid``, ``validated_extraction_ready``, and
            ``release_allowed``.

        Routes:
            Success proceeds to ``finalize``.  Missing, unsupported, altered,
            or mismatched evidence fails closed directly to human review.
        """

        if not self._record(
            state,
            component="evidence_guard",
            action="evidence_check_started",
            status="started",
        ):
            return self._audit_failure()
        try:
            artifact = self._artifact(state)
            candidate = artifact.candidate_extraction
            document_text = artifact.document_text
            if candidate is None:
                raise ExtractionValidationError("evidence_validation_failed")
            guarded_extraction = enforce_evidence_guard(candidate, document_text)
        except ExtractionValidationError as exc:
            failure = _failure_from_extraction_error(exc)
            update = {
                **self._safe_stop(failure),
                "deterministic_candidate_ready": False,
                "validated_extraction_ready": False,
                "evidence_valid": False,
            }
            try:
                artifact = self._artifact(state)
                artifact.candidate_extraction = None
                artifact.validated_extraction = None
            except RuntimeError:
                pass
            if not self._record(
                {**state, **update},
                component="evidence_guard",
                action="evidence_check_completed",
                status="blocked",
                failure_code=failure,
            ):
                update.update(self._audit_failure())
            return update

        field_count, supported_count, unsupported_count, evidence_count = _field_counts(candidate)
        artifact.validated_extraction = guarded_extraction
        artifact.candidate_extraction = None
        update: WorkflowState = {
            "validated_extraction_ready": True,
            "deterministic_candidate_ready": False,
            "evidence_valid": True,
            "release_allowed": True,
            "failure_code": None,
        }
        if not self._record(
            {**state, **update},
            component="evidence_guard",
            action="release_approved",
            status="succeeded",
            field_count=field_count,
            supported_field_count=supported_count,
            unsupported_field_count=unsupported_count,
            evidence_count=evidence_count,
        ):
            return self._audit_failure()
        return update

    def _finalize(self, state: WorkflowState) -> WorkflowState:
        """Reconcile terminal state, artifact presence, audit, and release.

        Called by:
            Every successful or failed terminal route.

        Reads:
            Release flags, failure code, audit health, and the
            presence of a guarded extraction in ``_RunArtifacts``.

        Writes:
            A terminal local audit event and the final safe state.  Unreleased
            runs have all extraction artifacts cleared.

        Release rule:
            State authorization and a guarded artifact must agree, and the
            terminal audit/telemetry boundaries must succeed.  Any disagreement
            converts the result to human review.
        """

        try:
            artifact = self._artifact(state)
            guarded_extraction_ready = artifact.validated_extraction is not None
        except RuntimeError:
            artifact = None
            guarded_extraction_ready = False
        release = (
            bool(state.get("release_allowed"))
            and bool(state.get("validated_extraction_ready"))
            and guarded_extraction_ready
        )
        failure = state.get("failure_code")
        final_status = "succeeded" if release else "human_review"
        final_failure = None if release else (failure or FailureCode.INTERNAL_FAILURE)
        final_for_event = {
            **state,
            "failure_code": final_failure,
            "release_allowed": release,
        }
        if not self._record(
            final_for_event,
            component="workflow",
            action="workflow_completed",
            status=final_status,
            failure_code=final_failure,
            retry_count=0,
        ):
            release = False
            final_failure = FailureCode.AUDIT_WRITE_FAILED

        # External telemetry remains disabled in this milestone. Even a future
        # sanitized sink may observe success only after the decisive local audit
        # record is durably verified.
        if (
            release
            and final_failure is None
            and not self._emit_telemetry(
                state,
                event_type=TelemetryEventType.WORKFLOW_COMPLETED,
                component="workflow",
                outcome="succeeded",
            )
        ):
            release = False
            final_failure = FailureCode.UNSAFE_CONFIGURATION
            if not self._record(
                {**state, "release_allowed": False},
                component="telemetry",
                action="release_blocked",
                status="blocked",
                failure_code=final_failure,
            ):
                final_failure = FailureCode.AUDIT_WRITE_FAILED

        if not release and artifact is not None:
            artifact.candidate_extraction = None
            artifact.validated_extraction = None

        return {
            "deterministic_candidate_ready": False,
            "validated_extraction_ready": release,
            "release_allowed": release,
            "failure_code": final_failure,
            "audit_healthy": final_failure is not FailureCode.AUDIT_WRITE_FAILED,
            "verification_findings_ready": False,
            "critic_decision_ready": False,
        }

    def _route_after_classification(
        self,
        state: WorkflowState,
    ) -> Literal["deterministic_extract", "finalize"]:
        """Send a classified document to the bounded deterministic extractor."""

        if state.get("failure_code") is not None or state.get("classification") is None:
            return "finalize"
        return "deterministic_extract"

    def _route_after_deterministic_extraction(
        self,
        state: WorkflowState,
    ) -> Literal["schema_validate", "finalize"]:
        """Validate a complete candidate or stop unresolved extraction safely."""

        if state.get("deterministic_candidate_ready"):
            return "schema_validate"
        return "finalize"

    def _route_after_schema_validation(
        self,
        state: WorkflowState,
    ) -> Literal["evidence_guard", "finalize"]:
        """Send a revalidated deterministic candidate to evidence checking."""

        if state.get("schema_valid") and state.get("deterministic_candidate_ready"):
            return "evidence_guard"
        return "finalize"

    def _route_after_evidence_guard(
        self,
        state: WorkflowState,
    ) -> Literal["finalize"]:
        """Finalize every evidence decision; this slice has no retry edge."""

        return "finalize"

    @staticmethod
    def _safe_stop(failure_code: FailureCode) -> WorkflowState:
        """Return the minimum state update that closes the release gate."""

        return {
            "release_allowed": False,
            "validated_extraction_ready": False,
            "failure_code": failure_code,
        }

    @staticmethod
    def _audit_failure() -> WorkflowState:
        """Return terminal fail-closed state for an unavailable local audit."""

        return {
            "audit_healthy": False,
            "release_allowed": False,
            "validated_extraction_ready": False,
            "deterministic_candidate_ready": False,
            "failure_code": FailureCode.AUDIT_WRITE_FAILED,
        }

    def _record(
        self,
        state: WorkflowState,
        *,
        component: str,
        action: str,
        status: str,
        failure_code: FailureCode | None = None,
        **counts: int,
    ) -> bool:
        """Write and verify one PII-safe node-level local audit event.

        The event schema accepts only opaque identifiers, safe versions,
        statuses, failure codes, and numeric counts.  Any construction, logging,
        persistence, or read-back error returns ``False`` so the caller can
        block release without exposing the underlying exception text.
        """

        try:
            artifact = self._artifact(state)
            metadata = SafeEventMetadata(
                correlation_id=artifact.correlation_id,
                idempotency_key=artifact.idempotency_key,
                opaque_document_id=artifact.document_id,
                component=component,
                action=action,
                status=status,
                error_code=(
                    _OBSERVABILITY_ERROR_CODES[failure_code]
                    if failure_code is not None
                    else None
                ),
                document_type=state.get("document_type", DocumentType.UNKNOWN).value,
                extraction_schema_version=SCHEMA_VERSION,
                workflow_version=WORKFLOW_VERSION,
                **counts,
            )
            event = PIISafeStructuredLogger.emit(self._logger, metadata)
            if event.metadata != metadata:
                return False
            if not _append_verified_local_audit(self._audit, event):
                return False
            return True
        except Exception:
            return False

    def _emit_telemetry(
        self,
        state: WorkflowState,
        *,
        event_type: TelemetryEventType,
        component: str,
        outcome: str,
        failure_code: FailureCode | None = None,
    ) -> bool:
        """Emit one sanitized event through the configured telemetry boundary.

        Milestone 1 supplies ``DisabledLangSmithTelemetrySink``, so the normal
        behavior is a validated no-op.  Returning ``False`` lets finalization
        fail closed if a future or substituted sink violates this contract.
        """

        try:
            event = SanitizedTelemetryEvent(
                event_type=event_type,
                component=component,
                outcome=outcome,
                reason_code=failure_code.value if failure_code else None,
                document_type=state.get("document_type", DocumentType.UNKNOWN).value,
                schema_version=SCHEMA_VERSION,
            )
            self._telemetry.emit(event)
            return True
        except Exception:
            return False


class DocumentExtractionReviewer:
    """Secure application facade for one complete Milestone 1 review.

    This is the class an eventual CLI, API, or desktop adapter should call.  It
    owns the pre-graph boundary: dependency validation, synthetic-only intake,
    private local storage, boundary audit events, and conversion of terminal
    graph state into ``ReviewOutcome``.  It never exposes an unguarded candidate.
    """

    def __init__(
        self,
        *,
        settings: LocalModelSettings,
        document_store: LocalDocumentStore,
        audit_store: LocalJsonlAuditStore,
        model: OllamaModel | None = None,
        intake: SecureIntake | None = None,
        logger: PIISafeStructuredLogger | None = None,
        telemetry: TelemetrySink | None = None,
        trace_sink: ReviewTraceSink | None = None,
    ) -> None:
        """Validate and bind only approved local production components.

        Exact concrete types are required at this boundary so an alternate
        adapter cannot quietly introduce cloud inference, unsafe logging, or a
        different storage implementation.  Optional dependencies receive safe
        local defaults, including disabled external telemetry.
        """

        ensure_cloud_tracing_disabled()
        if type(settings) is not LocalModelSettings:
            raise TypeError("Milestone 1 requires the validated local model settings.")
        try:
            settings = LocalModelSettings.model_validate(
                settings.model_dump(mode="python", warnings="none")
            )
        except (AttributeError, TypeError, ValueError):
            raise UnsafeRuntimeConfigurationError(
                "The local model settings failed security validation."
            ) from None
        if type(document_store) is not LocalDocumentStore:
            raise TypeError("Milestone 1 requires the local document store.")
        if type(audit_store) is not LocalJsonlAuditStore:
            raise TypeError("Milestone 1 requires the local JSONL audit store.")
        if logger is not None and type(logger) is not PIISafeStructuredLogger:
            raise TypeError("Milestone 1 requires the PII-safe structured logger.")
        if intake is not None and type(intake) is not SecureIntake:
            raise TypeError("Milestone 1 requires the secure local intake boundary.")
        if model is not None and type(model) is not OllamaModel:
            raise TypeError("Milestone 1 requires the loopback-only Ollama adapter.")
        if model is not None and model.settings != settings:
            raise UnsafeRuntimeConfigurationError(
                "The model adapter does not satisfy the production local boundary."
            )
        if trace_sink is not None and not isinstance(trace_sink, ReviewTraceSink):
            raise TypeError("The trace adapter must implement ReviewTraceSink.")
        self._settings = settings
        self._store = document_store
        self._audit = audit_store
        self._model = model or OllamaModel(settings)
        self._intake = intake or SecureIntake()
        self._logger = logger or PIISafeStructuredLogger()
        self._telemetry = telemetry or DisabledLangSmithTelemetrySink()
        self._trace_sink = trace_sink or NoOpReviewTraceSink()
        self._workflow = _Agent1Graph(
            audit_store=audit_store,
            logger=self._logger,
            telemetry=self._telemetry,
        )

    @classmethod
    def local(
        cls,
        root: Path,
        *,
        settings: LocalModelSettings | None = None,
        model: OllamaModel | None = None,
        trace_sink: ReviewTraceSink | None = None,
    ) -> "DocumentExtractionReviewer":
        """Create a reviewer under one owner-only local runtime directory.

        The factory rejects symlinks, non-directories, foreign ownership, and
        unsafe existing permissions.  It creates separate document and audit
        locations and then delegates component validation to the constructor.
        """

        existed = root.exists()
        try:
            if root.is_symlink():
                raise StorageError()
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            root_metadata = root.stat()
            if not stat.S_ISDIR(root_metadata.st_mode):
                raise StorageError()
            if hasattr(os, "getuid") and root_metadata.st_uid != os.getuid():
                raise StorageError()
            if stat.S_IMODE(root_metadata.st_mode) != 0o700:
                if existed:
                    raise StorageError()
                root.chmod(0o700)
            root = root.resolve(strict=True)
        except OSError:
            raise StorageError() from None
        settings = settings or LocalModelSettings()
        document_store = LocalDocumentStore(root / "documents")
        audit_store = LocalJsonlAuditStore(root / "audit" / "events.jsonl")
        return cls(
            settings=settings,
            document_store=document_store,
            audit_store=audit_store,
            model=model,
            trace_sink=trace_sink,
        )

    @property
    def workflow_node_names(self) -> tuple[str, ...]:
        """Expose graph node names for diagnostics and architecture tests."""

        return self._workflow.node_names

    def review(
        self,
        submission: DocumentSubmission | Mapping[str, Any],
    ) -> ReviewOutcome:
        """Run one submission inside a single root trace from entry to decision.

        Trace identifiers are generated randomly and never enter workflow
        state, model context, audit metadata, or the public outcome.  The root
        and final-decision spans close for every handled review result,
        including failures before an agent is invoked.
        """

        trace_session = ReviewTraceSession(self._trace_sink)
        review_span = trace_session.start_root()
        try:
            outcome = self.review_with_trace_parent(
                submission,
                trace_session=trace_session,
                parent_span=review_span,
            )
        except Exception:
            review_span.finish(
                TraceSpanOutcome.FAILED,
                causes=trace_session.causes,
            )
            raise

        terminal_outcome = (
            TraceSpanOutcome.SUCCEEDED
            if outcome.status is WorkflowStatus.RELEASED
            else TraceSpanOutcome.HUMAN_REVIEW
        )
        review_span.finish(
            terminal_outcome,
            document_type=outcome.document_type.value,
            causes=trace_session.causes,
        )
        return outcome

    def review_with_trace_parent(
        self,
        submission: DocumentSubmission | Mapping[str, Any],
        *,
        trace_session: ReviewTraceSession,
        parent_span: ReviewTraceSpan,
    ) -> ReviewOutcome:
        """Review one document beneath an existing private trace parent.

        The multi-agent orchestrator uses this composition boundary to retain
        one trace ID for the complete two-document review. Trace/session
        handles remain private and never enter LangGraph state, model context,
        audit metadata, or the public outcome.
        """

        if not isinstance(trace_session, ReviewTraceSession):
            raise TypeError("the trace session is invalid")
        if not isinstance(parent_span, ReviewTraceSpan):
            raise TypeError("the trace parent is invalid")
        cause_keys_before = {
            (cause.origin_stage, cause.kind, cause.reason_code)
            for cause in trace_session.causes
        }
        try:
            outcome = self._review_with_trace(
                submission,
                trace_session=trace_session,
                review_span=parent_span,
            )
        except Exception:
            cause = _trace_cause(
                TraceStage.FINAL_REVIEW_DECISION,
                FailureCode.INTERNAL_FAILURE,
            )
            final_span = parent_span.start_child(TraceStage.FINAL_REVIEW_DECISION)
            final_span.finish(TraceSpanOutcome.FAILED, causes=(cause,))
            raise

        causes = tuple(
            cause
            for cause in trace_session.causes
            if (cause.origin_stage, cause.kind, cause.reason_code)
            not in cause_keys_before
        )
        if outcome.failure_code is not None and not causes:
            causes = (
                _trace_cause(
                    TraceStage.FINAL_REVIEW_DECISION,
                    outcome.failure_code,
                ),
            )
        terminal_outcome = (
            TraceSpanOutcome.SUCCEEDED
            if outcome.status is WorkflowStatus.RELEASED
            else TraceSpanOutcome.HUMAN_REVIEW
        )
        final_span = parent_span.start_child(TraceStage.FINAL_REVIEW_DECISION)
        final_span.finish(
            terminal_outcome,
            document_type=outcome.document_type.value,
            causes=causes,
        )
        return outcome

    def _review_with_trace(
        self,
        submission: DocumentSubmission | Mapping[str, Any],
        *,
        trace_session: ReviewTraceSession,
        review_span: ReviewTraceSpan,
    ) -> ReviewOutcome:
        """Execute existing review behavior while recording safe stage spans.

        Entry point:
            Called by ``review`` or the multi-agent composition boundary after
            a private trace parent exists. Agent functions and graph nodes
            remain non-public entry points.

        Sequence:
            1. Create random correlation and idempotency identifiers.
            2. Recheck that cloud tracing is disabled.
            3. Create initial PII-safe graph control state.
            4. Audit receipt and validate the synthetic submission.
            5. Store/read the document locally and verify byte/hash integrity.
            6. Invoke the private Agent 1 graph with sensitive text separate
               from callback-visible state.
            7. Return a released outcome only for a guarded extraction;
               otherwise return a sanitized human-review outcome.

        Failure behavior:
            Invalid input never reaches storage or extraction. Storage,
            configuration, validation, evidence, telemetry, audit, and
            unexpected failures all close release and preserve a human-review
            path without returning document contents in the failure. This
            slice never calls the configured local-model adapter.
        """

        correlation_id = new_correlation_id()
        idempotency_key = new_idempotency_key()
        boundary_state: BoundaryState = {
            "correlation_id": correlation_id,
            "idempotency_key": idempotency_key,
            "document_type": DocumentType.UNKNOWN,
        }
        preflight_span = review_span.start_child(TraceStage.RUNTIME_PREFLIGHT)
        try:
            ensure_cloud_tracing_disabled()
            if self._model.settings != self._settings:
                raise UnsafeRuntimeConfigurationError(
                    "The local model configuration changed after reviewer construction."
                )
        except UnsafeRuntimeConfigurationError:
            causes = (
                _trace_cause(
                    TraceStage.RUNTIME_PREFLIGHT,
                    FailureCode.UNSAFE_CONFIGURATION,
                ),
            )
            if not self._record_boundary(
                boundary_state,
                component="workflow",
                action="workflow_failed",
                status="failed",
                failure_code=FailureCode.UNSAFE_CONFIGURATION,
            ):
                causes = _trace_causes_with(
                    causes,
                    TraceStage.RUNTIME_PREFLIGHT,
                    FailureCode.AUDIT_WRITE_FAILED,
                )
                preflight_span.finish(TraceSpanOutcome.FAILED, causes=causes)
                return _human_review_outcome(
                    correlation_id,
                    idempotency_key,
                    FailureCode.AUDIT_WRITE_FAILED,
                )
            preflight_span.finish(TraceSpanOutcome.FAILED, causes=causes)
            return _human_review_outcome(
                correlation_id,
                idempotency_key,
                FailureCode.UNSAFE_CONFIGURATION,
            )
        preflight_span.finish(TraceSpanOutcome.SUCCEEDED)
        # Callback-visible orchestration flags only.  Document text, raw model
        # output, and extraction objects live in _RunArtifacts instead.
        graph_state: WorkflowState = {
            "document_type": DocumentType.UNKNOWN,
            "deterministic_candidate_ready": False,
            "unresolved_fields": (),
            "validated_extraction_ready": False,
            "schema_valid": False,
            "evidence_valid": False,
            "release_allowed": False,
            "audit_healthy": True,
            "verification_findings_ready": False,
            "critic_decision_ready": False,
        }
        input_span = review_span.start_child(TraceStage.INPUT_VALIDATION)
        if not self._record_boundary(
            boundary_state,
            component="input_validator",
            action="input_received",
            status="started",
        ):
            cause = _trace_cause(
                TraceStage.INPUT_VALIDATION,
                FailureCode.AUDIT_WRITE_FAILED,
            )
            input_span.finish(TraceSpanOutcome.FAILED, causes=(cause,))
            return _human_review_outcome(
                correlation_id,
                idempotency_key,
                FailureCode.AUDIT_WRITE_FAILED,
            )
        try:
            validated = SecureIntake.validate(self._intake, submission)
        except IntakeError:
            invalid_cause = _trace_cause(
                TraceStage.INPUT_VALIDATION,
                FailureCode.INVALID_INPUT,
            )
            if not self._record_boundary(
                boundary_state,
                component="input_validator",
                action="input_rejected",
                status="rejected",
                failure_code=FailureCode.INVALID_INPUT,
            ):
                causes = _trace_causes_with(
                    (invalid_cause,),
                    TraceStage.INPUT_VALIDATION,
                    FailureCode.AUDIT_WRITE_FAILED,
                )
                input_span.finish(TraceSpanOutcome.FAILED, causes=causes)
                return _human_review_outcome(
                    correlation_id,
                    idempotency_key,
                    FailureCode.AUDIT_WRITE_FAILED,
                )
            input_span.finish(
                TraceSpanOutcome.REJECTED,
                causes=(invalid_cause,),
            )
            telemetry_span = review_span.start_child(TraceStage.TELEMETRY_POLICY)
            if not self._emit_boundary_telemetry(
                boundary_state,
                TelemetryEventType.INPUT_REJECTED,
                "input_validation",
                "rejected",
                FailureCode.INVALID_INPUT,
            ):
                telemetry_cause = _trace_cause(
                    TraceStage.TELEMETRY_POLICY,
                    FailureCode.UNSAFE_CONFIGURATION,
                )
                telemetry_span.finish(
                    TraceSpanOutcome.FAILED,
                    causes=(telemetry_cause,),
                )
                return _human_review_outcome(
                    correlation_id,
                    idempotency_key,
                    FailureCode.UNSAFE_CONFIGURATION,
                )
            telemetry_span.finish(TraceSpanOutcome.SUCCEEDED)
            return _human_review_outcome(
                correlation_id,
                idempotency_key,
                FailureCode.INVALID_INPUT,
            )

        if not self._record_boundary(
            boundary_state,
            component="input_validator",
            action="input_validated",
            status="succeeded",
            document_byte_count=validated.byte_size,
        ):
            cause = _trace_cause(
                TraceStage.INPUT_VALIDATION,
                FailureCode.AUDIT_WRITE_FAILED,
            )
            input_span.finish(TraceSpanOutcome.FAILED, causes=(cause,))
            return _human_review_outcome(
                correlation_id,
                idempotency_key,
                FailureCode.AUDIT_WRITE_FAILED,
            )
        input_span.finish(TraceSpanOutcome.ACCEPTED)

        storage_span = review_span.start_child(TraceStage.DOCUMENT_STORAGE)
        try:
            stored = LocalDocumentStore.store(self._store, correlation_id, validated)
            document_text = LocalDocumentStore.read_text(
                self._store,
                correlation_id,
                stored.document_id,
            )
            _verify_stored_document(stored, document_text)
        except (StorageError, OSError, UnicodeError, ValueError):
            causes = (
                _trace_cause(
                    TraceStage.DOCUMENT_STORAGE,
                    FailureCode.LOCAL_STORAGE_ERROR,
                ),
            )
            if not self._record_boundary(
                boundary_state,
                component="document_store",
                action="workflow_failed",
                status="failed",
                failure_code=FailureCode.LOCAL_STORAGE_ERROR,
            ):
                causes = _trace_causes_with(
                    causes,
                    TraceStage.DOCUMENT_STORAGE,
                    FailureCode.AUDIT_WRITE_FAILED,
                )
                storage_span.finish(TraceSpanOutcome.FAILED, causes=causes)
                return _human_review_outcome(
                    correlation_id,
                    idempotency_key,
                    FailureCode.AUDIT_WRITE_FAILED,
                )
            storage_span.finish(TraceSpanOutcome.FAILED, causes=causes)
            return _human_review_outcome(
                correlation_id,
                idempotency_key,
                FailureCode.LOCAL_STORAGE_ERROR,
            )

        stored_boundary: BoundaryState = {
            **boundary_state,
            "document_id": stored.document_id,
        }
        if not self._record_boundary(
            stored_boundary,
            component="document_store",
            action="document_stored",
            status="succeeded",
            document_byte_count=stored.byte_size,
        ):
            cause = _trace_cause(
                TraceStage.DOCUMENT_STORAGE,
                FailureCode.AUDIT_WRITE_FAILED,
            )
            storage_span.finish(TraceSpanOutcome.FAILED, causes=(cause,))
            return _human_review_outcome(
                correlation_id,
                idempotency_key,
                FailureCode.AUDIT_WRITE_FAILED,
            )
        storage_span.finish(
            TraceSpanOutcome.SUCCEEDED,
            document_type=DocumentType.UNKNOWN.value,
        )

        telemetry_span = review_span.start_child(TraceStage.TELEMETRY_POLICY)
        if not self._emit_boundary_telemetry(
            stored_boundary,
            TelemetryEventType.INPUT_VALIDATED,
            "input_validation",
            "accepted",
        ):
            failure = FailureCode.UNSAFE_CONFIGURATION
            if not self._record_boundary(
                stored_boundary,
                component="telemetry",
                action="release_blocked",
                status="blocked",
                failure_code=failure,
            ):
                failure = FailureCode.AUDIT_WRITE_FAILED
            causes = (
                _trace_cause(
                    TraceStage.TELEMETRY_POLICY,
                    FailureCode.UNSAFE_CONFIGURATION,
                ),
            )
            if failure is FailureCode.AUDIT_WRITE_FAILED:
                causes = _trace_causes_with(
                    causes,
                    TraceStage.TELEMETRY_POLICY,
                    failure,
                )
            telemetry_span.finish(TraceSpanOutcome.FAILED, causes=causes)
            return _human_review_outcome(
                correlation_id,
                idempotency_key,
                failure,
            )
        telemetry_span.finish(TraceSpanOutcome.SUCCEEDED)

        agent_1_span = review_span.start_child(TraceStage.AGENT_1_EXTRACTION)
        try:
            final_state, validated_extraction = self._workflow.invoke(
                graph_state,
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                document_id=stored.document_id,
                document_text=document_text,
                trace_session=trace_session,
                agent_1_trace_span=agent_1_span,
            )
        except UnsafeRuntimeConfigurationError:
            failure = FailureCode.UNSAFE_CONFIGURATION
            if not self._record_boundary(
                stored_boundary,
                component="workflow",
                action="workflow_failed",
                status="failed",
                failure_code=failure,
            ):
                failure = FailureCode.AUDIT_WRITE_FAILED
            causes = _trace_causes_with(
                trace_session.causes,
                TraceStage.AGENT_1_EXTRACTION,
                failure,
                propagated=True,
            )
            agent_1_span.finish(TraceSpanOutcome.FAILED, causes=causes)
            return _human_review_outcome(
                correlation_id,
                idempotency_key,
                failure,
            )
        except AuditWriteError:
            self._record_boundary(
                stored_boundary,
                component="workflow",
                action="workflow_failed",
                status="failed",
                failure_code=FailureCode.AUDIT_WRITE_FAILED,
            )
            causes = _trace_causes_with(
                trace_session.causes,
                TraceStage.AGENT_1_EXTRACTION,
                FailureCode.AUDIT_WRITE_FAILED,
                propagated=True,
            )
            agent_1_span.finish(TraceSpanOutcome.FAILED, causes=causes)
            return _human_review_outcome(
                correlation_id,
                idempotency_key,
                FailureCode.AUDIT_WRITE_FAILED,
            )
        except Exception:
            if not self._record_boundary(
                stored_boundary,
                component="workflow",
                action="workflow_failed",
                status="failed",
                failure_code=FailureCode.INTERNAL_FAILURE,
            ):
                causes = _trace_causes_with(
                    trace_session.causes,
                    TraceStage.AGENT_1_EXTRACTION,
                    FailureCode.AUDIT_WRITE_FAILED,
                    propagated=True,
                )
                agent_1_span.finish(TraceSpanOutcome.FAILED, causes=causes)
                return _human_review_outcome(
                    correlation_id,
                    idempotency_key,
                    FailureCode.AUDIT_WRITE_FAILED,
                )
            causes = _trace_causes_with(
                trace_session.causes,
                TraceStage.AGENT_1_EXTRACTION,
                FailureCode.INTERNAL_FAILURE,
                propagated=True,
            )
            agent_1_span.finish(TraceSpanOutcome.FAILED, causes=causes)
            return _human_review_outcome(
                correlation_id,
                idempotency_key,
                FailureCode.INTERNAL_FAILURE,
            )

        if final_state.get("release_allowed") and validated_extraction is not None:
            agent_1_span.finish(
                TraceSpanOutcome.SUCCEEDED,
                document_type=final_state.get(
                    "document_type",
                    DocumentType.UNKNOWN,
                ).value,
            )
            return ReviewOutcome(
                correlation_id=correlation_id,
                idempotency_key=idempotency_key,
                status=WorkflowStatus.RELEASED,
                document_type=final_state.get("document_type", DocumentType.UNKNOWN),
                validated_extraction=validated_extraction,
                failure_code=None,
                human_review_required=False,
            )
        failure = final_state.get("failure_code") or FailureCode.INTERNAL_FAILURE
        causes = _trace_causes_with(
            trace_session.causes,
            TraceStage.AGENT_1_EXTRACTION,
            failure,
            propagated=True,
        )
        agent_1_span.finish(
            TraceSpanOutcome.HUMAN_REVIEW,
            document_type=final_state.get(
                "document_type",
                DocumentType.UNKNOWN,
            ).value,
            causes=causes,
        )
        return _human_review_outcome(
            correlation_id,
            idempotency_key,
            failure,
            document_type=final_state.get("document_type", DocumentType.UNKNOWN),
        )

    def _record_boundary(
        self,
        state: BoundaryState,
        *,
        component: str,
        action: str,
        status: str,
        failure_code: FailureCode | None = None,
        **counts: int,
    ) -> bool:
        """Write and verify a PII-safe event outside graph execution.

        This is used before a ``run_token`` and ``_RunArtifacts`` entry exist,
        or after graph invocation returns.  It mirrors ``_Agent1Graph._record``
        while reading identifiers from ``BoundaryState``.
        """

        try:
            metadata = SafeEventMetadata(
                correlation_id=state["correlation_id"],
                idempotency_key=state.get("idempotency_key"),
                opaque_document_id=state.get("document_id"),
                component=component,
                action=action,
                status=status,
                error_code=(
                    _OBSERVABILITY_ERROR_CODES[failure_code]
                    if failure_code is not None
                    else None
                ),
                document_type=state.get("document_type", DocumentType.UNKNOWN).value,
                extraction_schema_version=SCHEMA_VERSION,
                workflow_version=WORKFLOW_VERSION,
                **counts,
            )
            event = PIISafeStructuredLogger.emit(self._logger, metadata)
            if event.metadata != metadata:
                return False
            if not _append_verified_local_audit(self._audit, event):
                return False
            return True
        except Exception:
            return False

    def _emit_boundary_telemetry(
        self,
        state: BoundaryState,
        event_type: TelemetryEventType,
        component: str,
        outcome: str,
        failure_code: FailureCode | None = None,
    ) -> bool:
        """Emit sanitized pre/post-graph telemetry through the disabled sink."""

        try:
            self._telemetry.emit(
                SanitizedTelemetryEvent(
                    event_type=event_type,
                    component=component,
                    outcome=outcome,
                    reason_code=failure_code.value if failure_code else None,
                    document_type=state.get("document_type", DocumentType.UNKNOWN).value,
                    schema_version=SCHEMA_VERSION,
                )
            )
            return True
        except Exception:
            return False


def _append_verified_local_audit(
    audit_store: LocalJsonlAuditStore,
    event: SafeLogEvent,
) -> bool:
    """Append through the sealed local class and verify the exact persisted record."""

    record = LocalJsonlAuditStore.append_event(audit_store, event)
    if not isinstance(record, AuditRecord) or record.event_id != event.event_id:
        return False
    persisted = LocalJsonlAuditStore.list_records(
        audit_store,
        after_sequence=record.sequence - 1,
        limit=1,
    )
    return persisted == (record,)


def _failure_from_extraction_error(error: ExtractionValidationError) -> FailureCode:
    """Translate Agent 1's sanitized error code into a workflow failure enum."""

    return {
        "invalid_model_output": FailureCode.INVALID_MODEL_OUTPUT,
        "schema_validation_failed": FailureCode.SCHEMA_VALIDATION_FAILED,
        "evidence_validation_failed": FailureCode.EVIDENCE_VALIDATION_FAILED,
        "unsupported_required_field": FailureCode.UNSUPPORTED_REQUIRED_FIELD,
        "unsupported_document_type": FailureCode.UNSUPPORTED_DOCUMENT_TYPE,
    }.get(error.code, FailureCode.SCHEMA_VALIDATION_FAILED)


def _field_counts(extraction: ValidatedExtraction) -> tuple[int, int, int, int]:
    """Return safe aggregate counts for audit events without field values."""

    fields = list(iter_extracted_fields(extraction))
    unsupported = sum(isinstance(field, UnsupportedField) for _, field in fields)
    supported = len(fields) - unsupported
    evidence = sum(len(field.provenance) for _, field in fields)
    return len(fields), supported, unsupported, evidence


def _verify_stored_document(stored: StoredDocument, document_text: str) -> None:
    """Verify that locally read text matches stored byte count and content hash."""

    payload = document_text.encode("utf-8")
    if (
        len(payload) != stored.byte_size
        or hashlib.sha256(payload).hexdigest() != stored.content_sha256
    ):
        raise StorageError()


def _human_review_outcome(
    correlation_id: str,
    idempotency_key: str,
    failure_code: FailureCode,
    *,
    document_type: DocumentType = DocumentType.UNKNOWN,
) -> ReviewOutcome:
    """Construct the only terminal outcome allowed for an unreleased run."""

    return ReviewOutcome(
        correlation_id=correlation_id,
        idempotency_key=idempotency_key,
        status=WorkflowStatus.HUMAN_REVIEW,
        document_type=document_type,
        validated_extraction=None,
        failure_code=failure_code,
        human_review_required=True,
    )
