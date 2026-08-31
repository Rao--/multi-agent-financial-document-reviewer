"""Define a sanitized telemetry extension while exporting nothing by default.

Why this file exists:
    Keep any future operational telemetry behind an explicit schema and visible
    configuration boundary instead of allowing framework callbacks to export
    arbitrary graph data.

What it owns:
    Closed event and completed-span schemas, the flat ``TelemetrySink``
    compatibility boundary, a vendor-neutral parent/child ``ReviewTraceSink``,
    private trace-session/span handles, no-op implementations, and a
    ``DisabledLangSmithTelemetrySink`` that refuses enablement.

What it excludes:
    There is no arbitrary ``metadata`` or ``payload`` field where document text,
    prompts, outputs, or business identifiers could be hidden. Trace handles
    never enter LangGraph state or model context. This increment constructs no
    OTLP or LangSmith client and exports nothing. Local durable audit records
    remain the responsibility of ``local.observability``.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from enum import Enum
from time import monotonic_ns
from typing import Literal, Protocol, Self, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from financial_reviewer.foundation.config import ensure_cloud_tracing_disabled


class TelemetryEventType(str, Enum):
    """Closed event-name allowlist for local audit/telemetry adapters."""

    WORKFLOW_STARTED = "workflow_started"
    INPUT_VALIDATED = "input_validated"
    INPUT_REJECTED = "input_rejected"
    EXTRACTION_STARTED = "extraction_started"
    EXTRACTION_COMPLETED = "extraction_completed"
    EXTRACTION_FAILED = "extraction_failed"
    SCHEMA_VALIDATION_PASSED = "schema_validation_passed"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    EVIDENCE_GUARD_PASSED = "evidence_guard_passed"
    EVIDENCE_GUARD_FAILED = "evidence_guard_failed"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    WORKFLOW_COMPLETED = "workflow_completed"
    WORKFLOW_FAILED = "workflow_failed"


# Closed vocabularies intentionally omit prompts, text, values, identifiers, and
# arbitrary metadata, so even a future sink cannot accept them through this API.
TelemetryComponent = Literal[
    "workflow",
    "input_validation",
    "storage",
    "agent_1_extraction",
    "schema_validation",
    "evidence_guard",
    "human_review",
]
TelemetryOutcome = Literal["started", "accepted", "rejected", "succeeded", "failed"]
TelemetryReasonCode = Literal[
    "invalid_input",
    "unsupported_document_type",
    "local_model_error",
    "invalid_model_output",
    "schema_validation_failed",
    "evidence_validation_failed",
    "unsupported_required_field",
    "local_storage_error",
    "audit_write_failed",
    "unsafe_configuration",
    "internal_failure",
    "invalid_bundle",
    "extraction_failed",
    "handoff_failed",
    "verification_failed",
    "critic_handoff_failed",
    "critic_failed",
    "final_gate_failed",
    "invalid_extraction_count",
    "duplicate_document",
    "unguarded_extraction",
    "unsupported_document_combination",
    "required_income_evidence_invalid",
    "model_decision_failed",
    "invalid_model_decision",
    "tool_call_limit_reached",
    "upstream_verification_failed",
    "invalid_source_set",
    "invalid_provenance",
    "invalid_comparison_linkage",
    "invalid_critic_request",
    "repair_exhausted",
    "income_inconsistent",
    "income_not_comparable",
    "insufficient_evidence",
]
SafeDocumentType = Literal[
    "pay_stub",
    "paystub",
    "bank_statement",
    "tax_form",
    "identity_document",
    "unknown",
    "unsupported",
]


class TraceStage(str, Enum):
    """Closed span-name vocabulary for the implemented review path."""

    FINANCIAL_REVIEW = "financial.review"
    INCOME_REVIEW_INPUT_VALIDATION = "financial.income_review.input_validation"
    EXTRACT_DOCUMENTS = "financial.income_review.extract_documents"
    VERIFICATION_INPUT_ASSEMBLE = (
        "financial.income_review.verification_input_assemble"
    )
    AGENT_2_VERIFICATION = "financial.income_review.agent.income_verification"
    CRITIC_INPUT_ASSEMBLE = "financial.income_review.critic_input_assemble"
    AGENT_3_CRITIC = "financial.income_review.agent.critic"
    FINAL_GATE = "financial.income_review.final_gate"
    RUNTIME_PREFLIGHT = "runtime.preflight"
    INPUT_VALIDATION = "input.validation"
    DOCUMENT_STORAGE = "document.storage"
    TELEMETRY_POLICY = "telemetry.policy"
    AGENT_1_EXTRACTION = "agent.evidence_extraction"
    CLASSIFICATION = "agent.evidence_extraction.classification"
    DETERMINISTIC_EXTRACTION = "agent.evidence_extraction.extract"
    SCHEMA_VALIDATION = "agent.evidence_extraction.schema_validation"
    EVIDENCE_GUARD = "agent.evidence_extraction.evidence_guard"
    AGENT_1_FINALIZE = "agent.evidence_extraction.finalize"
    FINAL_REVIEW_DECISION = "review.final_decision"


class TraceSpanOutcome(str, Enum):
    """Operational result of one span, separate from financial findings."""

    SUCCEEDED = "succeeded"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"
    HUMAN_REVIEW = "human_review"
    SKIPPED = "skipped"


class TraceCauseKind(str, Enum):
    """Why a stage did not produce its normal successful result."""

    TECHNICAL_FAILURE = "technical_failure"
    VALIDATION_FAILURE = "validation_failure"
    POLICY_BLOCK = "policy_block"
    BUSINESS_FINDING = "business_finding"


class SanitizedTraceCause(BaseModel):
    """One allowlisted, non-sensitive origin contributing to a result."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )

    origin_stage: TraceStage
    kind: TraceCauseKind
    reason_code: TelemetryReasonCode


class SanitizedTraceSpan(BaseModel):
    """Completed parent/child span containing no payload or business identifier.

    Trace and span identifiers are random OpenTelemetry-shaped identifiers used
    only to preserve hierarchy. They are not derived from a document,
    correlation ID, employee, account, or any other business value.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        revalidate_instances="always",
    )

    trace_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    span_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    parent_span_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{16}$",
    )
    stage: TraceStage
    outcome: TraceSpanOutcome
    started_at: datetime
    ended_at: datetime
    duration_ms: int = Field(ge=0, le=86_400_000)
    document_type: SafeDocumentType | None = None
    causes: tuple[SanitizedTraceCause, ...] = ()

    @field_validator("started_at", "ended_at")
    @classmethod
    def validate_trace_timestamp(cls, value: datetime) -> datetime:
        """Require timezone-aware UTC timestamps for stable local projection."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("trace timestamps must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def validate_span_relationships(self) -> Self:
        """Reject impossible timing, self-parenting, and duplicate causes."""

        if self.ended_at < self.started_at:
            raise ValueError("trace span cannot end before it starts")
        if self.parent_span_id == self.span_id:
            raise ValueError("trace span cannot be its own parent")
        cause_keys = {
            (cause.origin_stage, cause.kind, cause.reason_code)
            for cause in self.causes
        }
        if len(cause_keys) != len(self.causes):
            raise ValueError("trace span causes must be unique")
        if len(self.causes) > 16:
            raise ValueError("trace span cause count exceeds the safe limit")
        return self


@runtime_checkable
class ReviewTraceSink(Protocol):
    """Vendor-neutral destination for already-sanitized completed spans."""

    def emit_span(self, span: SanitizedTraceSpan) -> None:
        """Accept one closed span without prompts, values, or arbitrary fields."""


class NoOpReviewTraceSink:
    """Disabled-by-default trace adapter that validates and exports nothing."""

    __slots__ = ()

    def emit_span(self, span: SanitizedTraceSpan) -> None:
        """Revalidate the safe contract while deliberately performing no I/O."""

        if not isinstance(span, SanitizedTraceSpan):
            raise TypeError("trace sinks accept only SanitizedTraceSpan")
        try:
            SanitizedTraceSpan.model_validate(
                span.model_dump(mode="python", warnings="none")
            )
        except (TypeError, ValueError):
            raise TypeError("the sanitized trace span is invalid") from None


class ReviewTraceExportError(RuntimeError):
    """A sanitized projection failure that must not change review decisions."""


class ReviewTraceSpan:
    """Private in-process handle that preserves one span's parent context."""

    __slots__ = (
        "_ended",
        "_parent_span_id",
        "_session",
        "_span_id",
        "_stage",
        "_started_at",
        "_started_ns",
    )

    def __init__(
        self,
        *,
        session: "ReviewTraceSession",
        stage: TraceStage,
        parent_span_id: str | None,
    ) -> None:
        self._session = session
        self._stage = stage
        self._parent_span_id = parent_span_id
        self._span_id = uuid4().hex[:16]
        self._started_at = datetime.now(timezone.utc)
        self._started_ns = monotonic_ns()
        self._ended = False

    @property
    def span_id(self) -> str:
        """Return the random hierarchy identifier, never a business identifier."""

        return self._span_id

    @property
    def stage(self) -> TraceStage:
        """Return this handle's closed stage name."""

        return self._stage

    def start_child(self, stage: TraceStage) -> "ReviewTraceSpan":
        """Create one active child beneath this still-active parent span."""

        if self._ended:
            raise RuntimeError("cannot start a child from a completed trace span")
        return self._session.start_child(self, stage)

    def finish(
        self,
        outcome: TraceSpanOutcome,
        *,
        document_type: SafeDocumentType | None = None,
        causes: tuple[SanitizedTraceCause, ...] = (),
    ) -> SanitizedTraceSpan:
        """Complete and emit this span exactly once through the safe sink."""

        if self._ended:
            raise RuntimeError("trace span is already complete")
        try:
            span = self._session._finish_span(
                self,
                outcome,
                document_type=document_type,
                causes=causes,
            )
        except Exception:
            # A destination failure happens after local lifecycle cleanup,
            # while a premature parent finish leaves the handle active so the
            # caller can complete its children and retry.
            with self._session._lock:
                self._ended = self._span_id not in self._session._active_span_ids
            raise
        self._ended = True
        return span


class ReviewTraceSession:
    """Own one review trace while keeping context outside LangGraph state."""

    __slots__ = (
        "_active_parent_by_span",
        "_active_span_ids",
        "_causes",
        "_export_failure_count",
        "_lock",
        "_root_started",
        "_sink",
        "trace_id",
    )

    def __init__(self, sink: ReviewTraceSink) -> None:
        if not isinstance(sink, ReviewTraceSink):
            raise TypeError("trace sink must implement ReviewTraceSink")
        self._sink = sink
        self.trace_id = uuid4().hex
        self._lock = threading.RLock()
        self._active_span_ids: set[str] = set()
        self._active_parent_by_span: dict[str, str | None] = {}
        self._causes: list[SanitizedTraceCause] = []
        self._export_failure_count = 0
        self._root_started = False

    @property
    def causes(self) -> tuple[SanitizedTraceCause, ...]:
        """Return unique causes observed so far in first-occurrence order."""

        with self._lock:
            return tuple(self._causes)

    @property
    def export_failure_count(self) -> int:
        """Return sanitized projection failures without changing review state."""

        with self._lock:
            return self._export_failure_count

    def start_root(self) -> ReviewTraceSpan:
        """Start the single root span before any review validation occurs."""

        with self._lock:
            if self._root_started:
                raise RuntimeError("review trace already has a root span")
            self._root_started = True
        return self._start_span(TraceStage.FINANCIAL_REVIEW, parent_span_id=None)

    def start_child(
        self,
        parent: ReviewTraceSpan,
        stage: TraceStage,
    ) -> ReviewTraceSpan:
        """Start a child only when the supplied parent belongs to this trace."""

        if not isinstance(parent, ReviewTraceSpan) or parent._session is not self:
            raise TypeError("trace parent does not belong to this review")
        with self._lock:
            if parent.span_id not in self._active_span_ids:
                raise RuntimeError("trace parent is not active")
        return self._start_span(stage, parent_span_id=parent.span_id)

    def _start_span(
        self,
        stage: TraceStage,
        *,
        parent_span_id: str | None,
    ) -> ReviewTraceSpan:
        if not isinstance(stage, TraceStage):
            raise TypeError("trace stage must be allowlisted")
        span = ReviewTraceSpan(
            session=self,
            stage=stage,
            parent_span_id=parent_span_id,
        )
        with self._lock:
            if span.span_id in self._active_span_ids:
                raise RuntimeError("duplicate active trace span")
            self._active_span_ids.add(span.span_id)
            self._active_parent_by_span[span.span_id] = parent_span_id
        return span

    def _finish_span(
        self,
        handle: ReviewTraceSpan,
        outcome: TraceSpanOutcome,
        *,
        document_type: SafeDocumentType | None,
        causes: tuple[SanitizedTraceCause, ...],
    ) -> SanitizedTraceSpan:
        if not isinstance(outcome, TraceSpanOutcome):
            raise TypeError("trace outcome must be allowlisted")
        with self._lock:
            if handle._session is not self or handle.span_id not in self._active_span_ids:
                raise RuntimeError("trace span is not active")
            if handle.span_id in self._active_parent_by_span.values():
                raise RuntimeError("trace span cannot finish before its active child")
        ended_at = datetime.now(timezone.utc)
        duration_ms = max(0, (monotonic_ns() - handle._started_ns) // 1_000_000)
        span = SanitizedTraceSpan(
            trace_id=self.trace_id,
            span_id=handle.span_id,
            parent_span_id=handle._parent_span_id,
            stage=handle.stage,
            outcome=outcome,
            started_at=handle._started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            document_type=document_type,
            causes=causes,
        )
        with self._lock:
            self._active_span_ids.remove(handle.span_id)
            self._active_parent_by_span.pop(handle.span_id, None)
            existing = {
                (cause.origin_stage, cause.kind, cause.reason_code)
                for cause in self._causes
            }
            for cause in causes:
                key = (cause.origin_stage, cause.kind, cause.reason_code)
                if key not in existing:
                    self._causes.append(cause)
                    existing.add(key)
        try:
            self._sink.emit_span(span)
        except ReviewTraceExportError:
            # The mandatory local audit remains authoritative. Optional
            # Traceboard projection cannot change a financial-review outcome.
            with self._lock:
                self._export_failure_count += 1
        return span


class SanitizedTelemetryEvent(BaseModel):
    """An unlinkable event with no content, arbitrary metadata, or local IDs."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        use_enum_values=True,
        revalidate_instances="always",
    )

    event_type: TelemetryEventType
    component: TelemetryComponent
    outcome: TelemetryOutcome
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Optional values are limited to non-PII operational classifications and
    # aggregate counters.  Raw validation text belongs only in local control flow.
    reason_code: TelemetryReasonCode | None = None
    document_type: SafeDocumentType | None = None
    # Accept the prior value for historical local events while current workflow
    # records use the required-field-aware extraction schema version 1.1.
    schema_version: Literal["1.0", "1.1"] | None = None
    extracted_field_count: int | None = Field(default=None, ge=0, le=10_000)
    unsupported_field_count: int | None = Field(default=None, ge=0, le=10_000)
    duration_ms: int | None = Field(default=None, ge=0, le=86_400_000)

    @field_validator("occurred_at")
    @classmethod
    def validate_aware_timestamp(cls, value: datetime) -> datetime:
        """Require an aware timestamp and normalize it to UTC."""

        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value.astimezone(timezone.utc)

@runtime_checkable
class TelemetrySink(Protocol):
    """Minimal sink boundary accepted by workflow components."""

    def emit(self, event: SanitizedTelemetryEvent) -> None:
        """Accept one already-sanitized event."""


class NoOpTelemetrySink:
    """Default local behavior: validate at construction sites, export nothing."""

    __slots__ = ()

    def emit(self, event: SanitizedTelemetryEvent) -> None:
        """Validate a sanitized event while deliberately exporting nothing."""

        ensure_cloud_tracing_disabled()
        if not isinstance(event, SanitizedTelemetryEvent):
            raise TypeError("Telemetry sinks accept only SanitizedTelemetryEvent")
        try:
            SanitizedTelemetryEvent.model_validate(
                event.model_dump(mode="python", warnings="none")
            )
        except (TypeError, ValueError):
            raise TypeError("The sanitized telemetry event is invalid") from None


class LangSmithTelemetrySettings(BaseModel):
    """Milestone 1 switch that cannot be enabled by configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: Literal[False] = False

    @model_validator(mode="after")
    def enforce_disabled(self) -> Self:
        """Fail construction if process settings attempt to enable cloud tracing."""

        ensure_cloud_tracing_disabled()
        return self


class DisabledLangSmithTelemetrySink(NoOpTelemetrySink):
    """Documented extension point for a future *sanitized-only* integration.

    It creates no client and makes no network calls. A future milestone must
    replace this class deliberately after a privacy review; prompts, model
    outputs, document/account/employee data, and reversible IDs must remain
    excluded even then. The event contract omits correlation and document IDs.
    """

    __slots__ = ("_settings",)

    def __init__(self, settings: LangSmithTelemetrySettings | None = None) -> None:
        """Bind settings whose schema permits only the disabled state."""

        candidate = settings or LangSmithTelemetrySettings()
        try:
            self._settings = LangSmithTelemetrySettings.model_validate(
                candidate.model_dump(mode="python", warnings="none")
            )
        except (AttributeError, TypeError, ValueError):
            raise TypeError("LangSmith telemetry must remain disabled") from None

    @property
    def enabled(self) -> Literal[False]:
        """Expose the enforced disabled state for wiring and health checks."""

        return self._settings.enabled

    def emit(self, event: SanitizedTelemetryEvent) -> None:
        """Recheck the cloud guard, validate the event, and perform no export."""

        # Re-check on every use in case a process-level tracing flag changed
        # after this no-op sink was constructed.
        ensure_cloud_tracing_disabled()
        super().emit(event)


# Role-oriented aliases kept intentionally small for workflow wiring.
TelemetryEvent = SanitizedTelemetryEvent
DisabledLangSmithSink = DisabledLangSmithTelemetrySink


__all__ = [
    "DisabledLangSmithSink",
    "DisabledLangSmithTelemetrySink",
    "LangSmithTelemetrySettings",
    "NoOpTelemetrySink",
    "NoOpReviewTraceSink",
    "ReviewTraceExportError",
    "ReviewTraceSession",
    "ReviewTraceSink",
    "ReviewTraceSpan",
    "SafeDocumentType",
    "SanitizedTraceCause",
    "SanitizedTraceSpan",
    "SanitizedTelemetryEvent",
    "TelemetryComponent",
    "TelemetryEvent",
    "TelemetryEventType",
    "TelemetryOutcome",
    "TelemetryReasonCode",
    "TelemetrySink",
    "TraceCauseKind",
    "TraceSpanOutcome",
    "TraceStage",
]
