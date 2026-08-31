"""Provide local observability without representing document or PII content.

Why this file exists:
    Production troubleshooting and auditability need correlation, event order,
    safe failure reasons, and integrity checks, but normal logging APIs can
    accidentally retain financial content.

What it owns:
    Opaque identifier creation/validation, closed event schemas, exception-code
    sanitization, the PII-safe structured logger, append-only JSONL audit store,
    hash chaining, filtered reports, and full-file integrity verification.

What it excludes:
    There are no free-form log or audit fields.  Raw documents, extracted
    values, prompts, outputs, filenames, exception messages, and business
    identifiers have no representation here.  The store is local-only and
    fails closed if its sequence, schema, or hash chain has been altered.
"""

from __future__ import annotations

import logging
import os
import re
import stat
import threading
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Final, Literal, Protocol, cast, runtime_checkable
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    ValidationError,
    field_validator,
    model_validator,
)

try:  # pragma: no cover - available on the supported POSIX deployment target.
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]


# UUIDv4 structure plus role-specific prefixes makes identifiers locally
# correlatable while preventing account, employee, or document names in them.
_UUID4_HEX: Final[str] = r"[0-9a-f]{12}4[0-9a-f]{3}[89ab][0-9a-f]{15}"
_CORRELATION_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"\Acorr_{_UUID4_HEX}\Z"
)
_IDEMPOTENCY_KEY_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"\Aidem_{_UUID4_HEX}\Z"
)
_OPAQUE_DOCUMENT_ID_PATTERN: Final[re.Pattern[str]] = re.compile(
    rf"\Adoc_{_UUID4_HEX}\Z"
)
_EVENT_ID_PATTERN: Final[re.Pattern[str]] = re.compile(rf"\Aevt_{_UUID4_HEX}\Z")

# Reusable strict scalar types close the observability schema before anything
# reaches the logger or append-only audit store.
CorrelationId = Annotated[
    str,
    Field(strict=True, min_length=37, max_length=37, pattern=_CORRELATION_ID_PATTERN),
]
IdempotencyKey = Annotated[
    str,
    Field(strict=True, min_length=37, max_length=37, pattern=_IDEMPOTENCY_KEY_PATTERN),
]
OpaqueDocumentId = Annotated[
    str,
    Field(
        strict=True,
        min_length=36,
        max_length=36,
        pattern=_OPAQUE_DOCUMENT_ID_PATTERN,
    ),
]
EventId = Annotated[
    str,
    Field(strict=True, min_length=36, max_length=36, pattern=_EVENT_ID_PATTERN),
]
NonNegativeCount = Annotated[int, Field(strict=True, ge=0, le=2_147_483_647)]
PositiveSequence = Annotated[int, Field(strict=True, ge=1, le=9_223_372_036_854_775_807)]
SafeVersion = Annotated[
    str,
    Field(
        strict=True,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:@+-]{0,63}$",
    ),
]

Component = Literal[
    "system",
    "input_validator",
    "input_validation",
    "document_store",
    "storage",
    "workflow",
    "agent_1_extraction",
    "agent_2_validation",
    "agent_3_decision",
    "schema_validator",
    "schema_validation",
    "evidence_guard",
    "local_model",
    "human_review",
    "audit_store",
    "telemetry",
]
Action = Literal[
    "input_received",
    "input_validated",
    "input_rejected",
    "document_stored",
    "document_loaded",
    "workflow_started",
    "workflow_completed",
    "workflow_failed",
    "classification_started",
    "classification_completed",
    "extraction_started",
    "extraction_completed",
    "extraction_failed",
    "validation_started",
    "validation_completed",
    "schema_validation_passed",
    "schema_validation_failed",
    "evidence_check_started",
    "evidence_check_completed",
    "evidence_guard_passed",
    "evidence_guard_failed",
    "model_call_started",
    "model_call_completed",
    "model_call_failed",
    "release_blocked",
    "release_approved",
    "human_review_requested",
    "human_review_required",
    "audit_appended",
    "telemetry_disabled",
    "report_generated",
]
EventStatus = Literal[
    "started",
    "succeeded",
    "failed",
    "rejected",
    "blocked",
    "unsupported",
    "uncertain",
    "human_review",
    "disabled",
]
DocumentType = Literal[
    "pay_stub",
    "bank_statement",
    "w2",
    "tax_return",
    "tax_form",
    "loan_application",
    "identity_document",
    "income_statement",
    "other_financial_document",
    "unknown",
    "unsupported",
]
ErrorCode = Literal[
    "INVALID_INPUT",
    "EMPTY_INPUT",
    "INPUT_TOO_LARGE",
    "UNSUPPORTED_MEDIA_TYPE",
    "UNSUPPORTED_DOCUMENT",
    "UNSUPPORTED_DOCUMENT_TYPE",
    "BINARY_CONTENT_DETECTED",
    "PATH_POLICY_VIOLATION",
    "LOCAL_MODEL_UNAVAILABLE",
    "LOCAL_MODEL_TIMEOUT",
    "LOCAL_MODEL_ERROR",
    "MODEL_OUTPUT_INVALID",
    "INVALID_MODEL_OUTPUT",
    "SCHEMA_VALIDATION_FAILED",
    "EVIDENCE_MISSING",
    "EVIDENCE_MISMATCH",
    "EVIDENCE_UNSUPPORTED",
    "EVIDENCE_VALIDATION_FAILED",
    "UNSUPPORTED_REQUIRED_FIELD",
    "RELEASE_BLOCKED",
    "STORAGE_ERROR",
    "LOCAL_STORAGE_ERROR",
    "AUDIT_INTEGRITY_ERROR",
    "AUDIT_WRITE_ERROR",
    "AUDIT_WRITE_FAILED",
    "NETWORK_POLICY_VIOLATION",
    "TELEMETRY_POLICY_VIOLATION",
    "RETRY_EXHAUSTED",
    "UNSAFE_CONFIGURATION",
    "INTERNAL_FAILURE",
    "INTERNAL_ERROR",
]

# Runtime membership guard backing the public ErrorCode Literal. Unknown values
# collapse to INTERNAL_ERROR rather than leaking arbitrary exception text.
_ALLOWED_ERROR_CODES: Final[frozenset[str]] = frozenset(
    {
        "INVALID_INPUT",
        "EMPTY_INPUT",
        "INPUT_TOO_LARGE",
        "UNSUPPORTED_MEDIA_TYPE",
        "UNSUPPORTED_DOCUMENT",
        "UNSUPPORTED_DOCUMENT_TYPE",
        "BINARY_CONTENT_DETECTED",
        "PATH_POLICY_VIOLATION",
        "LOCAL_MODEL_UNAVAILABLE",
        "LOCAL_MODEL_TIMEOUT",
        "LOCAL_MODEL_ERROR",
        "MODEL_OUTPUT_INVALID",
        "INVALID_MODEL_OUTPUT",
        "SCHEMA_VALIDATION_FAILED",
        "EVIDENCE_MISSING",
        "EVIDENCE_MISMATCH",
        "EVIDENCE_UNSUPPORTED",
        "EVIDENCE_VALIDATION_FAILED",
        "UNSUPPORTED_REQUIRED_FIELD",
        "RELEASE_BLOCKED",
        "STORAGE_ERROR",
        "LOCAL_STORAGE_ERROR",
        "AUDIT_INTEGRITY_ERROR",
        "AUDIT_WRITE_ERROR",
        "AUDIT_WRITE_FAILED",
        "NETWORK_POLICY_VIOLATION",
        "TELEMETRY_POLICY_VIOLATION",
        "RETRY_EXHAUSTED",
        "UNSAFE_CONFIGURATION",
        "INTERNAL_FAILURE",
        "INTERNAL_ERROR",
    }
)
# These statuses must carry a sanitized failure code in SafeEventMetadata.
_FAILURE_STATUSES: Final[frozenset[str]] = frozenset(
    {"failed", "rejected", "blocked"}
)
# Only severity selection uses this set; it never changes release behavior.
_ERROR_LOG_STATUSES: Final[frozenset[str]] = frozenset(
    {"failed", "rejected", "blocked"}
)


class ObservabilityError(RuntimeError):
    """Base exception with a deliberately non-sensitive message."""


class UnsafeAuditPathError(ObservabilityError):
    """The requested audit location does not satisfy the local security policy."""


class AuditIntegrityError(ObservabilityError):
    """Existing audit data failed deterministic integrity validation."""


class AuditWriteError(ObservabilityError):
    """An append could not be durably completed."""


def _new_prefixed_uuid(prefix: str) -> str:
    """Generate a random opaque identifier with a caller-selected safe prefix."""

    return f"{prefix}{uuid4().hex}"


def new_correlation_id() -> str:
    """Return an opaque UUIDv4 correlation ID with no business identifiers."""

    return _new_prefixed_uuid("corr_")


def new_idempotency_key() -> str:
    """Return an opaque UUIDv4 idempotency key with no business identifiers."""

    return _new_prefixed_uuid("idem_")


def new_opaque_document_id() -> str:
    """Return an opaque local document handle suitable for safe metadata."""

    return _new_prefixed_uuid("doc_")


def new_event_id() -> str:
    """Return an opaque UUIDv4 event identifier."""

    return _new_prefixed_uuid("evt_")


def _validate_opaque_identifier(
    value: object,
    pattern: re.Pattern[str],
    label: str,
) -> str:
    """Validate an opaque identifier without echoing a rejected value."""

    if type(value) is not str or pattern.fullmatch(value) is None:
        raise ValueError(f"invalid {label}")
    return value


def validate_correlation_id(value: object) -> str:
    """Validate a Financial Document Reviewer correlation ID without echoing invalid input."""

    return _validate_opaque_identifier(value, _CORRELATION_ID_PATTERN, "correlation ID")


def validate_idempotency_key(value: object) -> str:
    """Validate a Financial Document Reviewer idempotency key without echoing invalid input."""

    return _validate_opaque_identifier(value, _IDEMPOTENCY_KEY_PATTERN, "idempotency key")


def validate_opaque_document_id(value: object) -> str:
    """Validate an opaque local document ID without echoing invalid input."""

    return _validate_opaque_identifier(
        value,
        _OPAQUE_DOCUMENT_ID_PATTERN,
        "opaque document ID",
    )


def validate_event_id(value: object) -> str:
    """Validate an audit/log event ID without echoing invalid input."""

    return _validate_opaque_identifier(value, _EVENT_ID_PATTERN, "event ID")


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


class _StrictModel(BaseModel):
    """Shared immutable and closed Pydantic policy for observability records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        validate_default=True,
        revalidate_instances="always",
    )


class SafeEventMetadata(_StrictModel):
    """Closed, PII-safe metadata allowed in logs and local audit records."""

    correlation_id: CorrelationId
    component: Component
    action: Action
    status: EventStatus
    idempotency_key: IdempotencyKey | None = None
    opaque_document_id: OpaqueDocumentId | None = None
    error_code: ErrorCode | None = None
    document_type: DocumentType | None = None
    attempt: NonNegativeCount | None = None
    retry_count: NonNegativeCount | None = None
    duration_ms: NonNegativeCount | None = None
    document_byte_count: NonNegativeCount | None = None
    page_count: NonNegativeCount | None = None
    field_count: NonNegativeCount | None = None
    supported_field_count: NonNegativeCount | None = None
    unsupported_field_count: NonNegativeCount | None = None
    evidence_count: NonNegativeCount | None = None
    validation_error_count: NonNegativeCount | None = None
    input_token_count: NonNegativeCount | None = None
    output_token_count: NonNegativeCount | None = None
    model_version: SafeVersion | None = None
    extraction_schema_version: SafeVersion | None = None
    workflow_version: SafeVersion | None = None

    @model_validator(mode="after")
    def require_sanitized_failure_code(self) -> SafeEventMetadata:
        """Require an allowlisted reason code whenever an operation fails closed."""

        if self.status in _FAILURE_STATUSES and self.error_code is None:
            raise ValueError("failure events require a sanitized error code")
        return self


class SafeLogEvent(_StrictModel):
    """Canonical JSON log payload; intentionally contains no message field."""

    record_schema_version: Literal["1"] = "1"
    event_id: EventId = Field(default_factory=new_event_id)
    occurred_at: datetime = Field(default_factory=utc_now)
    metadata: SafeEventMetadata

    @field_validator("occurred_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        """Require a timezone-aware event time and normalize it to UTC."""

        return _as_utc(value)


class AuditRecord(_StrictModel):
    """A validated append-only local audit record."""

    record_schema_version: Literal["1"] = "1"
    event_id: EventId
    sequence: PositiveSequence
    occurred_at: datetime
    metadata: SafeEventMetadata

    @field_validator("occurred_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        """Require a timezone-aware audit time and normalize it to UTC."""

        return _as_utc(value)


class SanitizedError(_StrictModel):
    """Safe exception classification with no exception text or stack trace."""

    error_code: ErrorCode
    retryable: StrictBool


class AuditStatusCounts(_StrictModel):
    """Aggregate event totals that reveal no document or extracted values."""

    started: NonNegativeCount = 0
    succeeded: NonNegativeCount = 0
    failed: NonNegativeCount = 0
    rejected: NonNegativeCount = 0
    blocked: NonNegativeCount = 0
    unsupported: NonNegativeCount = 0
    uncertain: NonNegativeCount = 0
    human_review: NonNegativeCount = 0
    disabled: NonNegativeCount = 0


class AuditReport(_StrictModel):
    """PII-safe aggregate report for all events or one opaque correlation ID."""

    generated_at: datetime = Field(default_factory=utc_now)
    correlation_id: CorrelationId | None = None
    record_count: NonNegativeCount
    first_sequence: PositiveSequence | None = None
    last_sequence: PositiveSequence | None = None
    first_occurred_at: datetime | None = None
    last_occurred_at: datetime | None = None
    status_counts: AuditStatusCounts

    @field_validator("generated_at", "first_occurred_at", "last_occurred_at")
    @classmethod
    def normalize_timestamps(cls, value: datetime | None) -> datetime | None:
        """Normalize each present report timestamp to UTC."""

        return None if value is None else _as_utc(value)

    @model_validator(mode="after")
    def validate_empty_report_bounds(self) -> AuditReport:
        """Keep sequence and time bounds consistent with the report record count."""

        bounds = (
            self.first_sequence,
            self.last_sequence,
            self.first_occurred_at,
            self.last_occurred_at,
        )
        if self.record_count == 0 and any(item is not None for item in bounds):
            raise ValueError("empty audit reports cannot have record bounds")
        if self.record_count > 0 and any(item is None for item in bounds):
            raise ValueError("non-empty audit reports require record bounds")
        return self


class AuditIntegrityReport(_StrictModel):
    """Result of validating the complete local audit chain."""

    verified_at: datetime = Field(default_factory=utc_now)
    valid: Literal[True] = True
    record_count: NonNegativeCount
    first_sequence: PositiveSequence | None = None
    last_sequence: PositiveSequence | None = None

    @field_validator("verified_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        """Require a timezone-aware verification time and normalize it to UTC."""

        return _as_utc(value)

    @model_validator(mode="after")
    def validate_sequence_bounds(self) -> AuditIntegrityReport:
        """Require sequence bounds exactly when the audit contains records."""

        if self.record_count == 0:
            if self.first_sequence is not None or self.last_sequence is not None:
                raise ValueError("empty audit integrity reports cannot have bounds")
        elif self.first_sequence is None or self.last_sequence is None:
            raise ValueError("non-empty audit integrity reports require bounds")
        return self


def _as_utc(value: datetime) -> datetime:
    """Reject naive datetimes and return an equivalent UTC value."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    if value.utcoffset() != timedelta(0):
        value = value.astimezone(timezone.utc)
    return value


def sanitize_error_code(value: object) -> ErrorCode:
    """Return an allowlisted error code, defaulting unknown input safely."""

    if type(value) is str and value in _ALLOWED_ERROR_CODES:
        return cast(ErrorCode, value)
    return "INTERNAL_ERROR"


def sanitize_exception(exception: BaseException) -> SanitizedError:
    """Classify an exception without retaining its text, arguments, or traceback."""

    if isinstance(exception, AuditIntegrityError):
        return SanitizedError(error_code="AUDIT_INTEGRITY_ERROR", retryable=False)
    if isinstance(exception, AuditWriteError):
        return SanitizedError(error_code="AUDIT_WRITE_ERROR", retryable=True)
    if isinstance(exception, UnsafeAuditPathError):
        return SanitizedError(error_code="PATH_POLICY_VIOLATION", retryable=False)
    if isinstance(exception, ValidationError):
        return SanitizedError(error_code="SCHEMA_VALIDATION_FAILED", retryable=False)
    if isinstance(exception, TimeoutError):
        return SanitizedError(error_code="LOCAL_MODEL_TIMEOUT", retryable=True)
    if isinstance(exception, ConnectionError):
        return SanitizedError(error_code="LOCAL_MODEL_UNAVAILABLE", retryable=True)
    if isinstance(exception, PermissionError):
        return SanitizedError(error_code="STORAGE_ERROR", retryable=False)
    if isinstance(exception, OSError):
        return SanitizedError(error_code="STORAGE_ERROR", retryable=True)
    if isinstance(exception, (TypeError, ValueError)):
        return SanitizedError(error_code="INVALID_INPUT", retryable=False)
    return SanitizedError(error_code="INTERNAL_ERROR", retryable=False)


def _revalidate_safe_metadata(metadata: object) -> SafeEventMetadata:
    """Revalidate even constructed instances before any observable side effect."""

    if not isinstance(metadata, SafeEventMetadata):
        raise TypeError("safe event metadata is required")
    try:
        return SafeEventMetadata.model_validate(
            metadata.model_dump(mode="python", warnings="none")
        )
    except (ValidationError, TypeError, ValueError):
        raise TypeError("safe event metadata is invalid") from None


def _revalidate_safe_log_event(event: object) -> SafeLogEvent:
    """Reject bypass-constructed events before logging or local persistence."""

    if not isinstance(event, SafeLogEvent):
        raise TypeError("a validated safe log event is required")
    try:
        return SafeLogEvent.model_validate(
            event.model_dump(mode="python", warnings="none")
        )
    except (ValidationError, TypeError, ValueError):
        raise TypeError("a validated safe log event is required") from None


class PIISafeStructuredLogger:
    """Emit JSON logs from validated safe metadata only.

    The logger never accepts a free-form message and never attaches ``exc_info``.
    The supplied standard-library logger controls destinations and formatting.
    """

    __slots__ = ("_logger",)

    def __init__(self, logger: logging.Logger | None = None) -> None:
        """Bind a standard logger while keeping this wrapper's schema-only API."""

        self._logger = (
            logger
            if logger is not None
            else logging.getLogger("financial_reviewer.safe")
        )

    def emit(self, metadata: SafeEventMetadata) -> SafeLogEvent:
        """Revalidate safe metadata, create an event, and emit its JSON form."""

        safe_metadata = _revalidate_safe_metadata(metadata)
        event = SafeLogEvent(metadata=safe_metadata)
        PIISafeStructuredLogger.emit_event(self, event)
        return event

    def emit_event(self, event: SafeLogEvent) -> None:
        """Log one revalidated closed-schema event at a status-derived level."""

        safe_event = _revalidate_safe_log_event(event)
        level = (
            logging.ERROR
            if safe_event.metadata.status in _ERROR_LOG_STATUSES
            else logging.INFO
        )
        self._logger.log(
            level,
            "%s",
            safe_event.model_dump_json(exclude_none=True),
            exc_info=None,
            stack_info=False,
        )

    def emit_exception(
        self,
        metadata: SafeEventMetadata,
        exception: BaseException,
    ) -> SafeLogEvent:
        """Emit only the sanitized classification of ``exception``."""

        metadata = _revalidate_safe_metadata(metadata)
        if not isinstance(exception, BaseException):
            raise TypeError("an exception instance is required")
        safe_error = sanitize_exception(exception)
        safe_values = metadata.model_dump(warnings="none")
        safe_values["status"] = "failed"
        safe_values["error_code"] = safe_error.error_code
        return self.emit(SafeEventMetadata.model_validate(safe_values))


@runtime_checkable
class AuditStore(Protocol):
    """Append-only local audit interface used by workflow components."""

    def append(self, metadata: SafeEventMetadata) -> AuditRecord:
        """Append validated safe metadata as a new audit record."""

    def append_event(self, event: SafeLogEvent) -> AuditRecord:
        """Append a pre-created safe event, preserving its event ID and time."""

    def list_records(
        self,
        *,
        correlation_id: str | None = None,
        after_sequence: int = 0,
        limit: int | None = None,
    ) -> tuple[AuditRecord, ...]:
        """Return validated records matching only opaque operational filters."""

    def report(self, *, correlation_id: str | None = None) -> AuditReport:
        """Return a PII-safe aggregate report."""

    def verify_integrity(self) -> AuditIntegrityReport:
        """Validate every record and the monotonic local sequence."""


class LocalJsonlAuditStore:
    """Thread-safe local append-only JSONL implementation of :class:`AuditStore`.

    ``path`` must identify a ``.jsonl`` file in a private directory.  A missing
    parent directory is created with mode ``0700``.  An existing parent must
    already be owned by the current user and have mode ``0700``; this avoids
    silently changing permissions on broad directories such as a workspace or
    ``/tmp``.  The audit file is always forced to mode ``0600``.
    """

    _MAX_RECORD_BYTES: Final[int] = 16_384
    _MAX_AUDIT_BYTES: Final[int] = 64 * 1024 * 1024
    _MAX_QUERY_LIMIT: Final[int] = 100_000
    __slots__ = ("_lock", "_path")

    def __init__(self, path: str | os.PathLike[str]) -> None:
        """Secure the audit path, initialize its file, and verify existing records."""

        self._lock = threading.RLock()
        self._path = self._prepare_path(path)
        self._initialize_file()
        self.verify_integrity()

    @property
    def path(self) -> Path:
        """Return the local audit path; it is never included in an event."""

        return self._path

    def append(self, metadata: SafeEventMetadata) -> AuditRecord:
        """Wrap validated metadata in a new event and append it durably."""

        safe_metadata = _revalidate_safe_metadata(metadata)
        return self.append_event(SafeLogEvent(metadata=safe_metadata))

    def append_event(self, event: SafeLogEvent) -> AuditRecord:
        """Append one unique event with the next monotonic local sequence number."""

        event = _revalidate_safe_log_event(event)

        with self._lock:
            fd = self._open_file(os.O_RDWR | os.O_APPEND)
            try:
                self._acquire_file_lock(fd, exclusive=True)
                records = self._records_from_fd(fd)
                if os.fstat(fd).st_size != os.lseek(fd, 0, os.SEEK_CUR):
                    raise AuditIntegrityError("audit file changed during validation")
                if any(record.event_id == event.event_id for record in records):
                    raise AuditIntegrityError("duplicate audit event identifier")
                record = AuditRecord(
                    event_id=event.event_id,
                    sequence=len(records) + 1,
                    occurred_at=event.occurred_at,
                    metadata=event.metadata,
                )
                payload = record.model_dump_json(exclude_none=True).encode("utf-8") + b"\n"
                if len(payload) > self._MAX_RECORD_BYTES:
                    raise AuditWriteError("audit record exceeds the safe size limit")
                self._append_all(fd, payload)
                os.fsync(fd)
                return record
            except (AuditIntegrityError, AuditWriteError):
                raise
            except OSError:
                raise AuditWriteError("local audit append failed") from None
            finally:
                self._release_file_lock(fd)
                os.close(fd)

    def list_records(
        self,
        *,
        correlation_id: str | None = None,
        after_sequence: int = 0,
        limit: int | None = None,
    ) -> tuple[AuditRecord, ...]:
        """Return validated records using only opaque and bounded filters."""

        if correlation_id is not None:
            correlation_id = validate_correlation_id(correlation_id)
        if type(after_sequence) is not int or after_sequence < 0:
            raise ValueError("after_sequence must be a non-negative integer")
        if limit is not None and (
            type(limit) is not int or limit < 1 or limit > self._MAX_QUERY_LIMIT
        ):
            raise ValueError("limit is outside the allowed range")

        records = self._read_validated_records()
        selected = (
            record
            for record in records
            if record.sequence > after_sequence
            and (
                correlation_id is None
                or record.metadata.correlation_id == correlation_id
            )
        )
        if limit is None:
            return tuple(selected)

        result: list[AuditRecord] = []
        for record in selected:
            result.append(record)
            if len(result) == limit:
                break
        return tuple(result)

    def report(self, *, correlation_id: str | None = None) -> AuditReport:
        """Build a PII-safe count-and-range summary of validated audit records."""

        records = self.list_records(correlation_id=correlation_id)
        status_counts = Counter(record.metadata.status for record in records)
        counts = AuditStatusCounts(
            started=status_counts["started"],
            succeeded=status_counts["succeeded"],
            failed=status_counts["failed"],
            rejected=status_counts["rejected"],
            blocked=status_counts["blocked"],
            unsupported=status_counts["unsupported"],
            uncertain=status_counts["uncertain"],
            human_review=status_counts["human_review"],
            disabled=status_counts["disabled"],
        )
        if not records:
            return AuditReport(
                correlation_id=correlation_id,
                record_count=0,
                status_counts=counts,
            )
        return AuditReport(
            correlation_id=correlation_id,
            record_count=len(records),
            first_sequence=records[0].sequence,
            last_sequence=records[-1].sequence,
            first_occurred_at=records[0].occurred_at,
            last_occurred_at=records[-1].occurred_at,
            status_counts=counts,
        )

    def verify_integrity(self) -> AuditIntegrityReport:
        """Validate the entire local file and report its verified sequence bounds."""

        records = self._read_validated_records()
        if not records:
            return AuditIntegrityReport(record_count=0)
        return AuditIntegrityReport(
            record_count=len(records),
            first_sequence=records[0].sequence,
            last_sequence=records[-1].sequence,
        )

    @classmethod
    def _prepare_path(cls, path: str | os.PathLike[str]) -> Path:
        """Require a private local JSONL path with safe ownership and permissions."""

        try:
            raw_path = os.fspath(path)
        except TypeError:
            raise UnsafeAuditPathError("audit path must be a local .jsonl file") from None
        if type(raw_path) is not str or "\x00" in raw_path or "://" in raw_path:
            raise UnsafeAuditPathError("audit path must be a local .jsonl file")

        candidate = Path(raw_path)
        if candidate.suffix != ".jsonl" or not candidate.name:
            raise UnsafeAuditPathError("audit path must be a local .jsonl file")
        if candidate.is_symlink() or (
            candidate.parent.exists() and candidate.parent.is_symlink()
        ):
            raise UnsafeAuditPathError("audit path cannot use a symbolic link")

        parent = candidate.parent.resolve()
        created_parent = not parent.exists()
        try:
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if created_parent:
                os.chmod(parent, 0o700)
            parent_stat = parent.stat()
        except OSError:
            raise UnsafeAuditPathError("audit directory is unavailable") from None
        if not stat.S_ISDIR(parent_stat.st_mode):
            raise UnsafeAuditPathError("audit parent must be a directory")
        if hasattr(os, "getuid") and parent_stat.st_uid != os.getuid():
            raise UnsafeAuditPathError("audit directory ownership is invalid")
        if stat.S_IMODE(parent_stat.st_mode) != 0o700:
            raise UnsafeAuditPathError("audit directory permissions must be 0700")
        return parent / candidate.name

    def _initialize_file(self) -> None:
        """Create or open the audit file with owner-only permissions and sync it."""

        fd = self._open_file(os.O_WRONLY | os.O_APPEND | os.O_CREAT)
        try:
            os.fchmod(fd, 0o600)
            os.fsync(fd)
        except OSError:
            raise AuditWriteError("local audit initialization failed") from None
        finally:
            os.close(fd)

    def _open_file(self, flags: int) -> int:
        """Open the regular audit file without symlink following or descriptor leaks."""

        secure_flags = flags
        if hasattr(os, "O_CLOEXEC"):
            secure_flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            secure_flags |= os.O_NOFOLLOW
        try:
            fd = os.open(self._path, secure_flags, 0o600)
            file_stat = os.fstat(fd)
            if not stat.S_ISREG(file_stat.st_mode):
                raise UnsafeAuditPathError("audit target must be a regular file")
            if hasattr(os, "getuid") and file_stat.st_uid != os.getuid():
                raise UnsafeAuditPathError("audit file ownership is invalid")
            if stat.S_IMODE(file_stat.st_mode) != 0o600:
                os.fchmod(fd, 0o600)
            return fd
        except UnsafeAuditPathError:
            if "fd" in locals():
                os.close(fd)
            raise
        except OSError:
            if "fd" in locals():
                os.close(fd)
            raise UnsafeAuditPathError("audit file is unavailable") from None

    def _read_validated_records(self) -> tuple[AuditRecord, ...]:
        """Read all records while holding thread and shared process-level locks."""

        with self._lock:
            fd = self._open_file(os.O_RDONLY)
            try:
                self._acquire_file_lock(fd, exclusive=False)
                return self._records_from_fd(fd)
            finally:
                self._release_file_lock(fd)
                os.close(fd)

    def _records_from_fd(self, fd: int) -> tuple[AuditRecord, ...]:
        """Parse a bounded file and verify every record, event ID, and sequence."""

        try:
            file_size = os.fstat(fd).st_size
            if file_size > self._MAX_AUDIT_BYTES:
                raise AuditIntegrityError("audit file exceeds the safe size limit")
            os.lseek(fd, 0, os.SEEK_SET)
            chunks: list[bytes] = []
            remaining = file_size
            while remaining:
                chunk = os.read(fd, min(remaining, 64 * 1024))
                if not chunk:
                    raise AuditIntegrityError("audit file changed during validation")
                chunks.append(chunk)
                remaining -= len(chunk)
        except AuditIntegrityError:
            raise
        except OSError:
            raise AuditIntegrityError("audit file could not be validated") from None

        payload = b"".join(chunks)
        if not payload:
            return ()
        if not payload.endswith(b"\n"):
            raise AuditIntegrityError("audit file contains an incomplete record")

        records: list[AuditRecord] = []
        event_ids: set[str] = set()
        for expected_sequence, line in enumerate(payload.split(b"\n")[:-1], start=1):
            if not line or len(line) + 1 > self._MAX_RECORD_BYTES:
                raise AuditIntegrityError("audit record framing is invalid")
            try:
                record = AuditRecord.model_validate_json(line)
            except (ValidationError, ValueError, UnicodeError):
                raise AuditIntegrityError("audit record validation failed") from None
            if record.sequence != expected_sequence:
                raise AuditIntegrityError("audit sequence validation failed")
            if record.event_id in event_ids:
                raise AuditIntegrityError("duplicate audit event identifier")
            event_ids.add(record.event_id)
            records.append(record)
        return tuple(records)

    @staticmethod
    def _append_all(fd: int, payload: bytes) -> None:
        """Write the complete encoded record or raise a sanitized append error."""

        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise AuditWriteError("local audit append failed")
            view = view[written:]

    @staticmethod
    def _acquire_file_lock(fd: int, *, exclusive: bool) -> None:
        """Acquire a shared or exclusive advisory lock when POSIX locks exist."""

        if fcntl is None:
            return
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(fd, operation)
        except OSError:
            raise AuditIntegrityError("audit lock acquisition failed") from None

    @staticmethod
    def _release_file_lock(fd: int) -> None:
        """Best-effort release of the advisory lock during cleanup."""

        if fcntl is None:
            return
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass


# Small compatibility aliases; both names retain the same strict implementation.
SafeStructuredLogger = PIISafeStructuredLogger
LocalAuditStore = LocalJsonlAuditStore


__all__ = [
    "Action",
    "AuditIntegrityError",
    "AuditIntegrityReport",
    "AuditRecord",
    "AuditReport",
    "AuditStatusCounts",
    "AuditStore",
    "AuditWriteError",
    "Component",
    "CorrelationId",
    "DocumentType",
    "ErrorCode",
    "EventId",
    "EventStatus",
    "IdempotencyKey",
    "LocalAuditStore",
    "LocalJsonlAuditStore",
    "OpaqueDocumentId",
    "PIISafeStructuredLogger",
    "SafeEventMetadata",
    "SafeLogEvent",
    "SafeStructuredLogger",
    "SanitizedError",
    "UnsafeAuditPathError",
    "new_correlation_id",
    "new_event_id",
    "new_idempotency_key",
    "new_opaque_document_id",
    "sanitize_error_code",
    "sanitize_exception",
    "utc_now",
    "validate_correlation_id",
    "validate_event_id",
    "validate_idempotency_key",
    "validate_opaque_document_id",
]
