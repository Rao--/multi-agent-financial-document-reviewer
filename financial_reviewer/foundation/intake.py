"""Validate untrusted input before storage, parsing, or model access.

Why this file exists:
    Invalid or unexpectedly real financial content must be stopped at the
    outer boundary rather than discovered after it reaches the model.

What it owns:
    Submission shape, safe filename and content-type rules, explicit synthetic
    declaration, size/encoding checks, the required synthetic marker, and the
    immutable ``ValidatedDocument`` returned to the reviewer.

What it does not own:
    It does not store files, classify document type, extract fields, or support
    PDFs yet. Any ``IntakeError`` is handled by
    ``DocumentExtractionReviewer.review`` and
    becomes a sanitized human-review outcome without a model call.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import PurePath
from typing import Literal, Mapping, Any

from pydantic import Field, StrictBool, StrictBytes, StrictStr, field_validator

from financial_reviewer.foundation.schemas import StrictModel


# Required in-file declaration. The separate boolean declaration must also be
# true, giving intake two independent signals that the fixture is synthetic.
SYNTHETIC_MARKER = "SYNTHETIC TEST DOCUMENT - NOT REAL"
# Pre-model resource limit for this text-only milestone.
DEFAULT_MAX_DOCUMENT_BYTES = 64 * 1024
# Closed filename grammar: basename-like ASCII `.txt` names only, with no paths.
_SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,126}\.txt$")

# Kept as a named alias so generated schemas and error locations stay readable.
AnnotatedFilename = StrictStr


class IntakeError(ValueError):
    """A sanitized intake failure that is safe to classify and log."""

    def __init__(self, code: str = "invalid_input") -> None:
        """Create a failure carrying only the safe intake classification code."""

        self.code = code
        super().__init__(code)


class DocumentSubmission(StrictModel):
    """An explicitly synthetic, local text document submitted for review."""

    filename: AnnotatedFilename
    content_type: Literal["text/plain"]
    content: StrictBytes = Field(repr=False)
    declared_synthetic: StrictBool

    @field_validator("filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        """Allow one simple local ``.txt`` basename with no path traversal."""

        if PurePath(value).name != value or not _SAFE_FILENAME.fullmatch(value):
            raise ValueError("unsafe filename")
        return value

    @field_validator("declared_synthetic")
    @classmethod
    def require_explicit_synthetic_declaration(cls, value: bool) -> bool:
        """Require the caller to affirm that the document is synthetic."""

        if value is not True:
            raise ValueError("synthetic declaration is required")
        return value

class ValidatedDocument(StrictModel):
    """Normalized content that has passed every pre-model validation check."""

    filename: StrictStr = Field(repr=False)
    content_type: Literal["text/plain"] = "text/plain"
    normalized_bytes: StrictBytes = Field(repr=False)
    text: StrictStr = Field(repr=False)
    byte_size: int = Field(ge=1)
    content_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    synthetic_marker_verified: Literal[True] = True


class SecureIntake:
    """Validate input before it can be stored or supplied to a model."""

    __slots__ = ("_max_document_bytes",)

    def __init__(self, *, max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES) -> None:
        """Set a bounded byte limit for documents accepted at this boundary."""

        if not 1 <= max_document_bytes <= 10 * 1024 * 1024:
            raise ValueError("invalid intake byte limit")
        self._max_document_bytes = max_document_bytes

    def validate(
        self,
        submission: DocumentSubmission | Mapping[str, Any],
    ) -> ValidatedDocument:
        """Validate, decode, normalize, and mark safe synthetic text for storage.

        Validation is repeated even for an existing ``DocumentSubmission`` so a
        caller cannot bypass field validators with Pydantic ``model_construct``.
        Any malformed shape, size, encoding, control character, normalization
        change, or missing synthetic marker becomes a sanitized ``IntakeError``.
        """

        try:
            # Existing Pydantic instances can be created with ``model_construct``;
            # round-trip through plain data so the trust boundary never skips
            # validation merely because the input has the expected Python type.
            candidate = (
                submission.model_dump(mode="python", warnings="none")
                if isinstance(submission, DocumentSubmission)
                else submission
            )
            parsed = DocumentSubmission.model_validate(candidate)
        except Exception:
            raise IntakeError() from None

        raw = parsed.content
        if not raw or len(raw) > self._max_document_bytes:
            raise IntakeError()

        try:
            decoded = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise IntakeError() from None

        normalized = unicodedata.normalize("NFC", decoded).replace("\r\n", "\n").replace("\r", "\n")
        if not normalized.strip():
            raise IntakeError()

        # Reject NULs, hidden formatting controls, and non-whitespace C0 controls.
        for character in normalized:
            if character in "\n\t":
                continue
            if unicodedata.category(character) in {"Cc", "Cf", "Cs"}:
                raise IntakeError()

        first_nonempty = next(
            (line.strip() for line in normalized.splitlines() if line.strip()),
            "",
        )
        if first_nonempty != SYNTHETIC_MARKER:
            raise IntakeError()

        normalized_bytes = normalized.encode("utf-8")
        if len(normalized_bytes) > self._max_document_bytes:
            raise IntakeError()

        return ValidatedDocument(
            filename=parsed.filename,
            normalized_bytes=normalized_bytes,
            text=normalized,
            byte_size=len(normalized_bytes),
            content_sha256=hashlib.sha256(normalized_bytes).hexdigest(),
        )
