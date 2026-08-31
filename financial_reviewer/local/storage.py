"""Store validated documents locally under an isolated review session.

Why this file exists:
    The workflow needs durable local bytes and later source verification without
    placing documents in graph state, logs, a database, or cloud storage.

What it owns:
    The ``DocumentStore`` interface, private correlation-scoped directories,
    opaque document IDs, restrictive path/permission checks, atomic creation,
    text read-back, byte/hash integrity verification, and session deletion.

What it does not own:
    It accepts only ``ValidatedDocument`` objects; intake policy happens first.
    It does not parse documents, call the model, log document content, or decide
    whether an extraction may be released.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
import uuid
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from pydantic import Field, StrictInt, StrictStr

from financial_reviewer.foundation.intake import DEFAULT_MAX_DOCUMENT_BYTES, ValidatedDocument
from financial_reviewer.foundation.schemas import StrictModel


# Opaque identifiers are the only caller-controlled path components. Their
# fixed grammars prevent traversal and exclude filenames or business IDs.
_CORRELATION_ID = re.compile(r"^corr_[0-9a-f]{32}$")
_DOCUMENT_ID = re.compile(r"^doc_[0-9a-f]{32}$")


class StorageError(RuntimeError):
    """Sanitized local-storage failure."""

    def __init__(self, code: str = "local_storage_error") -> None:
        """Expose only a safe failure code, never a filesystem path or document data."""

        self.code = code
        super().__init__(code)


class StoredDocument(StrictModel):
    """Opaque metadata needed to locate and integrity-check a stored document."""

    document_id: StrictStr = Field(pattern=r"^doc_[0-9a-f]{32}$")
    content_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: StrictInt = Field(ge=1)
    content_type: Literal["text/plain"] = "text/plain"


@runtime_checkable
class DocumentStore(Protocol):
    """Storage seam kept local for this milestone."""

    def store(self, correlation_id: str, document: ValidatedDocument) -> StoredDocument:
        """Persist a validated document under one opaque review session."""

        ...

    def read_text(self, correlation_id: str, document_id: str) -> str:
        """Read one correlation-owned document as strict UTF-8 text."""

        ...

    def delete_session(self, correlation_id: str) -> int:
        """Delete the files belonging to one explicit retention scope."""

        ...


class LocalDocumentStore:
    """Permission-restricted filesystem store with opaque path components."""

    __slots__ = ("_max_document_bytes", "_root")

    def __init__(
        self,
        root: Path,
        *,
        max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
    ) -> None:
        """Validate or create a private storage root and bind its size limit."""

        if not 1 <= max_document_bytes <= 10 * 1024 * 1024:
            raise ValueError("invalid storage byte limit")
        existed = root.exists()
        try:
            if root.is_symlink():
                raise StorageError()
            root.mkdir(parents=True, exist_ok=True, mode=0o700)
            metadata = root.stat()
            if not stat.S_ISDIR(metadata.st_mode):
                raise StorageError()
            if hasattr(os, "getuid") and metadata.st_uid != os.getuid():
                raise StorageError()
            if stat.S_IMODE(metadata.st_mode) != 0o700:
                if existed:
                    raise StorageError()
                os.chmod(root, 0o700)
        except OSError:
            raise StorageError() from None
        self._root = root.resolve(strict=True)
        self._max_document_bytes = max_document_bytes

    @property
    def root(self) -> Path:
        """Return the resolved, permission-checked local storage root."""

        return self._root

    def store(self, correlation_id: str, document: ValidatedDocument) -> StoredDocument:
        """Atomically write validated bytes using only opaque directory and file names."""

        try:
            validated = ValidatedDocument.model_validate(
                document.model_dump(mode="python", warnings="none")
            )
            decoded = validated.normalized_bytes.decode("utf-8", errors="strict")
        except (AttributeError, UnicodeError, ValueError):
            raise StorageError() from None
        if (
            decoded != validated.text
            or len(validated.normalized_bytes) != validated.byte_size
            or validated.byte_size > self._max_document_bytes
            or hashlib.sha256(validated.normalized_bytes).hexdigest()
            != validated.content_sha256
        ):
            raise StorageError()

        session_dir = self._session_dir(correlation_id, create=True)
        document_id = f"doc_{uuid.uuid4().hex}"
        final_path = session_dir / f"{document_id}.txt"
        temporary_path = session_dir / f".tmp_{uuid.uuid4().hex}"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
        descriptor: int | None = None
        try:
            descriptor = os.open(temporary_path, flags, 0o600)
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                descriptor = None
                stream.write(validated.normalized_bytes)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, final_path)
            os.chmod(final_path, 0o600)
        except OSError:
            if descriptor is not None:
                os.close(descriptor)
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise StorageError() from None

        return StoredDocument(
            document_id=document_id,
            content_sha256=validated.content_sha256,
            byte_size=validated.byte_size,
        )

    def read_text(self, correlation_id: str, document_id: str) -> str:
        """Safely read a bounded regular file without following symbolic links."""

        if not _DOCUMENT_ID.fullmatch(document_id):
            raise StorageError()
        session_dir = self._session_dir(correlation_id, create=False)
        path = session_dir / f"{document_id}.txt"
        if path.is_symlink():
            raise StorageError()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags)
            try:
                metadata = os.fstat(descriptor)
                if not stat.S_ISREG(metadata.st_mode):
                    raise StorageError()
                if metadata.st_size < 1 or metadata.st_size > self._max_document_bytes:
                    raise StorageError()
                with os.fdopen(descriptor, "rb", closefd=True) as stream:
                    descriptor = -1
                    payload = stream.read(self._max_document_bytes + 1)
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
        except (OSError, UnicodeError):
            raise StorageError() from None
        if len(payload) > self._max_document_bytes:
            raise StorageError()
        try:
            return payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise StorageError() from None

    def verify_integrity(
        self,
        correlation_id: str,
        stored: StoredDocument,
    ) -> bool:
        """Confirm that stored bytes still match their recorded length and SHA-256."""

        try:
            text = self.read_text(correlation_id, stored.document_id)
        except StorageError:
            return False
        payload = text.encode("utf-8")
        return (
            len(payload) == stored.byte_size
            and hashlib.sha256(payload).hexdigest() == stored.content_sha256
        )

    def delete_session(self, correlation_id: str) -> int:
        """Delete one validated opaque session directory for explicit retention cleanup."""

        session_dir = self._session_dir(correlation_id, create=False)
        removed = 0
        try:
            for path in session_dir.iterdir():
                if path.is_symlink() or not path.is_file():
                    raise StorageError()
                path.unlink()
                removed += 1
            session_dir.rmdir()
        except OSError:
            raise StorageError() from None
        return removed

    def _session_dir(self, correlation_id: str, *, create: bool) -> Path:
        """Resolve one opaque session directory and prevent traversal or symlink escape."""

        if not _CORRELATION_ID.fullmatch(correlation_id):
            raise StorageError()
        path = self._root / correlation_id
        if create:
            try:
                if path.is_symlink():
                    raise StorageError()
                path.mkdir(mode=0o700, exist_ok=True)
                if path.is_symlink():
                    raise StorageError()
                os.chmod(path, 0o700)
            except OSError:
                raise StorageError() from None
        if path.is_symlink() or not path.is_dir():
            raise StorageError()
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            raise StorageError() from None
        if resolved.parent != self._root:
            raise StorageError()
        return resolved
