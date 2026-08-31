from __future__ import annotations

import stat
from pathlib import Path

import pytest

from financial_reviewer.foundation.intake import (
    DEFAULT_MAX_DOCUMENT_BYTES,
    SYNTHETIC_MARKER,
    DocumentSubmission,
    IntakeError,
    SecureIntake,
)
from financial_reviewer.local.observability import new_correlation_id
from financial_reviewer.local.storage import LocalDocumentStore, StorageError


@pytest.mark.parametrize(
    "payload",
    [
        {
            "filename": "../escape.txt",
            "content_type": "text/plain",
            "content": (SYNTHETIC_MARKER + "\nPAY STUB\n").encode(),
            "declared_synthetic": True,
        },
        {
            "filename": "sample.pdf",
            "content_type": "text/plain",
            "content": (SYNTHETIC_MARKER + "\nPAY STUB\n").encode(),
            "declared_synthetic": True,
        },
        {
            "filename": "sample.txt",
            "content_type": "application/pdf",
            "content": (SYNTHETIC_MARKER + "\nPAY STUB\n").encode(),
            "declared_synthetic": True,
        },
        {
            "filename": "sample.txt",
            "content_type": "text/plain",
            "content": b"",
            "declared_synthetic": True,
        },
        {
            "filename": "sample.txt",
            "content_type": "text/plain",
            "content": b"REAL DOCUMENT\nPAY STUB\n",
            "declared_synthetic": True,
        },
        {
            "filename": "sample.txt",
            "content_type": "text/plain",
            "content": (SYNTHETIC_MARKER + "\nPAY\x00 STUB\n").encode(),
            "declared_synthetic": True,
        },
        {
            "filename": "sample.txt",
            "content_type": "text/plain",
            "content": b"\xff\xfe\x00\x00",
            "declared_synthetic": True,
        },
        {
            "filename": "sample.txt",
            "content_type": "text/plain",
            "content": (SYNTHETIC_MARKER.encode() + b"\n" + b"A" * DEFAULT_MAX_DOCUMENT_BYTES),
            "declared_synthetic": True,
        },
        {
            "filename": "sample.txt",
            "content_type": "text/plain",
            "content": (SYNTHETIC_MARKER + "\nPAY STUB\n").encode(),
            "declared_synthetic": 1,
        },
        {
            "filename": "sample.txt",
            "content_type": "text/plain",
            "content": (SYNTHETIC_MARKER + "\nPAY\u0085STUB\n").encode(),
            "declared_synthetic": True,
        },
    ],
)
def test_secure_intake_rejects_unsafe_input(payload: dict[str, object]) -> None:
    with pytest.raises(IntakeError):
        SecureIntake().validate(payload)


def test_secure_intake_normalizes_only_marked_synthetic_text(
    synthetic_submission: DocumentSubmission,
) -> None:
    document = SecureIntake().validate(synthetic_submission)
    assert document.synthetic_marker_verified is True
    assert document.content_sha256
    assert document.text.startswith(SYNTHETIC_MARKER)
    assert document.byte_size == len(document.normalized_bytes)


def test_constructed_submission_cannot_bypass_boundary_validation() -> None:
    bypass_attempt = DocumentSubmission.model_construct(
        filename="../synthetic.pdf",
        content_type="application/pdf",
        content=(SYNTHETIC_MARKER + "\nPAY STUB\n").encode(),
        declared_synthetic=False,
    )
    with pytest.raises(IntakeError):
        SecureIntake().validate(bypass_attempt)


def test_local_store_is_session_isolated_and_permission_restricted(
    tmp_path: Path,
    synthetic_submission: DocumentSubmission,
) -> None:
    document = SecureIntake().validate(synthetic_submission)
    store = LocalDocumentStore(tmp_path / "private_documents")
    correlation_id = new_correlation_id()
    stored = store.store(correlation_id, document)

    session_dir = store.root / correlation_id
    files = tuple(session_dir.iterdir())
    assert len(files) == 1
    assert files[0].name == f"{stored.document_id}.txt"
    assert synthetic_submission.filename not in str(files[0])
    assert stat.S_IMODE(store.root.stat().st_mode) == 0o700
    assert stat.S_IMODE(session_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(files[0].stat().st_mode) == 0o600
    assert store.read_text(correlation_id, stored.document_id) == document.text
    assert store.verify_integrity(correlation_id, stored)


def test_local_store_rejects_unscoped_paths(
    tmp_path: Path,
    synthetic_submission: DocumentSubmission,
) -> None:
    store = LocalDocumentStore(tmp_path / "private_documents")
    correlation_id = new_correlation_id()
    document = SecureIntake().validate(synthetic_submission)
    stored = store.store(correlation_id, document)

    with pytest.raises(StorageError):
        store.read_text("corr_../escape", stored.document_id)
    with pytest.raises(StorageError):
        store.read_text(correlation_id, "../escape")


def test_local_store_explicit_session_cleanup(
    tmp_path: Path,
    synthetic_submission: DocumentSubmission,
) -> None:
    store = LocalDocumentStore(tmp_path / "private_documents")
    correlation_id = new_correlation_id()
    store.store(correlation_id, SecureIntake().validate(synthetic_submission))
    assert store.delete_session(correlation_id) == 1
    assert not (store.root / correlation_id).exists()


def test_local_store_rejects_existing_broad_root_without_chmod(
    tmp_path: Path,
) -> None:
    shared_root = tmp_path / "shared_documents"
    shared_root.mkdir(mode=0o755)
    shared_root.chmod(0o755)
    with pytest.raises(StorageError):
        LocalDocumentStore(shared_root)
    assert stat.S_IMODE(shared_root.stat().st_mode) == 0o755
