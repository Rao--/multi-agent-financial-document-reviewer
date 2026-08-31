from pathlib import Path

import pytest
from langchain_core.globals import set_debug, set_verbose

from financial_reviewer.foundation.intake import DocumentSubmission


@pytest.fixture(autouse=True)
def disable_ambient_cloud_tracing(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "LANGSMITH_TRACING",
        "LANGSMITH_TRACING_V2",
        "LANGCHAIN_TRACING",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_HANDLER",
        "LANGCHAIN_DEBUG",
        "LANGCHAIN_VERBOSE",
    ):
        monkeypatch.delenv(name, raising=False)
    set_debug(False)
    set_verbose(False)
    yield
    set_debug(False)
    set_verbose(False)


@pytest.fixture
def synthetic_pay_stub_text() -> str:
    return (
        Path(__file__).parent / "fixtures" / "synthetic_pay_stub.txt"
    ).read_text(encoding="utf-8")


@pytest.fixture
def synthetic_w2_text() -> str:
    """Load the synthetic W-2 fixture used by deterministic extraction tests."""

    return (Path(__file__).parent / "fixtures" / "synthetic_w2.txt").read_text(
        encoding="utf-8"
    )


@pytest.fixture
def synthetic_bank_statement_text() -> str:
    """Load the synthetic bank-statement fixture used by extraction tests."""

    return (
        Path(__file__).parent / "fixtures" / "synthetic_bank_statement.txt"
    ).read_text(encoding="utf-8")


@pytest.fixture
def synthetic_submission(synthetic_pay_stub_text: str) -> DocumentSubmission:
    return DocumentSubmission(
        filename="synthetic_pay_stub.txt",
        content_type="text/plain",
        content=synthetic_pay_stub_text.encode("utf-8"),
        declared_synthetic=True,
    )
