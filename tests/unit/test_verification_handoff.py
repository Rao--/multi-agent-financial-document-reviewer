"""Contract tests for the deterministic Agent 1 → Agent 2 handoff."""

from __future__ import annotations

import traceback
from decimal import Decimal

import pytest

from financial_reviewer.agents.agent1_extraction import (
    classify_document,
    enforce_evidence_guard,
    extract_bank_statement_deterministically,
    extract_pay_stub_deterministically,
    extract_tax_form_deterministically,
)
from financial_reviewer.agents.agent2_verification import (
    IncomeBasis,
    IncomePeriod,
)
from financial_reviewer.foundation.handoffs import (
    VerificationHandoffError,
    VerificationHandoffFailureCode,
    VerificationInputAssembler,
)
from financial_reviewer.foundation.schemas import (
    DocumentType,
    UnsupportedField,
    UnsupportedReason,
)
from financial_reviewer.local.observability import new_opaque_document_id


def _guarded(document_text: str, extractor):
    """Produce the same evidence-attested Agent 1 artifact used by the workflow."""

    _, classification = classify_document(document_text)
    assert classification is not None
    extraction = extractor(
        document_id=new_opaque_document_id(),
        document_text=document_text,
        classification=classification,
    )
    return enforce_evidence_guard(extraction, document_text)


def test_assembler_builds_ordered_paystub_w2_request_from_guarded_evidence(
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
) -> None:
    """Input order cannot change Agent 2's pay-stub-then-W-2 contract."""

    pay_stub = _guarded(synthetic_pay_stub_text, extract_pay_stub_deterministically)
    tax_form = _guarded(synthetic_w2_text, extract_tax_form_deterministically)

    request = VerificationInputAssembler.assemble((tax_form, pay_stub))

    pay_stub_evidence, w2_evidence = request.evidence
    assert pay_stub_evidence.document_type is DocumentType.PAY_STUB
    assert pay_stub_evidence.document_ref == pay_stub.document_id
    assert pay_stub_evidence.amount == Decimal("6250.00")
    assert pay_stub_evidence.period is IncomePeriod.MONTHLY
    assert pay_stub_evidence.income_basis is IncomeBasis.GROSS
    assert pay_stub_evidence.calendar_year == 2025
    assert pay_stub_evidence.provenance == pay_stub.fields.monthly_income.provenance

    assert w2_evidence.document_type is DocumentType.TAX_FORM
    assert w2_evidence.document_ref == tax_form.document_id
    assert w2_evidence.amount == Decimal("75000.00")
    assert w2_evidence.period is IncomePeriod.ANNUAL
    assert w2_evidence.income_basis is IncomeBasis.GROSS
    assert w2_evidence.calendar_year == 2025
    assert w2_evidence.provenance == tax_form.fields.annual_wages.provenance

    assert pay_stub_evidence.evidence_ref != w2_evidence.evidence_ref
    assert pay_stub_evidence.evidence_ref.startswith("evidence_")
    assert w2_evidence.evidence_ref.startswith("evidence_")
    for evidence in request.evidence:
        assert evidence.document_ref not in evidence.evidence_ref


def test_assembler_preserves_independent_years_for_agent2_comparison(
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
) -> None:
    """The handoff must not copy one document's year onto the other."""

    pay_stub = _guarded(synthetic_pay_stub_text, extract_pay_stub_deterministically)
    older_w2 = synthetic_w2_text.replace("Tax Year: 2025", "Tax Year: 2024")
    tax_form = _guarded(older_w2, extract_tax_form_deterministically)

    request = VerificationInputAssembler.assemble((pay_stub, tax_form))

    assert request.evidence[0].calendar_year == 2025
    assert request.evidence[1].calendar_year == 2024


def test_assembler_rejects_bank_statement_before_agent2(
    synthetic_pay_stub_text: str,
    synthetic_bank_statement_text: str,
) -> None:
    """A deposit total is not silently relabeled as verified payroll income."""

    pay_stub = _guarded(synthetic_pay_stub_text, extract_pay_stub_deterministically)
    bank_statement = _guarded(
        synthetic_bank_statement_text,
        extract_bank_statement_deterministically,
    )

    with pytest.raises(VerificationHandoffError) as captured:
        VerificationInputAssembler.assemble((pay_stub, bank_statement))

    assert captured.value.code is (
        VerificationHandoffFailureCode.UNSUPPORTED_DOCUMENT_COMBINATION
    )
    rendered = "".join(traceback.format_exception(captured.value))
    assert synthetic_bank_statement_text not in rendered
    assert "SYN-ACCT-0001" not in rendered
    assert "$6,250.00" not in rendered


def test_assembler_rejects_unguarded_extraction(
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
) -> None:
    """Schema-valid extraction cannot bypass the Agent 1 evidence guard."""

    _, classification = classify_document(synthetic_pay_stub_text)
    assert classification is not None
    unguarded_pay_stub = extract_pay_stub_deterministically(
        document_id=new_opaque_document_id(),
        document_text=synthetic_pay_stub_text,
        classification=classification,
    )
    tax_form = _guarded(synthetic_w2_text, extract_tax_form_deterministically)

    with pytest.raises(VerificationHandoffError) as captured:
        VerificationInputAssembler.assemble((unguarded_pay_stub, tax_form))

    assert captured.value.code is VerificationHandoffFailureCode.UNGUARDED_EXTRACTION


def test_assembler_rejects_unsupported_required_income_field(
    synthetic_pay_stub_text: str,
    synthetic_w2_text: str,
) -> None:
    """A guarded-looking constructed object cannot forward absent income."""

    pay_stub = _guarded(synthetic_pay_stub_text, extract_pay_stub_deterministically)
    tax_form = _guarded(synthetic_w2_text, extract_tax_form_deterministically)
    unsupported_fields = pay_stub.fields.model_copy(
        update={
            "monthly_income": UnsupportedField(
                status="unsupported",
                reason=UnsupportedReason.NOT_PRESENT,
            )
        }
    )
    constructed = pay_stub.model_copy(update={"fields": unsupported_fields})

    with pytest.raises(VerificationHandoffError) as captured:
        VerificationInputAssembler.assemble((constructed, tax_form))

    assert captured.value.code is (
        VerificationHandoffFailureCode.REQUIRED_INCOME_EVIDENCE_INVALID
    )


@pytest.mark.parametrize("extractions", [(), (object(), object())])
def test_assembler_rejects_invalid_boundary_input(extractions: tuple[object, ...]) -> None:
    """Invalid count or object shape fails with a sanitized boundary code."""

    with pytest.raises(VerificationHandoffError) as captured:
        VerificationInputAssembler.assemble(extractions)  # type: ignore[arg-type]

    expected = (
        VerificationHandoffFailureCode.INVALID_EXTRACTION_COUNT
        if len(extractions) != 2
        else VerificationHandoffFailureCode.INVALID_INPUT
    )
    assert captured.value.code is expected
