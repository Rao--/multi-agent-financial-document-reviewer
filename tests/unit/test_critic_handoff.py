"""Contract tests for the deterministic Agent 2 → Critic handoff."""

from __future__ import annotations

import traceback
from decimal import Decimal

import pytest

from financial_reviewer.agents.agent2_verification import (
    ComparisonResult,
    IncomeBasis,
    IncomeComparison,
    IncomeVerificationResult,
    NormalizedIncome,
    TransformationRule,
    VerificationFailureCode,
    VerificationReason,
    VerificationStatus,
)
from financial_reviewer.foundation.handoffs import (
    CriticHandoffError,
    CriticHandoffFailureCode,
    CriticInputAssembler,
)
from financial_reviewer.foundation.schemas import DocumentType, SourceProvenance


def _normalized_source(
    document_type: DocumentType,
    *,
    year: int = 2025,
) -> NormalizedIncome:
    """Create one private normalized source with a valid provenance chain."""

    if document_type is DocumentType.PAY_STUB:
        token = "a"
        amount = Decimal("6250.00")
        rule = TransformationRule.PAYSTUB_MONTHLY_V1
    else:
        token = "b"
        amount = Decimal("6250.00")
        rule = TransformationRule.W2_ANNUAL_TO_MONTHLY_V1
    document_ref = f"doc_{token * 32}"
    return NormalizedIncome(
        normalized_ref=f"norm_{token * 32}",
        document_ref=document_ref,
        document_type=document_type,
        monthly_amount=amount,
        income_basis=IncomeBasis.GROSS,
        calendar_year=year,
        source_evidence_ref=f"evidence_{token * 32}",
        provenance=(
            SourceProvenance(
                document_id=document_ref,
                line_start=7,
                line_end=7,
                char_start=100,
                char_end=120,
                evidence_sha256=token * 64,
                confidence=0.99,
            ),
        ),
        transformation_rule=rule,
    )


def _result(status: VerificationStatus) -> IncomeVerificationResult:
    """Create a valid private Agent 2 result for one supported terminal status."""

    if status is VerificationStatus.FAILED:
        return IncomeVerificationResult(
            normalized_income=(),
            comparisons=(),
            status=status,
            evidence_complete=False,
            unsupported_reasons=(),
            tool_call_count=0,
            invalid_decision_count=0,
            failure_code=VerificationFailureCode.MODEL_DECISION_FAILED,
        )
    if status is VerificationStatus.INSUFFICIENT_EVIDENCE:
        return IncomeVerificationResult(
            normalized_income=(),
            comparisons=(),
            status=status,
            evidence_complete=False,
            unsupported_reasons=(
                VerificationReason.UNSUPPORTED_DOCUMENT_COMBINATION,
            ),
            tool_call_count=0,
            invalid_decision_count=0,
        )

    pay_stub = _normalized_source(DocumentType.PAY_STUB)
    w2_year = 2024 if status is VerificationStatus.NOT_COMPARABLE else 2025
    w2 = _normalized_source(DocumentType.TAX_FORM, year=w2_year)
    if status is VerificationStatus.CONSISTENT:
        comparison_result = ComparisonResult.CONSISTENT
        reason = VerificationReason.EXACT_MATCH
        difference = Decimal("0.00")
        percentage = Decimal("0.00")
    elif status is VerificationStatus.INCONSISTENT:
        comparison_result = ComparisonResult.INCONSISTENT
        reason = VerificationReason.INCOME_VALUES_INCONSISTENT
        difference = Decimal("250.00")
        percentage = Decimal("4.00")
    else:
        comparison_result = ComparisonResult.NOT_COMPARABLE
        reason = VerificationReason.INCOME_PERIOD_NOT_COMPARABLE
        difference = None
        percentage = None
    comparison = IncomeComparison(
        left_income_ref=pay_stub.normalized_ref,
        right_income_ref=w2.normalized_ref,
        amount_difference=difference,
        percentage_difference=percentage,
        result=comparison_result,
        reason_code=reason,
    )
    return IncomeVerificationResult(
        normalized_income=(w2, pay_stub),
        comparisons=(comparison,),
        status=status,
        evidence_complete=True,
        unsupported_reasons=(
            (reason,) if status is VerificationStatus.NOT_COMPARABLE else ()
        ),
        tool_call_count=3,
        invalid_decision_count=0,
    )


@pytest.mark.parametrize(
    ("status", "expected_result", "expected_reason"),
    [
        (
            VerificationStatus.CONSISTENT,
            ComparisonResult.CONSISTENT,
            VerificationReason.EXACT_MATCH,
        ),
        (
            VerificationStatus.INCONSISTENT,
            ComparisonResult.INCONSISTENT,
            VerificationReason.INCOME_VALUES_INCONSISTENT,
        ),
        (
            VerificationStatus.NOT_COMPARABLE,
            ComparisonResult.NOT_COMPARABLE,
            VerificationReason.INCOME_PERIOD_NOT_COMPARABLE,
        ),
    ],
)
def test_assembler_maps_linked_comparison_without_private_values(
    status: VerificationStatus,
    expected_result: ComparisonResult,
    expected_reason: VerificationReason,
) -> None:
    """Supported comparison states become ordered, amount-free Critic requests."""

    request = CriticInputAssembler.assemble(_result(status))

    assert request.verification_status is status
    assert request.evidence_complete is True
    assert [source.document_type for source in request.sources] == [
        DocumentType.PAY_STUB,
        DocumentType.TAX_FORM,
    ]
    assert request.comparison is not None
    assert request.comparison.result is expected_result
    assert request.comparison.reason_code is expected_reason
    assert request.tool_call_count == 3
    rendered = request.model_dump_json()
    for private_value in (
        "monthly_amount",
        "6250.00",
        "amount_difference",
        "percentage_difference",
        "doc_" + "a" * 32,
        "doc_" + "b" * 32,
        "norm_" + "a" * 32,
        "norm_" + "b" * 32,
        "evidence_" + "a" * 32,
        "evidence_" + "b" * 32,
        "a" * 64,
        "b" * 64,
    ):
        assert private_value not in rendered


def test_assembler_maps_insufficient_evidence_without_inventing_sources() -> None:
    """An evidence gap remains explicit and does not acquire fabricated context."""

    request = CriticInputAssembler.assemble(
        _result(VerificationStatus.INSUFFICIENT_EVIDENCE)
    )

    assert request.verification_status is VerificationStatus.INSUFFICIENT_EVIDENCE
    assert request.evidence_complete is False
    assert request.sources == ()
    assert request.comparison is None


def test_assembler_rejects_failed_agent2_result() -> None:
    """An operational Agent 2 failure is never converted into a critic opinion."""

    with pytest.raises(CriticHandoffError) as captured:
        CriticInputAssembler.assemble(_result(VerificationStatus.FAILED))

    assert captured.value.code is (
        CriticHandoffFailureCode.UPSTREAM_VERIFICATION_FAILED
    )


def test_assembler_rejects_status_that_disagrees_with_comparison() -> None:
    """A schema-valid but contradictory Agent 2 result cannot cascade trust."""

    inconsistent = _result(VerificationStatus.INCONSISTENT)
    contradictory = inconsistent.model_copy(
        update={"status": VerificationStatus.CONSISTENT}
    )

    with pytest.raises(CriticHandoffError) as captured:
        CriticInputAssembler.assemble(contradictory)

    assert captured.value.code is (
        CriticHandoffFailureCode.INVALID_COMPARISON_LINKAGE
    )


def test_assembler_rejects_comparison_targeting_an_unknown_source() -> None:
    """Both comparison references must target exactly Agent 2's normalized pair."""

    valid = _result(VerificationStatus.CONSISTENT)
    comparison = valid.comparisons[0].model_copy(
        update={"right_income_ref": "norm_" + "c" * 32}
    )
    mislinked = valid.model_copy(update={"comparisons": (comparison,)})

    with pytest.raises(CriticHandoffError) as captured:
        CriticInputAssembler.assemble(mislinked)

    assert captured.value.code is (
        CriticHandoffFailureCode.INVALID_COMPARISON_LINKAGE
    )


def test_assembler_rejects_provenance_targeting_another_document() -> None:
    """A normalized value cannot borrow provenance from another source."""

    valid = _result(VerificationStatus.CONSISTENT)
    pay_stub = valid.normalized_income[1]
    foreign_pointer = pay_stub.provenance[0].model_copy(
        update={"document_id": "doc_" + "d" * 32}
    )
    corrupted_pay_stub = pay_stub.model_copy(update={"provenance": (foreign_pointer,)})
    corrupted = valid.model_copy(
        update={"normalized_income": (valid.normalized_income[0], corrupted_pay_stub)}
    )

    with pytest.raises(CriticHandoffError) as captured:
        CriticInputAssembler.assemble(corrupted)

    assert captured.value.code is CriticHandoffFailureCode.INVALID_PROVENANCE


def test_assembler_rejects_duplicate_normalized_sources() -> None:
    """One normalized artifact cannot be presented as two independent sources."""

    valid = _result(VerificationStatus.CONSISTENT)
    duplicated = valid.model_copy(
        update={"normalized_income": (valid.normalized_income[0],) * 2}
    )

    with pytest.raises(CriticHandoffError) as captured:
        CriticInputAssembler.assemble(duplicated)

    assert captured.value.code is CriticHandoffFailureCode.INVALID_SOURCE_SET


def test_assembler_rejects_wrong_source_transformation_rule() -> None:
    """Document type and deterministic normalization rule must remain paired."""

    valid = _result(VerificationStatus.CONSISTENT)
    pay_stub = valid.normalized_income[1].model_copy(
        update={"transformation_rule": TransformationRule.W2_ANNUAL_TO_MONTHLY_V1}
    )
    corrupted = valid.model_copy(
        update={"normalized_income": (valid.normalized_income[0], pay_stub)}
    )

    with pytest.raises(CriticHandoffError) as captured:
        CriticInputAssembler.assemble(corrupted)

    assert captured.value.code is CriticHandoffFailureCode.INVALID_SOURCE_SET


def test_assembler_sanitizes_invalid_reduced_critic_contract() -> None:
    """A source valid for Agent 2 can still fail Agent 3's narrower year bounds."""

    valid = _result(VerificationStatus.CONSISTENT)
    pay_stub = valid.normalized_income[1].model_copy(update={"calendar_year": 1999})
    reduced_contract_failure = valid.model_copy(
        update={"normalized_income": (valid.normalized_income[0], pay_stub)}
    )

    with pytest.raises(CriticHandoffError) as captured:
        CriticInputAssembler.assemble(reduced_contract_failure)

    assert captured.value.code is CriticHandoffFailureCode.INVALID_CRITIC_REQUEST


def test_assembler_rejects_invalid_object_with_sanitized_error() -> None:
    """Invalid boundary input cannot leak object contents through its exception."""

    private_sentinel = "PRIVATE-AGENT2-RESULT-MUST-NOT-APPEAR"

    class InvalidResult:
        def model_dump(self, **_kwargs: object) -> dict[str, str]:
            return {"private": private_sentinel}

    with pytest.raises(CriticHandoffError) as captured:
        CriticInputAssembler.assemble(InvalidResult())  # type: ignore[arg-type]

    assert captured.value.code is CriticHandoffFailureCode.INVALID_INPUT
    rendered = "".join(traceback.format_exception(captured.value))
    assert private_sentinel not in rendered
