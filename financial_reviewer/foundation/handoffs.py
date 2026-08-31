"""Build deterministic, fail-closed contracts between reviewer components.

Why this file exists:
    Adjacent agents use different typed contracts. The workflow needs non-agent
    boundaries that translate validated artifacts without asking a model to
    reinterpret financial values or blindly trust an upstream result.

What this file owns:
    The pay-stub/W-2 ``IncomeVerificationRequest`` assembly rule, the amount-free
    ``CriticReviewRequest`` assembly rule, input revalidation, artifact-linkage
    checks, random evidence references, and sanitized rejection codes.

What it does not own:
    It does not extract documents, call the Verification Agent, execute tools,
    normalize money, compare income, route LangGraph, or make a final decision.
    Bank-statement payroll evidence remains a later approved increment.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from financial_reviewer.agents.agent2_verification import (
    ComparisonResult,
    IncomeBasis,
    IncomeEvidence,
    IncomePeriod,
    IncomeVerificationResult,
    IncomeVerificationRequest,
    TransformationRule,
    VerificationReason,
    VerificationStatus,
)
from financial_reviewer.agents.agent3_critic import (
    CriticComparisonSummary,
    CriticReviewRequest,
    CriticSourceSummary,
)
from financial_reviewer.foundation.schemas import (
    BankStatementExtraction,
    DocumentType,
    PayStubExtraction,
    SourceProvenance,
    SupportedIntegerField,
    SupportedMoneyField,
    TaxFormExtraction,
    ValidatedExtraction,
)


# Revalidate constructed or copied extraction objects against the complete
# document-discriminated Agent 1 contract before reading any private field.
_VALIDATED_EXTRACTION_ADAPTER = TypeAdapter(ValidatedExtraction)


class VerificationHandoffFailureCode(str, Enum):
    """Closed, PII-free reasons the Agent 1 → Agent 2 handoff can stop."""

    INVALID_INPUT = "invalid_input"
    INVALID_EXTRACTION_COUNT = "invalid_extraction_count"
    DUPLICATE_DOCUMENT = "duplicate_document"
    UNGUARDED_EXTRACTION = "unguarded_extraction"
    UNSUPPORTED_DOCUMENT_COMBINATION = "unsupported_document_combination"
    REQUIRED_INCOME_EVIDENCE_INVALID = "required_income_evidence_invalid"


class VerificationHandoffError(ValueError):
    """Sanitized handoff rejection that never retains source values."""

    def __init__(self, code: VerificationHandoffFailureCode) -> None:
        """Expose only one allowlisted code to workflow, audit, and tests."""

        self.code = code
        super().__init__(code.value)


class CriticHandoffFailureCode(str, Enum):
    """Closed reasons the deterministic Agent 2 → Agent 3 boundary can stop."""

    INVALID_INPUT = "invalid_input"
    UPSTREAM_VERIFICATION_FAILED = "upstream_verification_failed"
    INVALID_SOURCE_SET = "invalid_source_set"
    INVALID_PROVENANCE = "invalid_provenance"
    INVALID_COMPARISON_LINKAGE = "invalid_comparison_linkage"
    INVALID_CRITIC_REQUEST = "invalid_critic_request"


class CriticHandoffError(ValueError):
    """Sanitized Critic handoff rejection that never retains private artifacts."""

    def __init__(self, code: CriticHandoffFailureCode) -> None:
        """Expose only an allowlisted boundary code to callers and diagnostics."""

        self.code = code
        super().__init__(code.value)


class VerificationInputAssembler:
    """Translate two guarded Agent 1 extractions into Agent 2's request.

    The first slice accepts exactly one pay stub and one W-2. Input order does
    not affect output order. A bank statement is deliberately rejected because
    its monthly-deposit total is not verified payroll income.
    """

    __slots__ = ()

    @staticmethod
    def assemble(
        extractions: Sequence[ValidatedExtraction],
    ) -> IncomeVerificationRequest:
        """Return a typed pay-stub/W-2 request or fail before Agent 2 runs."""

        try:
            extraction_count = len(extractions)
        except (AttributeError, TypeError):
            raise VerificationHandoffError(
                VerificationHandoffFailureCode.INVALID_INPUT
            ) from None
        if isinstance(extractions, (str, bytes)) or extraction_count != 2:
            raise VerificationHandoffError(
                VerificationHandoffFailureCode.INVALID_EXTRACTION_COUNT
            )

        validated = tuple(
            VerificationInputAssembler._revalidate_extraction(item)
            for item in extractions
        )
        if len({item.document_id for item in validated}) != 2:
            raise VerificationHandoffError(
                VerificationHandoffFailureCode.DUPLICATE_DOCUMENT
            )
        if any(not item.metadata.evidence_guard_passed for item in validated):
            raise VerificationHandoffError(
                VerificationHandoffFailureCode.UNGUARDED_EXTRACTION
            )

        pay_stubs = [item for item in validated if isinstance(item, PayStubExtraction)]
        tax_forms = [item for item in validated if isinstance(item, TaxFormExtraction)]
        bank_statements = [
            item for item in validated if isinstance(item, BankStatementExtraction)
        ]
        if bank_statements or len(pay_stubs) != 1 or len(tax_forms) != 1:
            raise VerificationHandoffError(
                VerificationHandoffFailureCode.UNSUPPORTED_DOCUMENT_COMBINATION
            )

        pay_stub = pay_stubs[0]
        tax_form = tax_forms[0]
        monthly_income = pay_stub.fields.monthly_income
        pay_period_year = pay_stub.fields.pay_period_year
        annual_wages = tax_form.fields.annual_wages
        tax_year = tax_form.fields.tax_year
        if (
            not isinstance(monthly_income, SupportedMoneyField)
            or not isinstance(pay_period_year, SupportedIntegerField)
            or not isinstance(annual_wages, SupportedMoneyField)
            or not isinstance(tax_year, SupportedIntegerField)
        ):
            raise VerificationHandoffError(
                VerificationHandoffFailureCode.REQUIRED_INCOME_EVIDENCE_INVALID
            )

        VerificationInputAssembler._verify_provenance(
            pay_stub.document_id,
            monthly_income.provenance,
        )
        VerificationInputAssembler._verify_provenance(
            pay_stub.document_id,
            pay_period_year.provenance,
        )
        VerificationInputAssembler._verify_provenance(
            tax_form.document_id,
            annual_wages.provenance,
        )
        VerificationInputAssembler._verify_provenance(
            tax_form.document_id,
            tax_year.provenance,
        )

        try:
            return IncomeVerificationRequest(
                evidence=(
                    IncomeEvidence(
                        evidence_ref=VerificationInputAssembler._new_evidence_ref(),
                        document_ref=pay_stub.document_id,
                        document_type=DocumentType.PAY_STUB,
                        amount=monthly_income.value,
                        period=IncomePeriod.MONTHLY,
                        # The synthetic ``Monthly Income`` contract is gross.
                        income_basis=IncomeBasis.GROSS,
                        calendar_year=pay_period_year.value,
                        provenance=monthly_income.provenance,
                    ),
                    IncomeEvidence(
                        evidence_ref=VerificationInputAssembler._new_evidence_ref(),
                        document_ref=tax_form.document_id,
                        document_type=DocumentType.TAX_FORM,
                        amount=annual_wages.value,
                        period=IncomePeriod.ANNUAL,
                        # ``Annual Wages`` is defined as gross in this synthetic slice.
                        income_basis=IncomeBasis.GROSS,
                        calendar_year=tax_year.value,
                        provenance=annual_wages.provenance,
                    ),
                )
            )
        except (TypeError, ValueError, ValidationError):
            raise VerificationHandoffError(
                VerificationHandoffFailureCode.REQUIRED_INCOME_EVIDENCE_INVALID
            ) from None

    @staticmethod
    def _revalidate_extraction(item: object) -> ValidatedExtraction:
        """Recreate one strict Agent 1 extraction without retaining bad input."""

        try:
            payload = item.model_dump(mode="python", warnings="none")  # type: ignore[attr-defined]
            return _VALIDATED_EXTRACTION_ADAPTER.validate_python(payload)
        except (AttributeError, TypeError, ValueError, ValidationError):
            raise VerificationHandoffError(
                VerificationHandoffFailureCode.INVALID_INPUT
            ) from None

    @staticmethod
    def _verify_provenance(
        document_id: str,
        provenance: tuple[SourceProvenance, ...],
    ) -> None:
        """Require selected evidence pointers to target their source document."""

        if not provenance or any(item.document_id != document_id for item in provenance):
            raise VerificationHandoffError(
                VerificationHandoffFailureCode.REQUIRED_INCOME_EVIDENCE_INVALID
            )

    @staticmethod
    def _new_evidence_ref() -> str:
        """Create a random local reference containing no business identifier."""

        return f"evidence_{uuid4().hex}"


class CriticInputAssembler:
    """Translate one private Agent 2 result into Agent 3's amount-free request.

    This boundary verifies source provenance and comparison linkage but does not
    recalculate income. Financial amounts, opaque references, document IDs, and
    provenance contents are deliberately omitted from the returned contract.
    """

    __slots__ = ()

    _COMPARISON_STATUSES = {
        VerificationStatus.CONSISTENT,
        VerificationStatus.INCONSISTENT,
        VerificationStatus.NOT_COMPARABLE,
    }
    _EXPECTED_TRANSFORMATION = {
        DocumentType.PAY_STUB: TransformationRule.PAYSTUB_MONTHLY_V1,
        DocumentType.TAX_FORM: TransformationRule.W2_ANNUAL_TO_MONTHLY_V1,
    }
    _EXPECTED_COMPARISON = {
        VerificationStatus.CONSISTENT: (
            ComparisonResult.CONSISTENT,
            VerificationReason.EXACT_MATCH,
        ),
        VerificationStatus.INCONSISTENT: (
            ComparisonResult.INCONSISTENT,
            VerificationReason.INCOME_VALUES_INCONSISTENT,
        ),
        VerificationStatus.NOT_COMPARABLE: (
            ComparisonResult.NOT_COMPARABLE,
            None,
        ),
    }
    _NOT_COMPARABLE_REASONS = {
        VerificationReason.INCOME_BASIS_NOT_COMPARABLE,
        VerificationReason.INCOME_PERIOD_NOT_COMPARABLE,
    }

    @staticmethod
    def assemble(result: IncomeVerificationResult) -> CriticReviewRequest:
        """Return a safe Critic request or stop before Agent 3 can be invoked."""

        validated = CriticInputAssembler._revalidate_result(result)
        if validated.status is VerificationStatus.FAILED:
            raise CriticHandoffError(
                CriticHandoffFailureCode.UPSTREAM_VERIFICATION_FAILED
            )

        CriticInputAssembler._validate_source_set(validated)
        comparison = CriticInputAssembler._comparison_summary(validated)
        try:
            sources = tuple(
                CriticSourceSummary(
                    document_type=item.document_type,
                    income_basis=item.income_basis,
                    calendar_year=item.calendar_year,
                    transformation_rule=item.transformation_rule,
                    provenance_pointer_count=len(item.provenance),
                )
                for item in sorted(
                    validated.normalized_income,
                    key=lambda source: (
                        0 if source.document_type is DocumentType.PAY_STUB else 1
                    ),
                )
            )
            return CriticReviewRequest(
                verification_status=validated.status,
                evidence_complete=validated.evidence_complete,
                sources=sources,
                comparison=comparison,
                verification_failure_code=None,
                tool_call_count=validated.tool_call_count,
                invalid_decision_count=validated.invalid_decision_count,
            )
        except (TypeError, ValueError, ValidationError):
            raise CriticHandoffError(
                CriticHandoffFailureCode.INVALID_CRITIC_REQUEST
            ) from None

    @staticmethod
    def _revalidate_result(result: object) -> IncomeVerificationResult:
        """Defeat constructed or mutated Agent 2 objects at the trust boundary."""

        try:
            payload = result.model_dump(mode="python", warnings="none")  # type: ignore[attr-defined]
            return IncomeVerificationResult.model_validate(payload)
        except (AttributeError, TypeError, ValueError, ValidationError):
            raise CriticHandoffError(CriticHandoffFailureCode.INVALID_INPUT) from None

    @staticmethod
    def _validate_source_set(result: IncomeVerificationResult) -> None:
        """Validate source identity, transformation, and provenance relationships."""

        sources = result.normalized_income
        if len(sources) > 2:
            raise CriticHandoffError(CriticHandoffFailureCode.INVALID_SOURCE_SET)
        if result.status in CriticInputAssembler._COMPARISON_STATUSES and (
            not result.evidence_complete
            or len(sources) != 2
            or {source.document_type for source in sources}
            != {DocumentType.PAY_STUB, DocumentType.TAX_FORM}
        ):
            raise CriticHandoffError(CriticHandoffFailureCode.INVALID_SOURCE_SET)
        if result.status is VerificationStatus.INSUFFICIENT_EVIDENCE and (
            result.evidence_complete
            or result.unsupported_reasons
            != (VerificationReason.UNSUPPORTED_DOCUMENT_COMBINATION,)
        ):
            raise CriticHandoffError(CriticHandoffFailureCode.INVALID_SOURCE_SET)

        for attribute in ("normalized_ref", "document_ref", "source_evidence_ref"):
            values = [getattr(source, attribute) for source in sources]
            if len(values) != len(set(values)):
                raise CriticHandoffError(CriticHandoffFailureCode.INVALID_SOURCE_SET)
        for source in sources:
            expected_rule = CriticInputAssembler._EXPECTED_TRANSFORMATION.get(
                source.document_type
            )
            if expected_rule is None or source.transformation_rule is not expected_rule:
                raise CriticHandoffError(CriticHandoffFailureCode.INVALID_SOURCE_SET)
            if not source.provenance or any(
                pointer.document_id != source.document_ref
                for pointer in source.provenance
            ):
                raise CriticHandoffError(CriticHandoffFailureCode.INVALID_PROVENANCE)

    @staticmethod
    def _comparison_summary(
        result: IncomeVerificationResult,
    ) -> CriticComparisonSummary | None:
        """Verify that one comparison targets exactly the normalized source set."""

        if result.status is VerificationStatus.INSUFFICIENT_EVIDENCE:
            if result.comparisons:
                raise CriticHandoffError(
                    CriticHandoffFailureCode.INVALID_COMPARISON_LINKAGE
                )
            return None
        if result.status not in CriticInputAssembler._COMPARISON_STATUSES:
            raise CriticHandoffError(
                CriticHandoffFailureCode.INVALID_COMPARISON_LINKAGE
            )

        if len(result.comparisons) != 1:
            raise CriticHandoffError(
                CriticHandoffFailureCode.INVALID_COMPARISON_LINKAGE
            )
        comparison = result.comparisons[0]
        source_refs = {source.normalized_ref for source in result.normalized_income}
        comparison_refs = {
            comparison.left_income_ref,
            comparison.right_income_ref,
        }
        if len(comparison_refs) != 2 or comparison_refs != source_refs:
            raise CriticHandoffError(
                CriticHandoffFailureCode.INVALID_COMPARISON_LINKAGE
            )

        expected_result, expected_reason = CriticInputAssembler._EXPECTED_COMPARISON[
            result.status
        ]
        if comparison.result is not expected_result:
            raise CriticHandoffError(
                CriticHandoffFailureCode.INVALID_COMPARISON_LINKAGE
            )
        if expected_reason is not None:
            reason_valid = comparison.reason_code is expected_reason
        else:
            reason_valid = (
                comparison.reason_code
                in CriticInputAssembler._NOT_COMPARABLE_REASONS
            )
        expected_unsupported_reasons = (
            (comparison.reason_code,)
            if result.status is VerificationStatus.NOT_COMPARABLE
            else ()
        )
        if (
            not reason_valid
            or result.unsupported_reasons != expected_unsupported_reasons
        ):
            raise CriticHandoffError(
                CriticHandoffFailureCode.INVALID_COMPARISON_LINKAGE
            )
        try:
            return CriticComparisonSummary(
                result=comparison.result,
                reason_code=comparison.reason_code,
                policy_id=comparison.policy_id,
            )
        except (TypeError, ValueError, ValidationError):
            raise CriticHandoffError(
                CriticHandoffFailureCode.INVALID_COMPARISON_LINKAGE
            ) from None


__all__ = [
    "CriticHandoffError",
    "CriticHandoffFailureCode",
    "CriticInputAssembler",
    "VerificationHandoffError",
    "VerificationHandoffFailureCode",
    "VerificationInputAssembler",
]
