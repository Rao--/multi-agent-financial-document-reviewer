from __future__ import annotations

import copy
import json
import traceback
from decimal import Decimal

import pytest
from pydantic import ValidationError

from financial_reviewer.agents.agent1_extraction import (
    DeterministicExtractionError,
    ExtractionValidationError,
    classify_document,
    enforce_evidence_guard,
    extract_bank_statement_deterministically,
    extract_pay_stub_deterministically,
    extract_tax_form_deterministically,
    parse_extraction_proposal,
    resolve_proposal_evidence,
)
from financial_reviewer.local.observability import (
    new_correlation_id,
    new_idempotency_key,
    new_opaque_document_id,
)
from financial_reviewer.foundation.schemas import (
    BankStatementFieldName,
    DeterministicUnresolvedReason,
    DocumentType,
    ExtractionMetadata,
    PayStubFieldName,
    ReviewOutcome,
    SupportedMoneyProposal,
    SupportedMoneyField,
    TaxFormFieldName,
    UnsupportedField,
    WorkflowStatus,
    iter_extracted_fields,
    proposal_schema,
)
from tests.helpers import (
    mismatched_evidence_json,
    pay_stub_json,
    pay_stub_payload,
    unsupported_pay_stub_json,
)


def _resolve(document_text: str, raw: str):
    document_type, classification = classify_document(document_text)
    assert classification is not None
    proposal = parse_extraction_proposal(raw, document_type)
    return resolve_proposal_evidence(
        proposal,
        document_id=new_opaque_document_id(),
        document_text=document_text,
        classification=classification,
        model_name="qwen2.5:3b",
        attempt_count=1,
    )


def _extract_deterministically(document_text: str):
    """Classify and run the current pay-stub path without a model."""

    document_type, classification = classify_document(document_text)
    assert document_type is DocumentType.PAY_STUB
    assert classification is not None
    return extract_pay_stub_deterministically(
        document_id=new_opaque_document_id(),
        document_text=document_text,
        classification=classification,
    )


def test_deterministic_pay_stub_extraction_owns_values_and_provenance(
    synthetic_pay_stub_text: str,
) -> None:
    """The pay-stub slice extracts seven fields and records zero model attempts."""

    extraction = _extract_deterministically(synthetic_pay_stub_text)
    guarded = enforce_evidence_guard(extraction, synthetic_pay_stub_text)

    assert extraction.metadata.extraction_method == "deterministic_labels_v1"
    assert extraction.metadata.schema_version == "1.1"
    assert extraction.metadata.attempt_count == 0
    assert extraction.metadata.prompt_version is None
    assert extraction.metadata.model_provider is None
    assert extraction.metadata.model_name is None
    assert guarded.metadata.evidence_guard_passed is True
    assert guarded.metadata.extraction_method == "deterministic_labels_v1"
    assert guarded.fields.monthly_income.value == Decimal("6250.00")
    assert guarded.fields.pay_period_months.value == 12
    assert guarded.fields.pay_period_year.value == 2025
    for _, field in iter_extracted_fields(guarded):
        assert field.status == "supported"
        assert len(field.provenance) == 1
        assert field.provenance[0].confidence == 1.0


def test_deterministic_w2_extraction_owns_values_and_provenance(
    synthetic_w2_text: str,
) -> None:
    """W-2 annual income and all declared fields retain local source evidence."""

    document_type, classification = classify_document(synthetic_w2_text)
    assert document_type is DocumentType.TAX_FORM
    assert classification is not None

    extraction = extract_tax_form_deterministically(
        document_id=new_opaque_document_id(),
        document_text=synthetic_w2_text,
        classification=classification,
    )
    guarded = enforce_evidence_guard(extraction, synthetic_w2_text)

    assert guarded.fields.annual_wages.value == Decimal("75000.00")
    assert guarded.fields.federal_tax_withheld.value == Decimal("7500.00")
    assert guarded.fields.tax_year.value == 2025
    assert guarded.metadata.attempt_count == 0
    assert guarded.metadata.evidence_guard_passed is True
    assert all(
        field.status == "supported" and len(field.provenance) == 1
        for _, field in iter_extracted_fields(guarded)
    )


def test_deterministic_bank_statement_extraction_owns_values_and_provenance(
    synthetic_bank_statement_text: str,
) -> None:
    """The narrow bank slice extracts the declared monthly deposit total."""

    document_type, classification = classify_document(synthetic_bank_statement_text)
    assert document_type is DocumentType.BANK_STATEMENT
    assert classification is not None

    extraction = extract_bank_statement_deterministically(
        document_id=new_opaque_document_id(),
        document_text=synthetic_bank_statement_text,
        classification=classification,
    )
    guarded = enforce_evidence_guard(extraction, synthetic_bank_statement_text)

    assert guarded.fields.monthly_deposits.value == Decimal("6250.00")
    assert guarded.fields.statement_month.value == "2025-01"
    assert guarded.metadata.attempt_count == 0
    assert guarded.metadata.evidence_guard_passed is True
    assert all(
        field.status == "supported" and len(field.provenance) == 1
        for _, field in iter_extracted_fields(guarded)
    )


@pytest.mark.parametrize(
    (
        "fixture_name",
        "changed_line",
        "replacement",
        "field_name",
        "reason",
        "error_code",
        "extractor",
    ),
    [
        (
            "synthetic_w2_text",
            "Annual Wages: $75,000.00\n",
            "",
            TaxFormFieldName.ANNUAL_WAGES,
            DeterministicUnresolvedReason.MISSING_LABEL,
            "unsupported_required_field",
            extract_tax_form_deterministically,
        ),
        (
            "synthetic_w2_text",
            "Tax Year: 2025",
            "Tax Year: not-a-year",
            TaxFormFieldName.TAX_YEAR,
            DeterministicUnresolvedReason.INVALID_FORMAT,
            "schema_validation_failed",
            extract_tax_form_deterministically,
        ),
        (
            "synthetic_bank_statement_text",
            "Monthly Deposits: $6,250.00",
            "Monthly Deposits: $6,250.00\nMonthly Deposits: $6,250.00",
            BankStatementFieldName.MONTHLY_DEPOSITS,
            DeterministicUnresolvedReason.DUPLICATE_LABEL,
            "evidence_validation_failed",
            extract_bank_statement_deterministically,
        ),
        (
            "synthetic_bank_statement_text",
            "Monthly Deposits: $6,250.00",
            "Monthly Deposits: not-a-number",
            BankStatementFieldName.MONTHLY_DEPOSITS,
            DeterministicUnresolvedReason.INVALID_FORMAT,
            "schema_validation_failed",
            extract_bank_statement_deterministically,
        ),
    ],
)
def test_new_deterministic_extractors_fail_closed_with_sanitized_reasons(
    request: pytest.FixtureRequest,
    fixture_name: str,
    changed_line: str,
    replacement: str,
    field_name: BankStatementFieldName | TaxFormFieldName,
    reason: DeterministicUnresolvedReason,
    error_code: str,
    extractor,
) -> None:
    """W-2 and bank failures expose only field names and closed reason codes."""

    original = request.getfixturevalue(fixture_name)
    altered = original.replace(changed_line, replacement)
    _, classification = classify_document(altered)
    assert classification is not None

    with pytest.raises(DeterministicExtractionError) as captured:
        extractor(
            document_id=new_opaque_document_id(),
            document_text=altered,
            classification=classification,
        )

    assert captured.value.code == error_code
    assert [(item.field_name, item.reason) for item in captured.value.unresolved_fields] == [
        (field_name, reason)
    ]
    rendered = "".join(traceback.format_exception(captured.value))
    if replacement:
        assert replacement not in rendered
    assert "SYNTHETIC PERSON ALPHA" not in rendered


@pytest.mark.parametrize(
    ("changed_line", "replacement", "field_name", "reason", "error_code"),
    [
        (
            "Employee ID: SYN-EMP-0001\n",
            "",
            PayStubFieldName.EMPLOYEE_ID,
            DeterministicUnresolvedReason.MISSING_LABEL,
            "unsupported_required_field",
        ),
        (
            "Employee ID: SYN-EMP-0001",
            "Employee ID: SYN-EMP-0001\nEmployee ID: SYN-EMP-0001",
            PayStubFieldName.EMPLOYEE_ID,
            DeterministicUnresolvedReason.DUPLICATE_LABEL,
            "evidence_validation_failed",
        ),
        (
            "Employee ID: SYN-EMP-0001",
            "Employee ID: SYN-EMP-0001\nEmployee ID: SYN-EMP-9999",
            PayStubFieldName.EMPLOYEE_ID,
            DeterministicUnresolvedReason.CONFLICTING_VALUES,
            "evidence_validation_failed",
        ),
        (
            "Employee ID: SYN-EMP-0001",
            "Employee ID:",
            PayStubFieldName.EMPLOYEE_ID,
            DeterministicUnresolvedReason.EMPTY_VALUE,
            "schema_validation_failed",
        ),
        (
            "Monthly Income: $6,250.00",
            "Monthly Income: not-a-number",
            PayStubFieldName.MONTHLY_INCOME,
            DeterministicUnresolvedReason.INVALID_FORMAT,
            "schema_validation_failed",
        ),
    ],
)
def test_deterministic_unresolved_fields_are_reason_coded_without_values(
    synthetic_pay_stub_text: str,
    changed_line: str,
    replacement: str,
    field_name: PayStubFieldName,
    reason: DeterministicUnresolvedReason,
    error_code: str,
) -> None:
    altered = synthetic_pay_stub_text.replace(changed_line, replacement)

    with pytest.raises(DeterministicExtractionError) as captured:
        _extract_deterministically(altered)

    assert captured.value.code == error_code
    assert [(item.field_name, item.reason) for item in captured.value.unresolved_fields] == [
        (field_name, reason)
    ]
    rendered = "".join(traceback.format_exception(captured.value))
    assert "SYN-EMP-0001" not in rendered
    assert "not-a-number" not in rendered


def test_extraction_metadata_rejects_model_claims_on_deterministic_path() -> None:
    """Audit metadata cannot falsely claim both deterministic and model work."""

    with pytest.raises(ValidationError):
        ExtractionMetadata(
            extraction_method="deterministic_labels_v1",
            prompt_version="agent1-extraction-v1",
            model_provider="ollama",
            model_name="qwen2.5:3b",
            attempt_count=0,
        )


def test_money_proposal_pattern_preserves_rules_and_compiles_for_ollama() -> None:
    """Keep money validation strict without unsupported regex constructs."""

    source = {"line_number": 1, "quote": "synthetic", "confidence": 0.99}
    for value in ("0", "1", "6250.00", "10.5"):
        proposal = SupportedMoneyProposal(
            status="supported",
            value=value,
            source=source,
        )
        assert proposal.value == value

    for value in ("00", "01", "-1", "1.234"):
        with pytest.raises(ValidationError):
            SupportedMoneyProposal(
                status="supported",
                value=value,
                source=source,
            )

    schema = proposal_schema(DocumentType.PAY_STUB)
    pattern = schema["$defs"]["SupportedMoneyProposal"]["properties"]["value"][
        "pattern"
    ]
    assert pattern == r"^(0|[1-9][0-9]*)(\.[0-9]{1,2})?$"
    assert "(?:" not in pattern


def test_valid_proposal_becomes_typed_extraction_with_provenance(
    synthetic_pay_stub_text: str,
) -> None:
    extraction = _resolve(synthetic_pay_stub_text, pay_stub_json(synthetic_pay_stub_text))
    guarded = enforce_evidence_guard(extraction, synthetic_pay_stub_text)

    assert extraction.document_type == "pay_stub"
    assert extraction.metadata.evidence_guard_passed is False
    assert guarded.metadata.evidence_guard_passed is True
    assert guarded.metadata.evidence_guard_version == "evidence-guard-v1"
    assert isinstance(extraction.fields.monthly_income, SupportedMoneyField)
    assert extraction.fields.monthly_income.value == Decimal("6250.00")
    for _, field in iter_extracted_fields(extraction):
        assert field.status == "supported"
        assert field.provenance
        assert field.provenance[0].evidence_sha256


@pytest.mark.parametrize("mutation", ["extra", "missing", "coerced_integer"])
def test_schema_gate_rejects_extra_missing_and_coerced_fields(
    synthetic_pay_stub_text: str,
    mutation: str,
) -> None:
    payload = copy.deepcopy(pay_stub_payload(synthetic_pay_stub_text))
    if mutation == "extra":
        payload["fields"]["raw_document"] = "must never be accepted"
    elif mutation == "missing":
        del payload["fields"]["employer_ein"]
    else:
        payload["fields"]["pay_period_months"]["value"] = "12"

    with pytest.raises(ExtractionValidationError) as captured:
        parse_extraction_proposal(json.dumps(payload), DocumentType.PAY_STUB)
    assert captured.value.code == "invalid_model_output"
    assert "must never be accepted" not in str(captured.value)


def test_schema_gate_rejects_duplicate_json_keys() -> None:
    raw = '{"document_type":"pay_stub","document_type":"bank_statement","fields":{}}'
    with pytest.raises(ExtractionValidationError):
        parse_extraction_proposal(raw, DocumentType.PAY_STUB)


def test_fabricated_source_quote_cannot_be_resolved(
    synthetic_pay_stub_text: str,
) -> None:
    with pytest.raises(ExtractionValidationError) as captured:
        _resolve(
            synthetic_pay_stub_text,
            mismatched_evidence_json(synthetic_pay_stub_text),
        )
    assert captured.value.code == "evidence_validation_failed"


def test_partial_string_match_cannot_be_bound_as_evidence(
    synthetic_pay_stub_text: str,
) -> None:
    payload = pay_stub_payload(synthetic_pay_stub_text)
    payload["fields"]["employee_name"]["value"] = "PERSON ALPHA"
    with pytest.raises(ExtractionValidationError) as captured:
        _resolve(synthetic_pay_stub_text, json.dumps(payload))
    assert captured.value.code == "evidence_validation_failed"


def test_identifier_case_change_cannot_be_bound_as_exact_evidence(
    synthetic_pay_stub_text: str,
) -> None:
    payload = pay_stub_payload(synthetic_pay_stub_text)
    payload["fields"]["employee_id"]["value"] = "syn-emp-0001"
    with pytest.raises(ExtractionValidationError) as captured:
        _resolve(synthetic_pay_stub_text, json.dumps(payload))
    assert captured.value.code == "evidence_validation_failed"


def test_low_confidence_source_routes_to_validation_failure(
    synthetic_pay_stub_text: str,
) -> None:
    payload = pay_stub_payload(synthetic_pay_stub_text)
    payload["fields"]["employee_name"]["source"]["confidence"] = 0.10
    with pytest.raises(ExtractionValidationError) as captured:
        _resolve(synthetic_pay_stub_text, json.dumps(payload))
    assert captured.value.code == "evidence_validation_failed"


def test_sanitized_extraction_error_suppresses_sensitive_cause() -> None:
    sentinel = "SENSITIVE-MODEL-OUTPUT-MUST-NOT-SURVIVE"
    raw = json.dumps({"unexpected": sentinel})
    try:
        parse_extraction_proposal(raw, DocumentType.PAY_STUB)
    except ExtractionValidationError as error:
        rendered = "".join(traceback.format_exception(error))
        assert error.__suppress_context__ is True
        assert sentinel not in rendered
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("invalid output unexpectedly passed")


def test_explicit_unsupported_field_is_typed_but_blocks_release(
    synthetic_pay_stub_text: str,
) -> None:
    extraction = _resolve(
        synthetic_pay_stub_text,
        unsupported_pay_stub_json(synthetic_pay_stub_text),
    )
    assert isinstance(extraction.fields.employee_id, UnsupportedField)
    assert extraction.fields.employee_id.value is None
    assert extraction.fields.employee_id.provenance == ()
    assert extraction.fields.employee_id.reason.value == "not_present"

    with pytest.raises(ExtractionValidationError) as captured:
        enforce_evidence_guard(extraction, synthetic_pay_stub_text)
    assert captured.value.code == "unsupported_required_field"


def test_evidence_guard_detects_local_source_tampering(
    synthetic_pay_stub_text: str,
) -> None:
    extraction = _resolve(synthetic_pay_stub_text, pay_stub_json(synthetic_pay_stub_text))
    tampered = synthetic_pay_stub_text.replace("$6,250.00", "$9,999.99")
    with pytest.raises(ExtractionValidationError) as captured:
        enforce_evidence_guard(extraction, tampered)
    assert captured.value.code == "evidence_validation_failed"


def test_release_envelope_rejects_pre_guard_candidate_and_type_mismatch(
    synthetic_pay_stub_text: str,
) -> None:
    candidate = _resolve(
        synthetic_pay_stub_text,
        pay_stub_json(synthetic_pay_stub_text),
    )
    common = {
        "correlation_id": new_correlation_id(),
        "idempotency_key": new_idempotency_key(),
        "status": WorkflowStatus.RELEASED,
        "validated_extraction": candidate,
        "failure_code": None,
        "human_review_required": False,
    }
    with pytest.raises(ValidationError):
        ReviewOutcome(document_type=DocumentType.PAY_STUB, **common)

    guarded = enforce_evidence_guard(candidate, synthetic_pay_stub_text)
    with pytest.raises(ValidationError):
        ReviewOutcome(
            document_type=DocumentType.BANK_STATEMENT,
            **{**common, "validated_extraction": guarded},
        )


@pytest.mark.parametrize(
    ("document_text", "payload"),
    [
        (
            "\n".join(
                [
                    "SYNTHETIC TEST DOCUMENT - NOT REAL",
                    "BANK STATEMENT",
                    "Account Holder: SYNTHETIC PERSON BETA",
                    "Account Number: SYN-ACCT-0002",
                    "Bank Name: SYNTHETIC COMMUNITY BANK",
                    "Statement Month: January 2026",
                    "Monthly Deposits: $7,500.00",
                ]
            ),
            {
                "document_type": "bank_statement",
                "fields": {
                    "account_holder_name": ("SYNTHETIC PERSON BETA", 3, "string"),
                    "account_number": ("SYN-ACCT-0002", 4, "string"),
                    "bank_name": ("SYNTHETIC COMMUNITY BANK", 5, "string"),
                    "statement_month": ("January 2026", 6, "string"),
                    "monthly_deposits": ("7500.00", 7, "money"),
                },
            },
        ),
        (
            "\n".join(
                [
                    "SYNTHETIC TEST DOCUMENT - NOT REAL",
                    "W-2 TAX FORM",
                    "Employee Name: SYNTHETIC PERSON GAMMA",
                    "Employee SSN: 000-00-0003",
                    "Employee Address: 100 SYNTHETIC WAY",
                    "Employer Name: SYNTHETIC TAX LABS LLC",
                    "Employer EIN: 00-0000003",
                    "Annual Wages: $90,000.00",
                    "Federal Tax Withheld: $12,000.00",
                    "Tax Year: 2026",
                ]
            ),
            {
                "document_type": "tax_form",
                "fields": {
                    "employee_name": ("SYNTHETIC PERSON GAMMA", 3, "string"),
                    "employee_ssn": ("000-00-0003", 4, "string"),
                    "employee_address": ("100 SYNTHETIC WAY", 5, "string"),
                    "employer_name": ("SYNTHETIC TAX LABS LLC", 6, "string"),
                    "employer_ein": ("00-0000003", 7, "string"),
                    "annual_wages": ("90000.00", 8, "money"),
                    "federal_tax_withheld": ("12000.00", 9, "money"),
                    "tax_year": (2026, 10, "integer"),
                },
            },
        ),
    ],
)
def test_bank_and_tax_happy_paths_have_guarded_provenance(
    document_text: str,
    payload: dict[str, object],
) -> None:
    lines = document_text.splitlines()
    fields = payload["fields"]
    assert isinstance(fields, dict)
    proposal_fields: dict[str, object] = {}
    for name, specification in fields.items():
        value, line_number, _kind = specification
        proposal_fields[name] = {
            "status": "supported",
            "value": value,
            "source": {
                "line_number": line_number,
                "quote": lines[line_number - 1],
                "confidence": 0.99,
            },
        }
    raw = json.dumps(
        {
            "document_type": payload["document_type"],
            "fields": proposal_fields,
        }
    )
    extraction = _resolve(document_text, raw)
    guarded = enforce_evidence_guard(extraction, document_text)
    assert guarded.metadata.evidence_guard_passed is True
    assert all(field.provenance for _, field in iter_extracted_fields(guarded))
