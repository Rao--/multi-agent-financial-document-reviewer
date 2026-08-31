"""Implement Agent 1's document-specific extraction and evidence rules.

Why this file exists:
    Keep extraction-domain logic independent of LangGraph orchestration and the
    local model transport.  The workflow decides *when* these functions run;
    this module performs deterministic extraction for the approved synthetic
    document templates and retains the
    guarded local-model proposal contracts for a later selective-LLM route.

What this file owns:
    Deterministic synthetic-document classification, document-specific field
    extraction, field-level unresolved reasons, code-owned provenance, strict
    parsing of future model proposals, and the final evidence guard.

What it does not own:
    It does not call Ollama, retry, store documents, emit audit records, route
    graph nodes, or release an outcome.  Those responsibilities remain in
    ``local.model``, ``local.storage``, and ``workflow``.
"""

from __future__ import annotations

import hashlib
import json
import re
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import ValidationError

from financial_reviewer.foundation.intake import SYNTHETIC_MARKER
from financial_reviewer.foundation.schemas import (
    BankStatementExtraction,
    BankStatementFieldName,
    BankStatementFields,
    BankStatementProposal,
    DeterministicUnresolvedField,
    DeterministicUnresolvedReason,
    DocumentClassification,
    DocumentType,
    ExtractionMetadata,
    ExtractionProposal,
    PayStubExtraction,
    PayStubFieldName,
    PayStubFields,
    PayStubProposal,
    PROPOSAL_MODELS,
    SourceProvenance,
    SupportedIntegerField,
    SupportedIntegerProposal,
    SupportedMoneyField,
    SupportedMoneyProposal,
    SupportedStringField,
    SupportedStringProposal,
    TaxFormExtraction,
    TaxFormFieldName,
    TaxFormFields,
    TaxFormProposal,
    UnsupportedField,
    UnsupportedProposal,
    ValidatedExtraction,
    iter_extracted_fields,
)


# Parser resource guard for the UTF-8 JSON returned by the local model. This is
# a byte-size ceiling, not a model token limit or an extraction-quality score.
MAX_MODEL_OUTPUT_BYTES = 64 * 1024
# Policy gate applied to each model-supplied provenance claim. A score below
# 0.80 becomes unsupported; a score at/above it must still pass exact evidence
# matching. The number is a configurable policy assumption, not proof of truth.
MIN_SOURCE_CONFIDENCE = 0.80

# Deterministic type markers accepted in the synthetic Milestone 1 documents.
# Unknown headers fail closed instead of being guessed by the model.
_DOCUMENT_HEADERS = {
    "BANK STATEMENT": DocumentType.BANK_STATEMENT,
    "PAY STUB": DocumentType.PAY_STUB,
    "W2 TAX FORM": DocumentType.TAX_FORM,
    "W-2 TAX FORM": DocumentType.TAX_FORM,
    "TAX FORM": DocumentType.TAX_FORM,
}

# Per-document allowlist mapping schema field names to literal source labels.
# It keeps evidence lookup deterministic and prevents free-form label guessing.
_FIELD_LABELS: dict[DocumentType, dict[str, str]] = {
    DocumentType.BANK_STATEMENT: {
        "account_holder_name": "Account Holder:",
        "account_number": "Account Number:",
        "bank_name": "Bank Name:",
        "statement_month": "Statement Month:",
        "monthly_deposits": "Monthly Deposits:",
    },
    DocumentType.PAY_STUB: {
        "employee_name": "Employee Name:",
        "employee_id": "Employee ID:",
        "employer_name": "Employer Name:",
        "employer_ein": "Employer EIN:",
        "monthly_income": "Monthly Income:",
        "pay_period_months": "Pay Period:",
        "pay_period_year": "Pay Period Year:",
    },
    DocumentType.TAX_FORM: {
        "employee_name": "Employee Name:",
        "employee_ssn": "Employee SSN:",
        "employee_address": "Employee Address:",
        "employer_name": "Employer Name:",
        "employer_ein": "Employer EIN:",
        "annual_wages": "Annual Wages:",
        "federal_tax_withheld": "Federal Tax Withheld:",
        "tax_year": "Tax Year:",
    },
}

# Select the validated field container after deterministic evidence resolution.
_FINAL_FIELD_MODELS = {
    DocumentType.BANK_STATEMENT: BankStatementFields,
    DocumentType.PAY_STUB: PayStubFields,
    DocumentType.TAX_FORM: TaxFormFields,
}

# Select the release-candidate envelope matching the classified document type.
_FINAL_EXTRACTION_MODELS = {
    DocumentType.BANK_STATEMENT: BankStatementExtraction,
    DocumentType.PAY_STUB: PayStubExtraction,
    DocumentType.TAX_FORM: TaxFormExtraction,
}

# Closed field order for each deterministic template.  Iterating enums rather
# than document keys prevents an input document from introducing field names.
_DETERMINISTIC_FIELD_NAMES = {
    DocumentType.BANK_STATEMENT: tuple(BankStatementFieldName),
    DocumentType.PAY_STUB: tuple(PayStubFieldName),
    DocumentType.TAX_FORM: tuple(TaxFormFieldName),
}

# Parsing behavior is code-owned per schema field.  The extractor never asks a
# model to decide whether a source value is text, money, or an integer.
_DETERMINISTIC_FIELD_KINDS = {
    DocumentType.BANK_STATEMENT: {
        "account_holder_name": "string",
        "account_number": "string",
        "bank_name": "string",
        "statement_month": "string",
        "monthly_deposits": "money",
    },
    DocumentType.PAY_STUB: {
        "employee_name": "string",
        "employee_id": "string",
        "employer_name": "string",
        "employer_ein": "string",
        "monthly_income": "money",
        "pay_period_months": "integer",
        "pay_period_year": "integer",
    },
    DocumentType.TAX_FORM: {
        "employee_name": "string",
        "employee_ssn": "string",
        "employee_address": "string",
        "employer_name": "string",
        "employer_ein": "string",
        "annual_wages": "money",
        "federal_tax_withheld": "money",
        "tax_year": "integer",
    },
}


class ExtractionValidationError(ValueError):
    """Sanitized deterministic validation failure."""

    def __init__(self, code: str) -> None:
        """Retain only a closed failure code, never rejected document data."""

        self.code = code
        super().__init__(code)


class DeterministicExtractionError(ExtractionValidationError):
    """Safe deterministic failure carrying only field names and reason codes."""

    def __init__(
        self,
        code: str,
        unresolved_fields: tuple[DeterministicUnresolvedField, ...],
    ) -> None:
        """Retain no source values, lines, quotes, or document contents."""

        self.unresolved_fields = tuple(
            DeterministicUnresolvedField.model_validate(
                item.model_dump(mode="python", warnings="none")
            )
            for item in unresolved_fields
        )
        super().__init__(code)


def classify_document(
    document_text: str,
) -> tuple[DocumentType, DocumentClassification | None]:
    """Classify a marked synthetic document from its explicit type header."""

    nonempty_lines = [line.strip() for line in document_text.splitlines() if line.strip()]
    if len(nonempty_lines) < 2 or nonempty_lines[0] != SYNTHETIC_MARKER:
        return DocumentType.UNKNOWN, None
    header = nonempty_lines[1]
    document_type = _DOCUMENT_HEADERS.get(header.upper(), DocumentType.UNKNOWN)
    if document_type is DocumentType.UNKNOWN:
        return document_type, None
    classification = DocumentClassification(
        document_type=document_type,
        method="deterministic_header_v1",
        confidence=1.0,
        header_evidence_sha256=hashlib.sha256(header.encode("utf-8")).hexdigest(),
    )
    return document_type, classification


def extract_pay_stub_deterministically(
    *,
    document_id: str,
    document_text: str,
    classification: DocumentClassification,
) -> PayStubExtraction:
    """Extract one known pay-stub template without invoking a model.

    This public, document-specific entry point keeps workflow dispatch and unit
    tests explicit while delegating the shared label/provenance algorithm.
    """

    extraction = _extract_document_deterministically(
        document_id=document_id,
        document_text=document_text,
        classification=classification,
        expected_document_type=DocumentType.PAY_STUB,
    )
    if not isinstance(extraction, PayStubExtraction):
        raise ExtractionValidationError("schema_validation_failed")
    return extraction


def extract_bank_statement_deterministically(
    *,
    document_id: str,
    document_text: str,
    classification: DocumentClassification,
) -> BankStatementExtraction:
    """Extract the approved synthetic bank-statement template locally."""

    extraction = _extract_document_deterministically(
        document_id=document_id,
        document_text=document_text,
        classification=classification,
        expected_document_type=DocumentType.BANK_STATEMENT,
    )
    if not isinstance(extraction, BankStatementExtraction):
        raise ExtractionValidationError("schema_validation_failed")
    return extraction


def extract_tax_form_deterministically(
    *,
    document_id: str,
    document_text: str,
    classification: DocumentClassification,
) -> TaxFormExtraction:
    """Extract the approved synthetic W-2 template locally."""

    extraction = _extract_document_deterministically(
        document_id=document_id,
        document_text=document_text,
        classification=classification,
        expected_document_type=DocumentType.TAX_FORM,
    )
    if not isinstance(extraction, TaxFormExtraction):
        raise ExtractionValidationError("schema_validation_failed")
    return extraction


def _extract_document_deterministically(
    *,
    document_id: str,
    document_text: str,
    classification: DocumentClassification,
    expected_document_type: DocumentType,
) -> ValidatedExtraction:
    """Run the shared exact-label parser for one allowlisted document type.

    Each required label must occur exactly once. Code parses the value from the
    matched line and creates provenance from that same local line. Missing,
    duplicate, conflicting, empty, or malformed fields produce only PII-free
    field names and closed reason codes; no partial extraction is returned.
    """

    if classification.document_type is not expected_document_type:
        raise DeterministicExtractionError("unsupported_document_type", ())
    lines = document_text.splitlines()
    labels = _FIELD_LABELS[expected_document_type]
    field_names = _DETERMINISTIC_FIELD_NAMES[expected_document_type]
    field_kinds = _DETERMINISTIC_FIELD_KINDS[expected_document_type]
    resolved: dict[str, object] = {}
    unresolved: list[DeterministicUnresolvedField] = []

    for field_name in field_names:
        label = labels[field_name.value]
        matches = [
            (line_number, line)
            for line_number, line in enumerate(lines, start=1)
            if line.startswith(label)
        ]
        if not matches:
            unresolved.append(
                DeterministicUnresolvedField(
                    field_name=field_name,
                    reason=DeterministicUnresolvedReason.MISSING_LABEL,
                )
            )
            continue
        if len(matches) > 1:
            distinct_values = {
                line[len(label) :].strip()
                for _line_number, line in matches
            }
            unresolved.append(
                DeterministicUnresolvedField(
                    field_name=field_name,
                    reason=(
                        DeterministicUnresolvedReason.CONFLICTING_VALUES
                        if len(distinct_values) > 1
                        else DeterministicUnresolvedReason.DUPLICATE_LABEL
                    ),
                )
            )
            continue

        line_number, line = matches[0]
        source_value = line[len(label) :].strip()
        if not source_value:
            unresolved.append(
                DeterministicUnresolvedField(
                    field_name=field_name,
                    reason=DeterministicUnresolvedReason.EMPTY_VALUE,
                )
            )
            continue

        provenance = _local_line_provenance(
            document_id=document_id,
            line_number=line_number,
            line=line,
        )
        field_kind = field_kinds[field_name.value]
        if field_kind == "money":
            money_values = _decimal_values(source_value)
            if len(money_values) != 1:
                unresolved.append(
                    DeterministicUnresolvedField(
                        field_name=field_name,
                        reason=DeterministicUnresolvedReason.INVALID_FORMAT,
                    )
                )
                continue
            resolved[field_name.value] = SupportedMoneyField(
                status="supported",
                value=money_values[0],
                provenance=(provenance,),
            )
        elif field_kind == "integer":
            integer_values = [
                int(token)
                for token in re.findall(r"(?<![\w.])\d+(?![\w.])", source_value)
            ]
            if len(integer_values) != 1:
                unresolved.append(
                    DeterministicUnresolvedField(
                        field_name=field_name,
                        reason=DeterministicUnresolvedReason.INVALID_FORMAT,
                    )
                )
                continue
            resolved[field_name.value] = SupportedIntegerField(
                status="supported",
                value=integer_values[0],
                provenance=(provenance,),
            )
        elif len(source_value) > 512:
            unresolved.append(
                DeterministicUnresolvedField(
                    field_name=field_name,
                    reason=DeterministicUnresolvedReason.INVALID_FORMAT,
                )
            )
        else:
            resolved[field_name.value] = SupportedStringField(
                status="supported",
                value=source_value,
                provenance=(provenance,),
            )

    if unresolved:
        reasons = {item.reason for item in unresolved}
        if reasons & {
            DeterministicUnresolvedReason.DUPLICATE_LABEL,
            DeterministicUnresolvedReason.CONFLICTING_VALUES,
        }:
            code = "evidence_validation_failed"
        elif reasons & {
            DeterministicUnresolvedReason.EMPTY_VALUE,
            DeterministicUnresolvedReason.INVALID_FORMAT,
        }:
            code = "schema_validation_failed"
        else:
            code = "unsupported_required_field"
        raise DeterministicExtractionError(code, tuple(unresolved))

    try:
        fields_model = _FINAL_FIELD_MODELS[expected_document_type]
        fields = fields_model(**resolved)
        metadata = ExtractionMetadata(
            extraction_method="deterministic_labels_v1",
            prompt_version=None,
            model_provider=None,
            model_name=None,
            attempt_count=0,
        )
        extraction_model = _FINAL_EXTRACTION_MODELS[expected_document_type]
        return extraction_model(
            document_id=document_id,
            document_type=expected_document_type.value,
            classification=classification,
            metadata=metadata,
            fields=fields,
        )
    except ValidationError:
        raise ExtractionValidationError("schema_validation_failed") from None


def _local_line_provenance(
    *,
    document_id: str,
    line_number: int,
    line: str,
) -> SourceProvenance:
    """Create provenance directly from one local source line."""

    return SourceProvenance(
        document_id=document_id,
        line_start=line_number,
        line_end=line_number,
        char_start=0,
        char_end=len(line),
        evidence_sha256=hashlib.sha256(line.encode("utf-8")).hexdigest(),
        confidence=1.0,
    )


def parse_extraction_proposal(
    raw_output: str,
    document_type: DocumentType,
) -> ExtractionProposal:
    """Parse exactly one JSON object and enforce its document-specific schema."""

    if not isinstance(raw_output, str):
        raise ExtractionValidationError("invalid_model_output")
    if not raw_output or len(raw_output.encode("utf-8")) > MAX_MODEL_OUTPUT_BYTES:
        raise ExtractionValidationError("invalid_model_output")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        """Reject ambiguous JSON objects instead of keeping the last key."""

        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ExtractionValidationError("invalid_model_output")
            result[key] = value
        return result

    try:
        payload = json.loads(raw_output, object_pairs_hook=reject_duplicate_keys)
        if not isinstance(payload, dict):
            raise ExtractionValidationError("invalid_model_output")
        model = PROPOSAL_MODELS.get(document_type)
        if model is None:
            raise ExtractionValidationError("unsupported_document_type")
        return model.model_validate(payload)  # type: ignore[return-value]
    except ExtractionValidationError:
        raise
    except (JSONDecodeError, ValidationError, TypeError, ValueError):
        # Model-validation exceptions can retain the rejected output in their
        # input details.  Suppress that chain at this sanitized boundary.
        raise ExtractionValidationError("invalid_model_output") from None


# Alias kept separate so exception handling never has to include parser details.
JSONDecodeError = json.JSONDecodeError


def resolve_proposal_evidence(
    proposal: ExtractionProposal,
    *,
    document_id: str,
    document_text: str,
    classification: DocumentClassification,
    model_name: str,
    attempt_count: int,
) -> ValidatedExtraction:
    """Resolve temporary source quotes into local provenance pointers."""

    document_type = DocumentType(proposal.document_type)
    if classification.document_type is not document_type:
        raise ExtractionValidationError("schema_validation_failed")
    labels = _FIELD_LABELS[document_type]
    values: dict[str, object] = {}
    for field_name in type(proposal.fields).model_fields:
        proposed_field = getattr(proposal.fields, field_name)
        if isinstance(proposed_field, UnsupportedProposal):
            values[field_name] = UnsupportedField(
                status="unsupported",
                reason=proposed_field.reason,
            )
            continue
        provenance = _resolve_source_claim(
            proposed_field,
            document_id=document_id,
            document_text=document_text,
            expected_label=labels[field_name],
        )
        if isinstance(proposed_field, SupportedStringProposal):
            values[field_name] = SupportedStringField(
                status="supported",
                value=proposed_field.value,
                provenance=(provenance,),
            )
        elif isinstance(proposed_field, SupportedMoneyProposal):
            try:
                money_value = Decimal(proposed_field.value)
            except InvalidOperation:
                raise ExtractionValidationError("schema_validation_failed") from None
            values[field_name] = SupportedMoneyField(
                status="supported",
                value=money_value,
                provenance=(provenance,),
            )
        elif isinstance(proposed_field, SupportedIntegerProposal):
            values[field_name] = SupportedIntegerField(
                status="supported",
                value=proposed_field.value,
                provenance=(provenance,),
            )
        else:
            raise ExtractionValidationError("schema_validation_failed")

    try:
        fields = _FINAL_FIELD_MODELS[document_type](**values)
        metadata = ExtractionMetadata(
            model_name=model_name,
            attempt_count=attempt_count,
        )
        extraction_model = _FINAL_EXTRACTION_MODELS[document_type]
        return extraction_model(
            document_id=document_id,
            document_type=document_type.value,
            classification=classification,
            metadata=metadata,
            fields=fields,
        )
    except ValidationError:
        raise ExtractionValidationError("schema_validation_failed") from None


def enforce_evidence_guard(
    extraction: ValidatedExtraction,
    document_text: str,
) -> ValidatedExtraction:
    """Fail closed unless every required field is supported by intact evidence."""

    document_type = DocumentType(extraction.document_type)
    labels = _FIELD_LABELS[document_type]
    lines = document_text.splitlines()
    for field_name, field in iter_extracted_fields(extraction):
        if isinstance(field, UnsupportedField):
            raise ExtractionValidationError("unsupported_required_field")
        if not field.provenance:
            raise ExtractionValidationError("evidence_validation_failed")
        for provenance in field.provenance:
            if provenance.document_id != extraction.document_id:
                raise ExtractionValidationError("evidence_validation_failed")
            if provenance.line_start != provenance.line_end:
                raise ExtractionValidationError("evidence_validation_failed")
            index = provenance.line_start - 1
            if index < 0 or index >= len(lines):
                raise ExtractionValidationError("evidence_validation_failed")
            line = lines[index]
            if provenance.char_end > len(line):
                raise ExtractionValidationError("evidence_validation_failed")
            evidence = line[provenance.char_start : provenance.char_end]
            if not line.startswith(labels[field_name]):
                raise ExtractionValidationError("evidence_validation_failed")
            if provenance.confidence < MIN_SOURCE_CONFIDENCE:
                raise ExtractionValidationError("evidence_validation_failed")
            if hashlib.sha256(evidence.encode("utf-8")).hexdigest() != provenance.evidence_sha256:
                raise ExtractionValidationError("evidence_validation_failed")
            if not _evidence_supports_value(
                field.value,
                evidence,
                expected_label=labels[field_name],
            ):
                raise ExtractionValidationError("evidence_validation_failed")

    guarded_metadata = ExtractionMetadata(
        schema_version=extraction.metadata.schema_version,
        extraction_method=extraction.metadata.extraction_method,
        prompt_version=extraction.metadata.prompt_version,
        model_provider=extraction.metadata.model_provider,
        model_name=extraction.metadata.model_name,
        attempt_count=extraction.metadata.attempt_count,
        evidence_guard_passed=True,
        evidence_guard_version="evidence-guard-v1",
    )
    guarded_values = extraction.model_dump(
        mode="python",
        exclude={"metadata"},
        warnings="none",
    )
    guarded_values["metadata"] = guarded_metadata
    try:
        return type(extraction).model_validate(guarded_values)  # type: ignore[return-value]
    except ValidationError:
        raise ExtractionValidationError("schema_validation_failed") from None


def _resolve_source_claim(
    proposed_field: SupportedStringProposal | SupportedMoneyProposal | SupportedIntegerProposal,
    *,
    document_id: str,
    document_text: str,
    expected_label: str,
) -> SourceProvenance:
    """Verify one proposed quote and replace it with a local source locator."""

    lines = document_text.splitlines()
    source = proposed_field.source
    index = source.line_number - 1
    if index < 0 or index >= len(lines):
        raise ExtractionValidationError("evidence_validation_failed")
    line = lines[index]
    # Exact full-line evidence keeps locators deterministic and auditable.
    if source.quote != line or not line.startswith(expected_label):
        raise ExtractionValidationError("evidence_validation_failed")
    if source.confidence < MIN_SOURCE_CONFIDENCE:
        raise ExtractionValidationError("evidence_validation_failed")
    proposed_value: str | Decimal | int = proposed_field.value
    if isinstance(proposed_field, SupportedMoneyProposal):
        try:
            proposed_value = Decimal(proposed_field.value)
        except InvalidOperation:
            raise ExtractionValidationError("schema_validation_failed") from None
    if not _evidence_supports_value(
        proposed_value,
        source.quote,
        expected_label=expected_label,
    ):
        raise ExtractionValidationError("evidence_validation_failed")
    return SourceProvenance(
        document_id=document_id,
        line_start=source.line_number,
        line_end=source.line_number,
        char_start=0,
        char_end=len(line),
        evidence_sha256=hashlib.sha256(line.encode("utf-8")).hexdigest(),
        confidence=source.confidence,
    )


def _evidence_supports_value(
    value: str | Decimal | int,
    evidence: str,
    *,
    expected_label: str,
) -> bool:
    """Check that one typed value is exactly represented after its field label."""

    if not evidence.startswith(expected_label):
        return False
    source_value = evidence[len(expected_label) :].strip()
    if not source_value:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, Decimal):
        candidates = _decimal_values(source_value)
        return len(candidates) == 1 and candidates[0] == value
    if isinstance(value, int):
        candidates = [
            int(token)
            for token in re.findall(r"(?<![\w.])\d+(?![\w.])", source_value)
        ]
        return len(candidates) == 1 and candidates[0] == value
    if isinstance(value, str):
        # Agent 1 is an extractor, not a normalizer. Exact lexical equality
        # prevents case or whitespace changes from silently altering identifiers
        # (account numbers, employee IDs, EINs, and SSNs) or any other source value.
        return bool(value) and value == source_value
    return False


def _decimal_values(text: str) -> list[Decimal]:
    """Parse unambiguous money-like tokens for exact Decimal comparison."""

    values: list[Decimal] = []
    for token in re.findall(r"(?<!\w)[+-]?\$?\d[\d,]*(?:\.\d{1,2})?(?!\w)", text):
        try:
            values.append(Decimal(token.replace("$", "").replace(",", "")))
        except InvalidOperation:
            continue
    return values
