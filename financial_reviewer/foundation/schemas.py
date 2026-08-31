"""Define strict data contracts across the local review pipeline.

Why this file exists:
    The model's JSON is untrusted.  A separate, typed release representation is
    needed so unparsed or unverified values cannot accidentally cross the
    application boundary.

What it owns:
    Document/failure enums, model proposal schemas, supported/unsupported field
    variants, source provenance, document-specific validated extractions,
    extraction metadata, and the terminal ``ReviewOutcome`` invariant.

Key boundary:
    Proposal models may temporarily contain source quotes.  Evidence resolution
    replaces them with opaque locators and hashes.  ``ReviewOutcome`` rejects a
    released result unless its extraction passed the evidence guard and every
    supported field carries provenance.

What it does not own:
    It defines and validates shapes; it does not call the model, find evidence,
    route the graph, write storage, or decide retries.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Annotated, Iterator, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    model_validator,
)


# Persisted contract identifiers make audit records explain which output shape
# and extraction instruction were active when a review was produced.
SCHEMA_VERSION = "1.1"
PROMPT_VERSION = "agent1-extraction-v1"


class StrictModel(BaseModel):
    """Base model that rejects undeclared fields and remains immutable."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
    )


class DocumentType(str, Enum):
    """Supported document classifications plus the pre-classification state."""

    BANK_STATEMENT = "bank_statement"
    PAY_STUB = "pay_stub"
    TAX_FORM = "tax_form"
    UNKNOWN = "unknown"


class WorkflowStatus(str, Enum):
    """Terminal public outcomes: released or routed to human review."""

    RELEASED = "released"
    HUMAN_REVIEW = "human_review"


class FailureCode(str, Enum):
    """Sanitized failure categories allowed to cross the review boundary."""

    INVALID_INPUT = "invalid_input"
    UNSUPPORTED_DOCUMENT_TYPE = "unsupported_document_type"
    LOCAL_MODEL_ERROR = "local_model_error"
    INVALID_MODEL_OUTPUT = "invalid_model_output"
    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"
    EVIDENCE_VALIDATION_FAILED = "evidence_validation_failed"
    UNSUPPORTED_REQUIRED_FIELD = "unsupported_required_field"
    LOCAL_STORAGE_ERROR = "local_storage_error"
    AUDIT_WRITE_FAILED = "audit_write_failed"
    UNSAFE_CONFIGURATION = "unsafe_configuration"
    INTERNAL_FAILURE = "internal_failure"


class UnsupportedReason(str, Enum):
    """Closed reasons a model may use when source support is unavailable."""

    NOT_PRESENT = "not_present"
    ILLEGIBLE = "illegible"
    LOW_CONFIDENCE = "low_confidence"
    AMBIGUOUS = "ambiguous"


class DeterministicUnresolvedReason(str, Enum):
    """Closed reasons a deterministic document extractor may stop safely."""

    MISSING_LABEL = "missing_label"
    DUPLICATE_LABEL = "duplicate_label"
    CONFLICTING_VALUES = "conflicting_values"
    EMPTY_VALUE = "empty_value"
    INVALID_FORMAT = "invalid_format"


class PayStubFieldName(str, Enum):
    """Exact pay-stub fields supported by deterministic extraction."""

    EMPLOYEE_NAME = "employee_name"
    EMPLOYEE_ID = "employee_id"
    EMPLOYER_NAME = "employer_name"
    EMPLOYER_EIN = "employer_ein"
    MONTHLY_INCOME = "monthly_income"
    PAY_PERIOD_MONTHS = "pay_period_months"
    PAY_PERIOD_YEAR = "pay_period_year"


class BankStatementFieldName(str, Enum):
    """Exact bank-statement fields supported by deterministic extraction."""

    ACCOUNT_HOLDER_NAME = "account_holder_name"
    ACCOUNT_NUMBER = "account_number"
    BANK_NAME = "bank_name"
    STATEMENT_MONTH = "statement_month"
    MONTHLY_DEPOSITS = "monthly_deposits"


class TaxFormFieldName(str, Enum):
    """Exact W-2 fields supported by deterministic extraction."""

    EMPLOYEE_NAME = "employee_name"
    EMPLOYEE_SSN = "employee_ssn"
    EMPLOYEE_ADDRESS = "employee_address"
    EMPLOYER_NAME = "employer_name"
    EMPLOYER_EIN = "employer_ein"
    ANNUAL_WAGES = "annual_wages"
    FEDERAL_TAX_WITHHELD = "federal_tax_withheld"
    TAX_YEAR = "tax_year"


DeterministicFieldName: TypeAlias = (
    BankStatementFieldName | PayStubFieldName | TaxFormFieldName
)


class DeterministicUnresolvedField(StrictModel):
    """PII-free field-level reason retained when deterministic extraction stops."""

    field_name: DeterministicFieldName
    reason: DeterministicUnresolvedReason


class ModelSourceClaim(StrictModel):
    """Temporary evidence proposed by the local model.

    ``quote`` is never copied into logs, audit records, telemetry, or a released
    extraction.  The evidence guard verifies it against the local document and
    replaces it with ``SourceProvenance``.
    """

    line_number: Annotated[StrictInt, Field(ge=1)]
    quote: Annotated[StrictStr, Field(min_length=1, max_length=512, repr=False)]
    confidence: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]


class SupportedStringProposal(StrictModel):
    """Untrusted model-proposed string paired with a temporary source claim."""

    status: Literal["supported"]
    value: Annotated[StrictStr, Field(min_length=1, max_length=512, repr=False)]
    source: ModelSourceClaim = Field(repr=False)


class SupportedMoneyProposal(StrictModel):
    """Untrusted non-negative money string paired with a source claim."""

    status: Literal["supported"]
    value: Annotated[
        StrictStr,
        # Use only regex constructs that Ollama's JSON-grammar compiler accepts.
        # This is equivalent to the non-capturing-group form: zero is valid,
        # other whole values cannot have leading zeroes, and cents are optional.
        Field(pattern=r"^(0|[1-9][0-9]*)(\.[0-9]{1,2})?$", max_length=32, repr=False),
    ]
    source: ModelSourceClaim = Field(repr=False)


class SupportedIntegerProposal(StrictModel):
    """Untrusted non-negative integer paired with a temporary source claim."""

    status: Literal["supported"]
    value: Annotated[StrictInt, Field(ge=0)]
    source: ModelSourceClaim = Field(repr=False)


class UnsupportedProposal(StrictModel):
    """Explicit model statement that a required field lacks usable support."""

    status: Literal["unsupported"]
    reason: UnsupportedReason


StringProposal: TypeAlias = Annotated[
    SupportedStringProposal | UnsupportedProposal,
    Field(discriminator="status"),
]
MoneyProposal: TypeAlias = Annotated[
    SupportedMoneyProposal | UnsupportedProposal,
    Field(discriminator="status"),
]
IntegerProposal: TypeAlias = Annotated[
    SupportedIntegerProposal | UnsupportedProposal,
    Field(discriminator="status"),
]


class BankStatementProposalFields(StrictModel):
    """Exact model-response fields required for a bank statement proposal."""

    account_holder_name: StringProposal
    account_number: StringProposal
    bank_name: StringProposal
    statement_month: StringProposal
    monthly_deposits: MoneyProposal


class PayStubProposalFields(StrictModel):
    """Exact model-response fields required for a pay-stub proposal."""

    employee_name: StringProposal
    employee_id: StringProposal
    employer_name: StringProposal
    employer_ein: StringProposal
    monthly_income: MoneyProposal
    pay_period_months: IntegerProposal
    pay_period_year: IntegerProposal


class TaxFormProposalFields(StrictModel):
    """Exact model-response fields required for a tax-form proposal."""

    employee_name: StringProposal
    employee_ssn: StringProposal
    employee_address: StringProposal
    employer_name: StringProposal
    employer_ein: StringProposal
    annual_wages: MoneyProposal
    federal_tax_withheld: MoneyProposal
    tax_year: IntegerProposal


class BankStatementProposal(StrictModel):
    """Document-discriminated untrusted proposal for a bank statement."""

    document_type: Literal["bank_statement"]
    fields: BankStatementProposalFields = Field(repr=False)


class PayStubProposal(StrictModel):
    """Document-discriminated untrusted proposal for a pay stub."""

    document_type: Literal["pay_stub"]
    fields: PayStubProposalFields = Field(repr=False)


class TaxFormProposal(StrictModel):
    """Document-discriminated untrusted proposal for a tax form."""

    document_type: Literal["tax_form"]
    fields: TaxFormProposalFields = Field(repr=False)


ExtractionProposal: TypeAlias = (
    BankStatementProposal | PayStubProposal | TaxFormProposal
)


class SourceProvenance(StrictModel):
    """A local, non-content-bearing pointer to verified source evidence."""

    document_id: Annotated[StrictStr, Field(pattern=r"^doc_[0-9a-f]{32}$")]
    page_number: Literal[1] = 1
    line_start: Annotated[StrictInt, Field(ge=1)]
    line_end: Annotated[StrictInt, Field(ge=1)]
    char_start: Annotated[StrictInt, Field(ge=0)]
    char_end: Annotated[StrictInt, Field(gt=0)]
    evidence_sha256: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]
    confidence: Annotated[StrictFloat, Field(ge=0.0, le=1.0)]

    @model_validator(mode="after")
    def validate_span(self) -> "SourceProvenance":
        """Require forward-moving, non-empty line and character coordinates."""

        if self.line_end < self.line_start or self.char_end <= self.char_start:
            raise ValueError("invalid source span")
        return self


class SupportedStringField(StrictModel):
    """Verified string value carrying at least one provenance pointer."""

    status: Literal["supported"]
    value: Annotated[StrictStr, Field(min_length=1, max_length=512, repr=False)]
    provenance: Annotated[tuple[SourceProvenance, ...], Field(min_length=1)]


class SupportedMoneyField(StrictModel):
    """Verified non-negative Decimal carrying source provenance."""

    status: Literal["supported"]
    value: Annotated[Decimal, Field(ge=Decimal("0"), repr=False)]
    provenance: Annotated[tuple[SourceProvenance, ...], Field(min_length=1)]


class SupportedIntegerField(StrictModel):
    """Verified non-negative integer carrying source provenance."""

    status: Literal["supported"]
    value: Annotated[StrictInt, Field(ge=0)]
    provenance: Annotated[tuple[SourceProvenance, ...], Field(min_length=1)]


class UnsupportedField(StrictModel):
    """Validated absence marker that cannot carry a value or provenance."""

    status: Literal["unsupported"]
    value: None = None
    provenance: Annotated[tuple[SourceProvenance, ...], Field(max_length=0)] = ()
    reason: UnsupportedReason


StringField: TypeAlias = Annotated[
    SupportedStringField | UnsupportedField,
    Field(discriminator="status"),
]
MoneyField: TypeAlias = Annotated[
    SupportedMoneyField | UnsupportedField,
    Field(discriminator="status"),
]
IntegerField: TypeAlias = Annotated[
    SupportedIntegerField | UnsupportedField,
    Field(discriminator="status"),
]


class BankStatementFields(StrictModel):
    """Typed supported-or-unsupported fields for a bank statement."""

    account_holder_name: StringField
    account_number: StringField
    bank_name: StringField
    statement_month: StringField
    monthly_deposits: MoneyField


class PayStubFields(StrictModel):
    """Typed supported-or-unsupported fields for a pay stub."""

    employee_name: StringField
    employee_id: StringField
    employer_name: StringField
    employer_ein: StringField
    monthly_income: MoneyField
    pay_period_months: IntegerField
    pay_period_year: IntegerField


class TaxFormFields(StrictModel):
    """Typed supported-or-unsupported fields for a tax form."""

    employee_name: StringField
    employee_ssn: StringField
    employee_address: StringField
    employer_name: StringField
    employer_ein: StringField
    annual_wages: MoneyField
    federal_tax_withheld: MoneyField
    tax_year: IntegerField


class DocumentClassification(StrictModel):
    """Deterministic document classification with hashed header evidence."""

    document_type: DocumentType
    method: Literal["deterministic_header_v1"]
    confidence: Literal[1.0]
    header_evidence_sha256: Annotated[StrictStr, Field(pattern=r"^[0-9a-f]{64}$")]


class ExtractionMetadata(StrictModel):
    """Extraction method, versions, attempts, and release attestation.

    Model metadata is required only for the legacy local-model proposal path.
    The deterministic pay-stub path records no provider, model, prompt, or model
    attempt so provenance never implies that inference produced the value.
    """

    schema_version: Literal["1.1"] = SCHEMA_VERSION
    extraction_method: Literal["local_model_v1", "deterministic_labels_v1"] = (
        "local_model_v1"
    )
    prompt_version: Literal["agent1-extraction-v1"] | None = PROMPT_VERSION
    model_provider: Literal["ollama"] | None = "ollama"
    model_name: Annotated[StrictStr, Field(min_length=1, max_length=128)] | None = None
    attempt_count: Annotated[StrictInt, Field(ge=0, le=2)]
    evidence_guard_passed: StrictBool = False
    evidence_guard_version: Literal["evidence-guard-v1"] | None = None

    @model_validator(mode="after")
    def validate_evidence_attestation(self) -> "ExtractionMetadata":
        """Require evidence status and evidence-guard version to agree."""

        if self.evidence_guard_passed != (self.evidence_guard_version is not None):
            raise ValueError("evidence guard status and version must agree")
        model_metadata = (
            self.prompt_version,
            self.model_provider,
            self.model_name,
        )
        if self.extraction_method == "deterministic_labels_v1":
            if self.attempt_count != 0 or any(value is not None for value in model_metadata):
                raise ValueError("deterministic extraction cannot carry model metadata")
        elif self.attempt_count < 1 or any(value is None for value in model_metadata):
            raise ValueError("local-model extraction requires model metadata")
        return self


class BankStatementExtraction(StrictModel):
    """Source-linked typed extraction for a classified bank statement."""

    document_id: Annotated[StrictStr, Field(pattern=r"^doc_[0-9a-f]{32}$")]
    document_type: Literal["bank_statement"]
    classification: DocumentClassification
    metadata: ExtractionMetadata
    fields: BankStatementFields = Field(repr=False)


class PayStubExtraction(StrictModel):
    """Source-linked typed extraction for a classified pay stub."""

    document_id: Annotated[StrictStr, Field(pattern=r"^doc_[0-9a-f]{32}$")]
    document_type: Literal["pay_stub"]
    classification: DocumentClassification
    metadata: ExtractionMetadata
    fields: PayStubFields = Field(repr=False)


class TaxFormExtraction(StrictModel):
    """Source-linked typed extraction for a classified tax form."""

    document_id: Annotated[StrictStr, Field(pattern=r"^doc_[0-9a-f]{32}$")]
    document_type: Literal["tax_form"]
    classification: DocumentClassification
    metadata: ExtractionMetadata
    fields: TaxFormFields = Field(repr=False)


ValidatedExtraction: TypeAlias = Annotated[
    BankStatementExtraction | PayStubExtraction | TaxFormExtraction,
    Field(discriminator="document_type"),
]


class ReviewOutcome(StrictModel):
    """The only public release envelope for Milestone 1."""

    correlation_id: Annotated[StrictStr, Field(pattern=r"^corr_[0-9a-f]{32}$")]
    idempotency_key: Annotated[StrictStr, Field(pattern=r"^idem_[0-9a-f]{32}$")]
    status: WorkflowStatus
    document_type: DocumentType
    validated_extraction: ValidatedExtraction | None = Field(default=None, repr=False)
    failure_code: FailureCode | None = None
    human_review_required: StrictBool

    @model_validator(mode="after")
    def enforce_release_boundary(self) -> "ReviewOutcome":
        """Prevent unguarded, unsupported, or mismatched data from release.

        Released outcomes require an evidence-attested extraction whose type,
        classification, fields, and document IDs agree.  Human-review outcomes
        must carry no extraction and must include a sanitized failure code.
        """

        if self.status is WorkflowStatus.RELEASED:
            if self.validated_extraction is None or self.failure_code is not None:
                raise ValueError("released outcome requires validated extraction")
            if self.human_review_required:
                raise ValueError("released outcome cannot require human review")
            extraction = self.validated_extraction
            if not extraction.metadata.evidence_guard_passed:
                raise ValueError("released outcome requires evidence-guard attestation")
            if extraction.document_type != self.document_type.value:
                raise ValueError("outcome and extraction document types must agree")
            if extraction.classification.document_type is not self.document_type:
                raise ValueError("classification and extraction document types must agree")
            for _field_name, field in iter_extracted_fields(extraction):
                if isinstance(field, UnsupportedField):
                    raise ValueError("released outcomes cannot contain unsupported fields")
                if any(
                    source.document_id != extraction.document_id
                    for source in field.provenance
                ):
                    raise ValueError("released provenance must target the extraction document")
        else:
            if self.validated_extraction is not None:
                raise ValueError("unreleased extraction must not cross the boundary")
            if self.failure_code is None or not self.human_review_required:
                raise ValueError("human-review outcome requires a failure code")
        return self


# The classification result selects exactly one proposal schema for structured
# local-model output. Unknown or unsupported types have no entry and fail closed.
PROPOSAL_MODELS: dict[DocumentType, type[StrictModel]] = {
    DocumentType.BANK_STATEMENT: BankStatementProposal,
    DocumentType.PAY_STUB: PayStubProposal,
    DocumentType.TAX_FORM: TaxFormProposal,
}


def proposal_schema(document_type: DocumentType) -> dict[str, object]:
    """Return the exact local-model response schema for a supported class."""

    model = PROPOSAL_MODELS.get(document_type)
    if model is None:
        raise ValueError("unsupported document type")
    return model.model_json_schema()


def iter_extracted_fields(
    extraction: BankStatementExtraction | PayStubExtraction | TaxFormExtraction,
) -> Iterator[tuple[str, SupportedStringField | SupportedMoneyField | SupportedIntegerField | UnsupportedField]]:
    """Iterate declared fields without accepting dynamic keys."""

    for name in type(extraction.fields).model_fields:
        yield name, getattr(extraction.fields, name)
