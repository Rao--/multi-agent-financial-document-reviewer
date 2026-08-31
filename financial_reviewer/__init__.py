"""Local-first multi-agent financial-document reviewer.

Why this package exists:
    Keep the production-oriented review slice separate from older reference
    code in the repository.

What it contains:
    ``workflow`` owns the single-document Extractor flow; ``orchestration``
    connects specialist agents; ``foundation`` defines trusted configuration,
    intake, data, and handoff contracts; ``agents`` owns agent-specific logic;
    ``local`` owns device-local inference, storage, audit, and telemetry adapters.

Where to start:
    Start with ``financial_reviewer.orchestration.income_review`` for the
    multi-document flow or
    :meth:`financial_reviewer.workflow.DocumentExtractionReviewer.review`
    for one Extractor review. Package imports do not construct clients or storage.
"""
