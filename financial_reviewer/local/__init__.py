"""Device-local infrastructure adapters used by the review workflow.

``model`` is the loopback-only Ollama boundary; ``storage`` keeps validated
documents in private correlation-scoped directories; ``observability`` emits
PII-safe logs and hash-linked local audit records; ``telemetry`` defines a
sanitized trace contract with a no-op default; and ``otlp_trace`` is the
explicit loopback-only adapter for Traceboard or another local OTLP receiver.
Agent business rules and graph routing do not belong in this package.
"""
