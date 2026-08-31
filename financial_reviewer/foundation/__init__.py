"""Shared trust-boundary rules used before and between workflow components.

``config`` proves that inference and tracing settings are local and safe;
``intake`` converts an untrusted synthetic submission into a validated local
document; ``schemas`` defines the strict proposal, provenance, extraction, and
outcome contracts; and ``handoffs`` translates guarded outputs between agents
without making a model or business decision. These modules define policy and
types but do not orchestrate a review run.
"""
