"""Agent-owned domain behavior and future multi-agent contracts.

``agent1_extraction`` contains the implemented deterministic helpers used by
the workflow's classification, schema-validation, and evidence-guard nodes.
It is not the application entry point and does not build the graph or call the
model by itself.  ``agent2_verification`` and ``agent3_critic`` define only the
typed interfaces needed for later milestones; they perform no work today.
"""
