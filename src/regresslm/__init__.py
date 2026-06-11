"""RegressLM — regression testing for LLM/agent systems.

Treat prompts and models like code: golden datasets, scorers (deterministic +
LLM-as-judge), a CI gate that fails the build when quality drops, and drift
tracking across model versions over time.
"""

__version__ = "0.1.0"
