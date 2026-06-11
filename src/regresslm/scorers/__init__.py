from .base import Scorer, ScorerContext
from .deterministic import (
    ExactMatch,
    Contains,
    RegexMatch,
    NumericTolerance,
    JsonValid,
    LabelMatch,
)
from .judge import LLMJudge, MockJudge

# Registry of built-in scorers by name, for config-driven eval specs.
BUILTIN_SCORERS = {
    "exact_match": ExactMatch,
    "contains": Contains,
    "regex": RegexMatch,
    "numeric_tolerance": NumericTolerance,
    "json_valid": JsonValid,
    "label_match": LabelMatch,
}

__all__ = [
    "Scorer", "ScorerContext", "ExactMatch", "Contains", "RegexMatch",
    "NumericTolerance", "JsonValid", "LabelMatch", "LLMJudge", "MockJudge",
    "BUILTIN_SCORERS",
]
