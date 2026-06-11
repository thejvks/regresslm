"""Scorer protocol. A scorer maps (output, case) -> Score in [0,1]."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..schema import Case, Score


@dataclass
class ScorerContext:
    """Everything a scorer might need beyond the raw output."""
    case: Case


@runtime_checkable
class Scorer(Protocol):
    name: str

    def score(self, output: Any, ctx: ScorerContext) -> Score:
        ...
