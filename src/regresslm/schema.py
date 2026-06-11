"""Core datamodel: cases, datasets, scores, results."""
from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, Field


class Case(BaseModel):
    """One golden example: an input, an optional reference answer, and metadata
    used by scorers (e.g. expected label, accepted substrings, json schema)."""
    id: str
    input: Any
    reference: Any | None = None
    tags: list[str] = Field(default_factory=list)
    # Free-form bag the scorers read (expected label, regex, schema, etc.)
    expect: dict[str, Any] = Field(default_factory=dict)


class Dataset(BaseModel):
    name: str
    cases: list[Case]

    def __len__(self) -> int:
        return len(self.cases)


class Score(BaseModel):
    scorer: str
    value: float            # normalized 0.0..1.0
    passed: bool
    detail: str = ""


class CaseResult(BaseModel):
    case_id: str
    output: Any
    scores: list[Score]
    # Aggregate of this case's scores (mean of values).
    score: float
    passed: bool
    latency_ms: int = 0
    error: str | None = None


class RunResult(BaseModel):
    run_id: str
    dataset: str
    target: str
    model: str = "unknown"
    created_at: dt.datetime
    case_results: list[CaseResult]

    @property
    def mean_score(self) -> float:
        if not self.case_results:
            return 0.0
        return round(sum(c.score for c in self.case_results) / len(self.case_results), 6)

    @property
    def pass_rate(self) -> float:
        if not self.case_results:
            return 0.0
        return round(sum(1 for c in self.case_results if c.passed) / len(self.case_results), 6)

    def per_scorer_means(self) -> dict[str, float]:
        sums: dict[str, list[float]] = {}
        for cr in self.case_results:
            for s in cr.scores:
                sums.setdefault(s.scorer, []).append(s.value)
        return {k: round(sum(v) / len(v), 6) for k, v in sums.items()}
