"""The eval runner: execute a target over a dataset, score each case, aggregate.

A *target* is the system under test: any callable `input -> output` (a prompt
chain, an agent, a classifier). Keeping it a plain callable means RegressLM
evaluates anything, not just one framework.
"""
from __future__ import annotations

import datetime as dt
import time
from typing import Any, Callable

from .schema import Case, CaseResult, Dataset, RunResult, Score
from .scorers.base import Scorer, ScorerContext

Target = Callable[[Any], Any]


def _aggregate(scores: list[Score]) -> tuple[float, bool]:
    if not scores:
        return 0.0, False
    mean = sum(s.value for s in scores) / len(scores)
    passed = all(s.passed for s in scores)
    return round(mean, 6), passed


def run_case(target: Target, case: Case, scorers: list[Scorer]) -> CaseResult:
    ctx = ScorerContext(case=case)
    started = time.perf_counter()
    try:
        output = target(case.input)
        error = None
    except Exception as e:  # a target crash is a failing case, not a crashed run
        output = None
        error = f"{type(e).__name__}: {e}"
    latency_ms = int((time.perf_counter() - started) * 1000)

    if error is not None:
        return CaseResult(case_id=case.id, output=None, scores=[], score=0.0,
                          passed=False, latency_ms=latency_ms, error=error)

    scores = [s.score(output, ctx) for s in scorers]
    agg_score, passed = _aggregate(scores)
    return CaseResult(case_id=case.id, output=output, scores=scores, score=agg_score,
                      passed=passed, latency_ms=latency_ms)


def run_eval(
    target: Target,
    dataset: Dataset,
    scorers: list[Scorer],
    *,
    target_name: str = "target",
    model: str = "unknown",
    run_id: str | None = None,
    now: dt.datetime | None = None,
) -> RunResult:
    now = now or dt.datetime.utcnow()
    run_id = run_id or f"{dataset.name}-{int(now.timestamp())}"
    results = [run_case(target, c, scorers) for c in dataset.cases]
    return RunResult(
        run_id=run_id,
        dataset=dataset.name,
        target=target_name,
        model=model,
        created_at=now,
        case_results=results,
    )
