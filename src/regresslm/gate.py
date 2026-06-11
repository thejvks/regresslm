"""Regression gate: compare a candidate run to a baseline and decide pass/fail.

This is what you wire into CI. It fails the build when:
  • mean score drops more than `max_score_drop` below baseline, OR
  • any case that PASSED in baseline now FAILS (a hard regression), beyond an
    optional tolerance count.
"""
from __future__ import annotations

from pydantic import BaseModel

from .schema import RunResult


class GateResult(BaseModel):
    passed: bool
    baseline_run: str
    candidate_run: str
    baseline_mean: float
    candidate_mean: float
    score_delta: float
    newly_failing: list[str]      # cases that regressed pass -> fail
    newly_passing: list[str]      # cases that improved fail -> pass
    reasons: list[str]


def evaluate_gate(
    candidate: RunResult,
    baseline: RunResult,
    *,
    max_score_drop: float = 0.02,
    max_newly_failing: int = 0,
) -> GateResult:
    base_pass = {cr.case_id: cr.passed for cr in baseline.case_results}
    cand_pass = {cr.case_id: cr.passed for cr in candidate.case_results}

    common = set(base_pass) & set(cand_pass)
    newly_failing = sorted(c for c in common if base_pass[c] and not cand_pass[c])
    newly_passing = sorted(c for c in common if not base_pass[c] and cand_pass[c])

    score_delta = round(candidate.mean_score - baseline.mean_score, 6)

    reasons: list[str] = []
    passed = True
    if score_delta < -max_score_drop:
        passed = False
        reasons.append(
            f"mean score dropped {score_delta:+.4f} (limit -{max_score_drop})"
        )
    if len(newly_failing) > max_newly_failing:
        passed = False
        reasons.append(
            f"{len(newly_failing)} case(s) regressed pass→fail (limit {max_newly_failing}): {newly_failing}"
        )
    if passed:
        reasons.append("no regressions detected")

    return GateResult(
        passed=passed,
        baseline_run=baseline.run_id,
        candidate_run=candidate.run_id,
        baseline_mean=baseline.mean_score,
        candidate_mean=candidate.mean_score,
        score_delta=score_delta,
        newly_failing=newly_failing,
        newly_passing=newly_passing,
        reasons=reasons,
    )
