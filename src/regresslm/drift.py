"""Drift tracking: how has quality moved across runs / model versions over time?"""
from __future__ import annotations

from pydantic import BaseModel

from .schema import RunResult


class DriftPoint(BaseModel):
    run_id: str
    model: str
    created_at: str
    mean_score: float
    pass_rate: float


class DriftReport(BaseModel):
    dataset: str
    points: list[DriftPoint]
    # Per-model average score, to compare model versions head-to-head.
    by_model: dict[str, float]
    best_model: str | None
    worst_model: str | None


def build_drift_report(dataset: str, runs: list[RunResult]) -> DriftReport:
    points = [
        DriftPoint(
            run_id=r.run_id,
            model=r.model,
            created_at=r.created_at.isoformat(),
            mean_score=r.mean_score,
            pass_rate=r.pass_rate,
        )
        for r in runs
    ]
    model_scores: dict[str, list[float]] = {}
    for r in runs:
        model_scores.setdefault(r.model, []).append(r.mean_score)
    by_model = {m: round(sum(v) / len(v), 6) for m, v in model_scores.items()}

    best = max(by_model, key=by_model.get) if by_model else None
    worst = min(by_model, key=by_model.get) if by_model else None
    return DriftReport(dataset=dataset, points=points, by_model=by_model,
                       best_model=best, worst_model=worst)
