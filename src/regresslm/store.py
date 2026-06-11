"""Persist run results so we have history to gate against and detect drift.

Stored as JSON blobs keyed by run — simple, portable, diff-friendly. Swap for a
real warehouse table when volume grows; the interface stays the same.
"""
from __future__ import annotations

import json
from pathlib import Path

from .schema import RunResult


class RunStore:
    def __init__(self, root: str | Path = ".regresslm/runs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, run: RunResult) -> Path:
        path = self.root / f"{run.run_id}.json"
        path.write_text(run.model_dump_json(indent=2))
        return path

    def load(self, run_id: str) -> RunResult:
        return RunResult.model_validate_json((self.root / f"{run_id}.json").read_text())

    def list_runs(self, dataset: str | None = None) -> list[RunResult]:
        runs = []
        for p in self.root.glob("*.json"):
            try:
                r = RunResult.model_validate_json(p.read_text())
            except Exception:
                continue
            if dataset is None or r.dataset == dataset:
                runs.append(r)
        runs.sort(key=lambda r: r.created_at)
        return runs

    def latest(self, dataset: str, exclude_run_id: str | None = None) -> RunResult | None:
        runs = [r for r in self.list_runs(dataset) if r.run_id != exclude_run_id]
        return runs[-1] if runs else None

    def baseline(self, dataset: str) -> RunResult | None:
        """The run tagged as baseline, else the earliest run for the dataset."""
        marker = self.root / f"_baseline_{dataset}.txt"
        if marker.exists():
            return self.load(marker.read_text().strip())
        runs = self.list_runs(dataset)
        return runs[0] if runs else None

    def set_baseline(self, run_id: str, dataset: str) -> None:
        (self.root / f"_baseline_{dataset}.txt").write_text(run_id)
