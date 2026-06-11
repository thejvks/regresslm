"""Typer CLI: run evals, gate in CI, inspect drift."""
from __future__ import annotations

import importlib
import os
import sys
from typing import Callable

import typer
from rich.console import Console
from rich.table import Table

from .config import get_settings
from .dataset import load_dataset
from .drift import build_drift_report
from .gate import evaluate_gate
from .runner import run_eval
from .scorers import BUILTIN_SCORERS, MockJudge
from .store import RunStore

app = typer.Typer(help="RegressLM — regression testing for LLM/agent systems.")
console = Console()


def _load_callable(spec: str) -> Callable:
    """Load 'module.path:function' into a callable, resolving relative to CWD so
    users can point at their own target modules (e.g. examples.targets:fn)."""
    module_path, _, func = spec.partition(":")
    if not func:
        raise typer.BadParameter("target must be 'module.path:function'")
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    mod = importlib.import_module(module_path)
    return getattr(mod, func)


def _build_scorers(names: str):
    scorers = []
    for raw in [n.strip() for n in names.split(",") if n.strip()]:
        if raw == "llm_judge":
            scorers.append(MockJudge())  # offline-safe default; swap for LLMJudge in prod
        elif raw in BUILTIN_SCORERS:
            scorers.append(BUILTIN_SCORERS[raw]())
        else:
            raise typer.BadParameter(f"unknown scorer '{raw}'")
    return scorers


@app.command()
def run(
    dataset: str = typer.Option(..., help="path to dataset .yaml/.jsonl"),
    target: str = typer.Option(..., help="module:function of the system under test"),
    scorers: str = typer.Option("label_match", help="comma-separated scorer names"),
    model: str = typer.Option("unknown", help="model/version tag for drift tracking"),
    run_id: str = typer.Option(None, help="explicit run id"),
):
    """Run an eval and persist the result."""
    settings = get_settings()
    ds = load_dataset(dataset)
    fn = _load_callable(target)
    result = run_eval(fn, ds, _build_scorers(scorers), target_name=target, model=model, run_id=run_id)
    store = RunStore(settings.runs_dir)
    path = store.save(result)

    table = Table(title=f"Run {result.run_id} — {ds.name} ({model})")
    for c in ("case", "passed", "score", "detail"):
        table.add_column(c)
    for cr in result.case_results:
        detail = cr.error or "; ".join(s.detail for s in cr.scores if s.detail) or "ok"
        table.add_row(cr.case_id, "✅" if cr.passed else "❌", f"{cr.score:.2f}", detail[:60])
    console.print(table)
    console.print(f"mean_score=[bold]{result.mean_score:.4f}[/bold]  pass_rate={result.pass_rate:.1%}")
    console.print(f"saved → {path}")


@app.command()
def gate(
    dataset: str = typer.Option(..., help="dataset name to gate"),
    candidate: str = typer.Option(None, help="candidate run id (default: latest)"),
):
    """Compare the candidate run to baseline; exit non-zero on regression (CI gate)."""
    settings = get_settings()
    store = RunStore(settings.runs_dir)
    cand = store.load(candidate) if candidate else store.latest(dataset)
    base = store.baseline(dataset)
    if cand is None or base is None:
        console.print("[red]need at least a baseline and a candidate run[/red]")
        raise typer.Exit(2)
    if cand.run_id == base.run_id:
        console.print("[yellow]candidate == baseline; nothing to gate[/yellow]")
        raise typer.Exit(0)

    res = evaluate_gate(cand, base, max_score_drop=settings.max_score_drop,
                        max_newly_failing=settings.max_newly_failing)
    color = "green" if res.passed else "red"
    console.print(f"[{color}]gate {'PASS' if res.passed else 'FAIL'}[/{color}]  "
                  f"baseline={res.baseline_mean:.4f} candidate={res.candidate_mean:.4f} "
                  f"delta={res.score_delta:+.4f}")
    for r in res.reasons:
        console.print(f"  • {r}")
    if res.newly_failing:
        console.print(f"  regressed: {res.newly_failing}")
    raise typer.Exit(0 if res.passed else 1)


@app.command()
def baseline(dataset: str, run_id: str):
    """Pin a run as the baseline for a dataset."""
    store = RunStore(get_settings().runs_dir)
    store.set_baseline(run_id, dataset)
    console.print(f"baseline for '{dataset}' → {run_id}")


@app.command()
def drift(dataset: str):
    """Show score/pass-rate across runs and compare models."""
    store = RunStore(get_settings().runs_dir)
    runs = store.list_runs(dataset)
    if not runs:
        console.print("[yellow]no runs[/yellow]")
        raise typer.Exit(0)
    report = build_drift_report(dataset, runs)
    table = Table(title=f"Drift — {dataset}")
    for c in ("run", "model", "mean_score", "pass_rate"):
        table.add_column(c)
    for p in report.points:
        table.add_row(p.run_id, p.model, f"{p.mean_score:.4f}", f"{p.pass_rate:.1%}")
    console.print(table)
    console.print(f"by_model={report.by_model}  best={report.best_model}  worst={report.worst_model}")


def main():
    app()


if __name__ == "__main__":
    main()
