"""Generate REAL screenshots (SVG) from live RegressLM output.

Runs the actual eval/gate/drift pipeline on the bundled support-triage
dataset (good v1 vs. regressed v2) and renders the same Rich tables the
`regresslm` CLI prints, captured to SVG. No mockups.

Run:  python scripts/make_screenshots.py
Out:  docs/run.svg, docs/gate.svg, docs/drift.svg
"""
from __future__ import annotations

import os
import sys

from rich.console import Console
from rich.table import Table

# Make the bundled examples + src importable when run from the repo root.
sys.path.insert(0, os.getcwd())

from regresslm.config import get_settings
from regresslm.dataset import load_dataset
from regresslm.drift import build_drift_report
from regresslm.gate import evaluate_gate
from regresslm.runner import run_eval
from regresslm.scorers import BUILTIN_SCORERS
from regresslm.store import RunStore

from examples.targets import triage_v1, triage_v2_regressed  # type: ignore

DOCS = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
DATASET = "examples/datasets/support_triage.yaml"


def _console() -> Console:
    return Console(record=True, width=78)


def _save_svg(con: Console, name: str, title: str) -> None:
    path = os.path.join(DOCS, name)
    con.save_svg(path, title=title)
    # Strip the library's auto-inserted HTML attribution comment for a clean artifact.
    with open(path) as fh:
        lines = [ln for ln in fh if not ln.lstrip().startswith("<!--")]
    with open(path, "w") as fh:
        fh.writelines(lines)


def _run_table(con: Console, result, ds, model: str) -> None:
    table = Table(title=f"Run {result.run_id} — {ds.name} ({model})")
    for c in ("case", "passed", "score", "detail"):
        table.add_column(c)
    for cr in result.case_results:
        detail = cr.error or "; ".join(s.detail for s in cr.scores if s.detail) or "ok"
        table.add_row(cr.case_id, "✅" if cr.passed else "❌", f"{cr.score:.2f}", detail[:60])
    con.print(table)
    con.print(f"mean_score=[bold]{result.mean_score:.4f}[/bold]  pass_rate={result.pass_rate:.1%}")


def main() -> None:
    os.makedirs(DOCS, exist_ok=True)
    settings = get_settings()
    ds = load_dataset(DATASET)
    scorers = [BUILTIN_SCORERS["label_match"]()]
    store = RunStore(settings.runs_dir)

    base = run_eval(triage_v1, ds, scorers, target_name="triage_v1", model="v1", run_id="run-v1")
    store.save(base)
    store.set_baseline("run-v1", ds.name)
    cand = run_eval(triage_v2_regressed, ds, scorers, target_name="triage_v2", model="v2", run_id="run-v2")
    store.save(cand)

    # 1) Candidate run — billing detection regressed.
    con = _console()
    _run_table(con, cand, ds, "v2")
    _save_svg(con, "run.svg", "regresslm run  (candidate v2)")

    # 2) CI gate — fails the build and names the regressions.
    res = evaluate_gate(cand, base, max_score_drop=settings.max_score_drop,
                        max_newly_failing=settings.max_newly_failing)
    con = _console()
    con.print(f"[red]gate {'PASS' if res.passed else 'FAIL'}[/red]  "
              f"baseline={res.baseline_mean:.4f} candidate={res.candidate_mean:.4f} "
              f"delta={res.score_delta:+.4f}")
    for r in res.reasons:
        con.print(f"  • {r}")
    if res.newly_failing:
        con.print(f"  regressed: {res.newly_failing}")
    con.print("[dim]exit code = 1  → CI build fails[/dim]")
    _save_svg(con, "gate.svg", "regresslm gate  (CI guard)")

    # 3) Drift across runs/models.
    report = build_drift_report(ds.name, store.list_runs(ds.name))
    con = _console()
    table = Table(title=f"Drift — {ds.name}")
    for c in ("run", "model", "mean_score", "pass_rate"):
        table.add_column(c)
    for p in report.points:
        table.add_row(p.run_id, p.model, f"{p.mean_score:.4f}", f"{p.pass_rate:.1%}")
    con.print(table)
    con.print(f"best={report.best_model}  worst={report.worst_model}")
    _save_svg(con, "drift.svg", "regresslm drift")

    print(f"Wrote SVGs to {DOCS}/  (run.svg, gate.svg, drift.svg)")


if __name__ == "__main__":
    main()
