import datetime as dt

from regresslm.dataset import load_dataset
from regresslm.drift import build_drift_report
from regresslm.gate import evaluate_gate
from regresslm.runner import run_eval
from regresslm.scorers import LabelMatch
from regresslm.store import RunStore

from examples.targets import triage_v1, triage_v2_regressed

DATASET = "examples/datasets/support_triage.yaml"


def _run(target, model, run_id, now):
    ds = load_dataset(DATASET)
    return run_eval(target, ds, [LabelMatch()], target_name=model, model=model,
                    run_id=run_id, now=now)


def test_runner_scores_good_target():
    now = dt.datetime(2026, 6, 1)
    result = _run(triage_v1, "v1", "run-v1", now)
    assert len(result.case_results) == 6
    # v1 should classify all of the labeled cases correctly.
    assert result.pass_rate == 1.0
    assert result.mean_score == 1.0


def test_runner_catches_bad_target():
    now = dt.datetime(2026, 6, 2)
    result = _run(triage_v2_regressed, "v2", "run-v2", now)
    # v2 broke billing detection -> the two billing cases now fail.
    failing = [cr.case_id for cr in result.case_results if not cr.passed]
    assert "billing-1" in failing and "billing-2" in failing
    assert result.pass_rate < 1.0


def test_target_exception_is_a_failing_case():
    now = dt.datetime(2026, 6, 2)
    ds = load_dataset(DATASET)

    def boom(_):
        raise RuntimeError("model timeout")

    result = run_eval(boom, ds, [LabelMatch()], run_id="boom", now=now)
    assert all(not cr.passed for cr in result.case_results)
    assert all(cr.error for cr in result.case_results)


def test_gate_fails_on_regression():
    now = dt.datetime(2026, 6, 1)
    base = _run(triage_v1, "v1", "run-v1", now)
    cand = _run(triage_v2_regressed, "v2", "run-v2", now)
    res = evaluate_gate(cand, base, max_score_drop=0.02, max_newly_failing=0)
    assert res.passed is False
    assert "billing-1" in res.newly_failing
    assert res.score_delta < 0


def test_gate_passes_when_stable():
    now = dt.datetime(2026, 6, 1)
    base = _run(triage_v1, "v1", "run-v1", now)
    cand = _run(triage_v1, "v1b", "run-v1b", now)
    res = evaluate_gate(cand, base)
    assert res.passed is True


def test_store_roundtrip_and_drift(tmp_path):
    store = RunStore(tmp_path / "runs")
    now = dt.datetime(2026, 6, 1)
    r1 = _run(triage_v1, "v1", "run-v1", now)
    r2 = _run(triage_v2_regressed, "v2", "run-v2", now + dt.timedelta(days=1))
    store.save(r1)
    store.save(r2)

    loaded = store.load("run-v1")
    assert loaded.mean_score == r1.mean_score

    report = build_drift_report("support_triage", store.list_runs("support_triage"))
    assert report.best_model == "v1"
    assert report.worst_model == "v2"
    assert report.by_model["v1"] > report.by_model["v2"]
