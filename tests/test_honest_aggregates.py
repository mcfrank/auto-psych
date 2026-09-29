"""Sweep summaries say which cells they cover and compare like with like
(first audit C6; second audit W1, W2).

- ``survey_sweep`` sorts a sweep's expected cells into complete (a
  ``holdout.json``), partial (started, no result) and missing (never
  started); every summary prints that accounting.
- Trajectories are pooled by position within an experiment — its seed step,
  the candidate rounds every cell has, and the end of the experiment — never
  by the running ``global_step``, which put different rounds of different
  cells at one x when an abandoned round wrote no step.
- The fitted-seed baseline is scored at the end of every experiment on that
  experiment's cumulative data (``fitted_baseline_by_experiment``), and at
  every position the loop and every baseline are averaged over the same cells.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.subjective_randomness.reporting import (
    aggregate_holdout_trajectories,
    holdout_combined_frames,
    holdout_trajectories_ggplot,
)
from src.subjective_randomness.sweep_cells import accounting_lines, survey_sweep

# ── the cell survey ───────────────────────────────────────────────────────


def _sweep(tmp_path: Path) -> Path:
    root = tmp_path / "sweep"
    for label in ("run1/gt_a", "run2/gt_a"):
        (root / label).mkdir(parents=True)
        (root / label / "holdout.json").write_text("{}", encoding="utf-8")
    (root / "run1" / "gt_b").mkdir(parents=True)  # started, never finished
    return root


def test_the_survey_sorts_expected_cells_into_complete_partial_and_missing(tmp_path):
    survey = survey_sweep(_sweep(tmp_path), n_repeats=3, gt_models=["gt_a", "gt_b"])
    assert sorted(survey.complete) == ["run1/gt_a", "run2/gt_a"]
    assert sorted(survey.partial) == ["run1/gt_b"]
    assert sorted(survey.missing) == ["run2/gt_b", "run3/gt_a", "run3/gt_b"]
    assert "stated" in survey.expected_from


def test_an_inferred_grid_says_it_was_inferred(tmp_path):
    survey = survey_sweep(_sweep(tmp_path))
    assert sorted(survey.missing) == ["run2/gt_b"]
    assert "inferred" in survey.expected_from


def test_the_accounting_lists_every_cell_left_out_and_why(tmp_path):
    survey = survey_sweep(_sweep(tmp_path), n_repeats=2, gt_models=["gt_a", "gt_b"])
    text = "\n".join(accounting_lines(survey, included=["run1/gt_a"]))
    assert "Complete: 2. Partial: 1. Missing: 1. Included in the numbers below: 1." in text
    assert "`run1/gt_b` (partial)" in text and "`run2/gt_b` (missing)" in text


# ── alignment and like-for-like baselines ────────────────────────────────


def _row(experiment, step, iteration, value):
    return {"experiment": experiment, "step": step, "iteration": iteration,
            "pearson_r": value, "pearson_r_bma": value, "rmse": 1 - value,
            "rmse_bma": 1 - value}


def _numbered(rows):
    return [{**row, "global_step": i} for i, row in enumerate(rows)]


def _fitted(values):
    return [
        {"experiment": e, "elpd_best_r": v, "elpd_best_rmse": None if v is None else 1 - v,
         "per_model": {"s": {}}}
        for e, v in enumerate(values, start=1)
    ]


def _result(rows, fitted_by_experiment, default_r=0.1, gt="gt_a"):
    final = dict(fitted_by_experiment[-1]) if fitted_by_experiment else {}
    final.pop("experiment", None)
    run = {"gt_model": gt, "trajectory": _numbered(rows), "n_eval_stimuli": 10,
           "baseline": {"per_model": {"s": default_r}},
           "fitted_baseline": final}
    if fitted_by_experiment is not None:
        run["fitted_baseline_by_experiment"] = fitted_by_experiment
    return {"gt_runs": [run]}


# Cell A ran every round of experiment 1; cell B abandoned round 1 (no step).
CELL_A = _result(
    [_row(1, 0, None, 0.10), _row(1, 1, 0, 0.20), _row(1, 2, 1, 0.30), _row(1, 3, 2, 0.40),
     _row(2, 0, None, 0.50), _row(2, 1, 0, 0.60)],
    _fitted([0.15, 0.35]),
)
CELL_B = _result(
    [_row(1, 0, None, 0.30), _row(1, 1, 0, 0.40), _row(1, 2, 2, 0.60),
     _row(2, 0, None, 0.70), _row(2, 1, 0, 0.80)],
    _fitted([0.25, 0.45]),
)


def _points(series):
    return {(p["experiment"], p["label"]): p for p in series}


def test_steps_are_pooled_by_position_within_an_experiment_not_by_global_step():
    agg = aggregate_holdout_trajectories([CELL_A, CELL_B], metric="pearson_r", error="std",
                                         labels=["run1/gt_a", "run2/gt_a"])
    (panel,) = agg["gt_models"]
    best = _points(panel["best"])
    # Round 1 exists only in cell A: it is not a pooled position.
    assert sorted(best) == [(1, "end"), (1, "round 0"), (1, "seed"),
                            (2, "end"), (2, "seed")]
    assert best[(1, "end")]["mean"] == pytest.approx((0.40 + 0.60) / 2)
    assert best[(1, "round 0")]["mean"] == pytest.approx((0.20 + 0.40) / 2)
    # global_step 2 would have averaged A's round 1 (0.30) with B's round 2 (0.60).
    assert all(p["n"] == 2 for p in panel["best"])
    xs = [p["x"] for p in panel["best"]]
    assert xs == sorted(xs) and len(set(xs)) == len(xs)
    assert panel["experiment_boundaries"] == [best[(2, "seed")]["x"]]


def test_the_fitted_baseline_follows_the_experiments_data():
    agg = aggregate_holdout_trajectories([CELL_A, CELL_B], metric="pearson_r",
                                         labels=["run1/gt_a", "run2/gt_a"])
    fitted = _points(agg["gt_models"][0]["baseline_series"]["fitted_baseline"])
    assert fitted[(1, "end")]["mean"] == pytest.approx((0.15 + 0.25) / 2)
    assert fitted[(2, "end")]["mean"] == pytest.approx((0.35 + 0.45) / 2)
    # The headline baselines are those at the end of the final experiment.
    headline = agg["gt_models"][0]["baselines"]["fitted_baseline"]
    assert headline["mean"] == pytest.approx(0.40) and headline["n"] == 2


def test_the_loop_and_every_baseline_cover_the_same_cells_at_each_position():
    # Cell B has no trusted seed at experiment 2: it leaves the loop's
    # experiment-2 points too, and is listed with the reason.
    cell_b = _result(
        [_row(1, 0, None, 0.30), _row(1, 1, 0, 0.40), _row(1, 2, 2, 0.60),
         _row(2, 0, None, 0.70), _row(2, 1, 0, 0.80)],
        _fitted([0.25, None]),
    )
    agg = aggregate_holdout_trajectories([CELL_A, cell_b], metric="pearson_r",
                                         labels=["run1/gt_a", "run2/gt_a"])
    (panel,) = agg["gt_models"]
    best = _points(panel["best"])
    assert best[(2, "end")]["n"] == 1 and best[(2, "end")]["mean"] == pytest.approx(0.60)
    assert best[(1, "end")]["n"] == 2
    assert panel["baselines"]["fitted_baseline"]["n"] == 1
    assert panel["baselines"]["baseline"]["n"] == 1
    excluded = [e for e in panel["excluded"] if e["cell"] == "run2/gt_a"]
    assert {(e["experiment"], e["label"]) for e in excluded} == {(2, "seed"), (2, "end")}
    assert "fitted-seed baseline" in excluded[0]["reason"]


def test_a_result_scored_before_per_experiment_baselines_joins_only_at_the_end():
    old = _result(
        [_row(1, 0, None, 0.30), _row(1, 1, 0, 0.40), _row(1, 2, 2, 0.60),
         _row(2, 0, None, 0.70), _row(2, 1, 0, 0.80)],
        None,
    )
    old["gt_runs"][0]["fitted_baseline"] = {"elpd_best_r": 0.45, "elpd_best_rmse": 0.55,
                                            "per_model": {"s": {}}}
    agg = aggregate_holdout_trajectories([CELL_A, old], metric="pearson_r",
                                         labels=["run1/gt_a", "run2/gt_a"])
    best = _points(agg["gt_models"][0]["best"])
    assert best[(2, "end")]["n"] == 2
    assert best[(1, "end")]["n"] == 1
    reasons = {e["reason"] for e in agg["gt_models"][0]["excluded"] if e["cell"] == "run2/gt_a"}
    assert any("re-score" in reason for reason in reasons)


def test_the_combined_figure_draws_positions_and_per_experiment_baselines():
    agg = aggregate_holdout_trajectories([CELL_A, CELL_B], metric="rmse",
                                         labels=["run1/gt_a", "run2/gt_a"])
    frames = holdout_combined_frames(agg)
    assert set(frames["trajectory"]["position"]) == {p["x"] for p in agg["gt_models"][0]["best"]}
    assert "global_step" not in frames["trajectory"].columns
    fitted = frames["baselines"][frames["baselines"]["series"] != "best model"]
    assert len(fitted) == len(agg["gt_models"][0]["best"])
    holdout_trajectories_ggplot(agg)  # builds


# ── the per-experiment fitted-seed baseline ──────────────────────────────


def test_the_fitted_seed_baseline_is_scored_on_each_experiments_cumulative_data(
    tmp_path, monkeypatch
):
    from tests.test_subjective_randomness_holdout_recovery import (
        EVAL_STIMULI,
        SEED_MODELS_DIR,
        _baseline_run,
        _stub_baseline_fits,
    )
    from src.subjective_randomness.holdout_eval import (
        fitted_seed_baseline_by_experiment,
    )

    run_root = _baseline_run(tmp_path)
    fit_responses = _stub_baseline_fits(
        monkeypatch,
        {"seed_x": np.array([0.2, 0.5, 0.9]), "seed_y": np.array([0.9, 0.6, 0.2])},
        elpd={"seed_x": -10.0, "seed_y": -5.0},
    )
    out = fitted_seed_baseline_by_experiment(
        run_root, "prototype_similarity", {"theta_alt": 0.65}, EVAL_STIMULI,
        seed_models_dir=SEED_MODELS_DIR, n_experiments=2,
        other_seed_models=["seed_x", "seed_y"], cache_dir=None, fit_kwargs={},
    )
    assert [entry["experiment"] for entry in out] == [1, 2]
    assert [entry["n_responses"] for entry in out] == [2, 3]
    exp1 = run_root / "experiment1" / "model_loop" / "responses.csv"
    exp2 = run_root / "experiment2" / "model_loop" / "responses.csv"
    assert fit_responses == [exp1, exp1, exp2, exp2]
    assert out[-1]["elpd_best_model"] == "seed_y"


def test_the_seed_means_cover_the_seeds_they_name(tmp_path, monkeypatch):
    from tests.test_subjective_randomness_holdout_recovery import (
        _baseline_run,
        _fitted_baseline,
        _stub_baseline_fits,
    )

    run_root = _baseline_run(tmp_path)
    # seed_y predicts a constant: its r is undefined, its RMSE is not.
    _stub_baseline_fits(
        monkeypatch,
        {"seed_x": np.array([0.2, 0.5, 0.9]), "seed_y": np.array([0.5, 0.5, 0.5])},
        elpd={"seed_x": -10.0, "seed_y": -5.0},
    )
    out = _fitted_baseline(run_root)
    assert out["mean_r_models"] == ["seed_x"]
    assert out["mean_rmse_models"] == ["seed_x", "seed_y"]


# ── the CLIs print the accounting ────────────────────────────────────────


def _cell(root: Path, label: str, rows, fitted_by_experiment, *, csv_rows=True):
    cell = root / label
    cell.mkdir(parents=True)
    result = _result(rows, fitted_by_experiment, gt=label.split("/")[1])
    (cell / "holdout.json").write_text(json.dumps(result), encoding="utf-8")
    if csv_rows:
        columns = ["gt_model", "experiment", "step", "iteration", "global_step",
                   "best_model", "pearson_r", "rmse", "kl_regret", "pearson_r_bma"]
        lines = [",".join(columns)]
        for row in result["gt_runs"][0]["trajectory"]:
            values = {**row, "gt_model": label.split("/")[1], "best_model": "m",
                      "kl_regret": 0.1, "iteration": "" if row["iteration"] is None else row["iteration"]}
            lines.append(",".join(str(values[c]) for c in columns))
        (cell / "holdout.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cli_sweep(tmp_path: Path) -> Path:
    root = tmp_path / "sweep"
    rows = CELL_A["gt_runs"][0]["trajectory"]
    _cell(root, "run1/gt_a", rows, _fitted([0.15, 0.35]))
    _cell(root, "run2/gt_a", rows, _fitted([0.15, None]))  # no trusted seed at the end
    (root / "run1" / "gt_b").mkdir(parents=True)  # partial
    return root


def test_the_test_retest_summary_lists_cells_and_compares_on_common_cells(tmp_path):
    from tests.paths import REPO_ROOT, load_script_module

    module = load_script_module(REPO_ROOT / "scripts/subjective_randomness/holdout_test_retest.py")
    root = _cli_sweep(tmp_path)
    out = tmp_path / "tr.json"
    module.main(module.Args(runs_root=root, out=out, n_repeats=2, gt_models="gt_a gt_b"))
    summary = json.loads(out.read_text())
    assert summary["cells"]["partial"] == {"run1/gt_b": pytest.approx(summary["cells"]["partial"]["run1/gt_b"])}
    assert set(summary["cells"]["missing"]) == {"run2/gt_b"}
    ends = {(e["gt_model"], e["experiment"]): e for e in summary["loop_vs_fitted_baseline"]}
    assert ends[("gt_a", 1)]["n_cells"] == 2
    assert ends[("gt_a", 2)]["n_cells"] == 1
    assert ends[("gt_a", 2)]["excluded"] == {"run2/gt_a": pytest.approx(ends[("gt_a", 2)]["excluded"]["run2/gt_a"])}


def test_the_incumbent_report_reads_only_complete_cells(tmp_path, monkeypatch):
    from tests.paths import REPO_ROOT, load_script_module

    module = load_script_module(REPO_ROOT / "scripts/subjective_randomness/incumbent_report.py")
    root = _cli_sweep(tmp_path)
    read = []
    monkeypatch.setattr(
        module, "cell_histories",
        lambda cell_dir: read.append(cell_dir.relative_to(root).as_posix()) or [
            [{"step": 0, "iteration": None, "best_model": "s", "posteriors": {"s": 1.0}}]
        ],
    )
    out = tmp_path / "inc.md"
    module.main(module.Args(sweep=root, out=out, n_repeats=2, gt_models="gt_a gt_b"))
    assert read == ["run1/gt_a", "run2/gt_a"]
    assert "`run1/gt_b` (partial)" in out.read_text()
