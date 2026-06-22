"""Tests for combining holdout-recovery runs across repeats with error bars.

Each finished ``holdout.json`` holds one ground truth's trajectory for a single
run. ``aggregate_holdout_trajectories`` pools the matching ground truths across
runs and reduces each inner-loop step (and each flat seed-model baseline) to a
mean and a spread, so ``plot_holdout_trajectories_combined`` can draw the same
per-model panels as the single-run figure but with error bars across runs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import plotnine
import pytest

from src.subjective_randomness.reporting import (
    aggregate_holdout_trajectories,
    holdout_combined_frames,
    holdout_trajectories_ggplot,
    plot_holdout_trajectories_combined,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts" / "analysis"


def _gt_run(gt_model: str, traj_rows, *, baseline, fitted_baseline):
    return {
        "gt_model": gt_model,
        "trajectory": list(traj_rows),
        "baseline": baseline,
        "fitted_baseline": fitted_baseline,
    }


# Two runs of the same ground truth. Values are chosen so the per-step means and
# sample standard deviations are exact and hand-checkable.
RUN_A = {
    "gt_runs": [
        _gt_run(
            "m",
            [
                {"global_step": 0, "step": 0, "experiment": 1,
                 "pearson_r": 0.8, "rmse": 0.2, "pearson_r_bma": 0.7, "rmse_bma": 0.3},
                {"global_step": 1, "step": 1, "experiment": 1,
                 "pearson_r": 0.9, "rmse": 0.1, "pearson_r_bma": 0.8, "rmse_bma": 0.2},
                # New outer experiment starts here (step 0, experiment 2). This
                # run's correlation is undefined (None) and must be skipped.
                {"global_step": 2, "step": 0, "experiment": 2,
                 "pearson_r": None, "rmse": 0.05, "pearson_r_bma": None, "rmse_bma": 0.05},
            ],
            baseline={"mean_r": 0.5, "per_model": {"s1": 0.45, "s2": 0.55}},
            fitted_baseline={
                "mean_r": 0.6,
                "mean_rmse": 0.35,
                "per_model": {
                    "s1": {"pearson_r": 0.55, "rmse": 0.35},
                    "s2": {"pearson_r": 0.70, "rmse": 0.30},
                },
            },
        )
    ]
}
RUN_B = {
    "gt_runs": [
        _gt_run(
            "m",
            [
                {"global_step": 0, "step": 0, "experiment": 1,
                 "pearson_r": 0.6, "rmse": 0.4, "pearson_r_bma": 0.5, "rmse_bma": 0.5},
                {"global_step": 1, "step": 1, "experiment": 1,
                 "pearson_r": 1.0, "rmse": 0.0, "pearson_r_bma": 0.9, "rmse_bma": 0.1},
                {"global_step": 2, "step": 0, "experiment": 2,
                 "pearson_r": 0.5, "rmse": 0.05, "pearson_r_bma": 0.5, "rmse_bma": 0.05},
            ],
            baseline={"mean_r": 0.3, "per_model": {"s1": 0.35, "s2": 0.45}},
            fitted_baseline={
                "mean_r": 0.4,
                "mean_rmse": 0.45,
                "per_model": {
                    "s1": {"pearson_r": 0.50, "rmse": 0.45},
                    "s2": {"pearson_r": 0.65, "rmse": 0.40},
                },
            },
        )
    ]
}


def test_aggregate_means_and_std_per_step():
    agg = aggregate_holdout_trajectories([RUN_A, RUN_B], metric="pearson_r", error="std")
    assert agg["metric"] == "pearson_r"
    assert [m["gt_model"] for m in agg["gt_models"]] == ["m"]
    panel = agg["gt_models"][0]
    assert panel["n_runs"] == 2

    best = panel["best"]
    assert [p["global_step"] for p in best] == [0, 1, 2]
    assert best[0]["mean"] == pytest.approx(0.7)
    assert best[0]["err"] == pytest.approx(0.1414213562, abs=1e-6)  # stdev([0.8, 0.6])
    assert best[0]["n"] == 2
    # Step 2 has one undefined value (RUN_A): it is skipped, not counted as 0.
    assert best[2]["mean"] == pytest.approx(0.5)
    assert best[2]["n"] == 1
    assert best[2]["err"] == pytest.approx(0.0)  # spread undefined for n < 2 -> 0

    bma = panel["bma"]
    assert bma[0]["mean"] == pytest.approx(0.6)  # mean([0.7, 0.5])


def test_aggregate_sem_halves_the_std_for_two_runs():
    agg = aggregate_holdout_trajectories([RUN_A, RUN_B], metric="pearson_r", error="sem")
    best0 = agg["gt_models"][0]["best"][0]
    # SEM = stdev / sqrt(n) = 0.14142 / sqrt(2) = 0.1
    assert best0["err"] == pytest.approx(0.1, abs=1e-6)


def test_aggregate_baselines_use_best_seed_not_mean():
    # The baseline is the BEST sibling seed per run, then pooled across runs.
    # RMSE: best = the min-RMSE sibling. fitted per run = [0.30, 0.40] -> 0.35.
    agg = aggregate_holdout_trajectories([RUN_A, RUN_B], metric="rmse", error="std")
    baselines = agg["gt_models"][0]["baselines"]
    assert baselines["fitted_baseline"]["mean"] == pytest.approx(0.35)
    assert baselines["fitted_baseline"]["n"] == 2
    # The default-param baseline only stores Pearson r, so it has no RMSE.
    assert baselines["baseline"] is None

    # Pearson r: best = the max-r sibling. fitted per run = [0.70, 0.65] -> 0.675;
    # default per run = [0.55, 0.45] -> 0.5.
    agg_r = aggregate_holdout_trajectories([RUN_A, RUN_B], metric="pearson_r")
    br = agg_r["gt_models"][0]["baselines"]
    assert br["fitted_baseline"]["mean"] == pytest.approx(0.675)
    assert br["baseline"]["mean"] == pytest.approx(0.5)


def test_aggregate_marks_outer_experiment_boundaries():
    agg = aggregate_holdout_trajectories([RUN_A, RUN_B], metric="pearson_r")
    assert agg["gt_models"][0]["experiment_boundaries"] == [2]


def test_aggregate_rejects_unknown_metric():
    with pytest.raises(ValueError, match="metric"):
        aggregate_holdout_trajectories([RUN_A], metric="nonsense")


def test_combined_frames_carry_error_band_bounds_for_plotnine():
    agg = aggregate_holdout_trajectories([RUN_A, RUN_B], metric="rmse", error="std")
    frames = holdout_combined_frames(agg)

    traj = frames["trajectory"]
    assert set(traj["series"].dropna().unique()) == {"best model"}
    step0 = traj[traj["global_step"] == 0].iloc[0]
    assert step0["mean"] == pytest.approx(0.3)  # mean([0.2, 0.4]) rmse
    # Error bars are drawn from explicit ymin/ymax = mean ± spread columns.
    assert step0["ymin"] == pytest.approx(0.3 - 0.1414213562, abs=1e-6)
    assert step0["ymax"] == pytest.approx(0.3 + 0.1414213562, abs=1e-6)

    baselines = frames["baselines"]
    fitted = baselines[baselines["series"] == "best seed (fit to all data)"].iloc[0]
    assert fitted["mean"] == pytest.approx(0.35)  # min-RMSE sibling per run: [0.30, 0.40]

    boundaries = frames["boundaries"]
    assert list(boundaries["boundary"]) == [2]
    # The dotted line is drawn half a step before the new experiment begins.
    assert boundaries.iloc[0]["x"] == pytest.approx(1.5)


def test_holdout_trajectories_ggplot_returns_a_ggplot_object():
    agg = aggregate_holdout_trajectories([RUN_A, RUN_B], metric="rmse")
    assert isinstance(holdout_trajectories_ggplot(agg), plotnine.ggplot)


def test_plot_combined_writes_a_figure(tmp_path):
    agg = aggregate_holdout_trajectories([RUN_A, RUN_B], metric="rmse")
    out = tmp_path / "combined_rmse.png"
    plot_holdout_trajectories_combined(agg, out)
    assert out.exists() and out.stat().st_size > 0


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "plot_holdout_combined", SCRIPTS / "plot_holdout_combined.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_cli_combines_run_tree_into_figures(tmp_path):
    runs_root = tmp_path / "holdout_test_retest"
    for run_name, result in (("run1", RUN_A), ("run2", RUN_B)):
        gt = result["gt_runs"][0]["gt_model"]
        dest = runs_root / run_name / gt
        dest.mkdir(parents=True)
        (dest / "holdout.json").write_text(json.dumps(result), encoding="utf-8")

    cli = _load_cli()
    out_dir = tmp_path / "figs"
    cli.main(cli.Args(runs_root=runs_root, out_dir=out_dir, metric="both"))

    assert (out_dir / "holdout_combined_rmse.png").exists()
    assert (out_dir / "holdout_combined_pearson_r.png").exists()
    assert (out_dir / "holdout_combined.csv").exists()
