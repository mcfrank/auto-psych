"""Tests for combining holdout-recovery runs across repeats with error bars.

Each finished ``holdout.json`` holds one ground truth's trajectory for a single
run. ``aggregate_holdout_trajectories`` pools the matching ground truths across
runs and reduces each inner-loop step (and each flat seed-model baseline) to a
mean and a spread (``scripts/subjective_randomness/plot_holdout_curves.py``
draws it).
"""

from __future__ import annotations

import pytest

from src.subjective_randomness.reporting import (
    aggregate_holdout_trajectories,
)


def _gt_run(
    gt_model: str, traj_rows, *, baseline, fitted_baseline, fitted_experiment_1=None
):
    run = {
        "gt_model": gt_model,
        "trajectory": list(traj_rows),
        "baseline": baseline,
        "fitted_baseline": fitted_baseline,
    }
    if fitted_experiment_1 is not None:
        # The fitted-seed baseline at the end of each experiment, on that
        # experiment's data; the final one is ``fitted_baseline``.
        run["fitted_baseline_by_experiment"] = [
            {"experiment": 1, **fitted_experiment_1},
            {"experiment": 2, **fitted_baseline},
        ]
    return run


# Two runs of the same ground truth. Values are chosen so the per-step means and
# sample standard deviations are exact and hand-checkable.
RUN_A = {
    "gt_runs": [
        _gt_run(
            "m",
            [
                {
                    "global_step": 0,
                    "step": 0,
                    "experiment": 1,
                    "pearson_r": 0.8,
                    "rmse": 0.2,
                    "pearson_r_bma": 0.7,
                    "rmse_bma": 0.3,
                },
                {
                    "global_step": 1,
                    "step": 1,
                    "experiment": 1,
                    "pearson_r": 0.9,
                    "rmse": 0.1,
                    "pearson_r_bma": 0.8,
                    "rmse_bma": 0.2,
                },
                # New outer experiment starts here (step 0, experiment 2). This
                # run's correlation is undefined (None) and must be skipped.
                {
                    "global_step": 2,
                    "step": 0,
                    "experiment": 2,
                    "pearson_r": None,
                    "rmse": 0.05,
                    "pearson_r_bma": None,
                    "rmse_bma": 0.05,
                },
            ],
            baseline={"mean_r": 0.5, "per_model": {"s1": 0.45, "s2": 0.55}},
            fitted_baseline={
                "mean_r": 0.6,
                "mean_rmse": 0.35,
                "per_model": {
                    "s1": {"pearson_r": 0.55, "rmse": 0.35},
                    "s2": {"pearson_r": 0.70, "rmse": 0.30},
                },
                # The ELPD-best seed is s1, though s2 is closer to the GT.
                "elpd_best_model": "s1",
                "elpd_best_r": 0.55,
                "elpd_best_rmse": 0.35,
            },
            fitted_experiment_1={"elpd_best_r": 0.5, "elpd_best_rmse": 0.4},
        )
    ]
}
RUN_B = {
    "gt_runs": [
        _gt_run(
            "m",
            [
                {
                    "global_step": 0,
                    "step": 0,
                    "experiment": 1,
                    "pearson_r": 0.6,
                    "rmse": 0.4,
                    "pearson_r_bma": 0.5,
                    "rmse_bma": 0.5,
                },
                {
                    "global_step": 1,
                    "step": 1,
                    "experiment": 1,
                    "pearson_r": 1.0,
                    "rmse": 0.0,
                    "pearson_r_bma": 0.9,
                    "rmse_bma": 0.1,
                },
                {
                    "global_step": 2,
                    "step": 0,
                    "experiment": 2,
                    "pearson_r": 0.5,
                    "rmse": 0.05,
                    "pearson_r_bma": 0.5,
                    "rmse_bma": 0.05,
                },
            ],
            baseline={"mean_r": 0.3, "per_model": {"s1": 0.35, "s2": 0.45}},
            fitted_baseline={
                "mean_r": 0.4,
                "mean_rmse": 0.45,
                "per_model": {
                    "s1": {"pearson_r": 0.50, "rmse": 0.45},
                    "s2": {"pearson_r": 0.65, "rmse": 0.40},
                },
                "elpd_best_model": "s2",
                "elpd_best_r": 0.65,
                "elpd_best_rmse": 0.40,
            },
            fitted_experiment_1={"elpd_best_r": 0.6, "elpd_best_rmse": 0.45},
        )
    ]
}


def test_aggregate_means_and_std_per_step():
    agg = aggregate_holdout_trajectories(
        [RUN_A, RUN_B], metric="pearson_r", error="std"
    )
    assert agg["metric"] == "pearson_r"
    assert [m["gt_model"] for m in agg["gt_models"]] == ["m"]
    panel = agg["gt_models"][0]
    assert panel["n_runs"] == 2

    best = panel["best"]
    # Aligned by experiment: experiment 1's seed step and end, experiment 2's
    # end (its only step).
    assert [(p["x"], p["experiment"], p["label"]) for p in best] == [
        (0, 1, "seed"),
        (1, 1, "end"),
        (2, 2, "end"),
    ]
    assert best[0]["mean"] == pytest.approx(0.7)
    assert best[0]["err"] == pytest.approx(0.1414213562, abs=1e-6)  # stdev([0.8, 0.6])
    assert best[0]["n"] == 2
    # Position 2 has one undefined value (RUN_A): that cell is left out (of
    # the baselines too), not counted as 0.
    assert best[2]["mean"] == pytest.approx(0.5)
    assert best[2]["n"] == 1
    assert best[2]["err"] == pytest.approx(0.0)  # spread undefined for n < 2 -> 0

    bma = panel["bma"]
    assert bma[0]["mean"] == pytest.approx(0.6)  # mean([0.7, 0.5])


def test_aggregate_sem_halves_the_std_for_two_runs():
    agg = aggregate_holdout_trajectories(
        [RUN_A, RUN_B], metric="pearson_r", error="sem"
    )
    best0 = agg["gt_models"][0]["best"][0]
    # SEM = stdev / sqrt(n) = 0.14142 / sqrt(2) = 0.1
    assert best0["err"] == pytest.approx(0.1, abs=1e-6)


def test_aggregate_drops_nan_and_inf_like_none():
    # Impossible ground truths can make a metric NaN (constant prediction ->
    # undefined correlation) or unbounded; those points must be dropped from a
    # step's sample exactly like an explicit None, not crash statistics.stdev.
    nan, inf = float("nan"), float("inf")
    run_bad = {
        "gt_runs": [
            _gt_run(
                "m",
                [
                    {
                        "global_step": 0,
                        "step": 0,
                        "experiment": 1,
                        "pearson_r": 0.8,
                        "rmse": 0.2,
                        "pearson_r_bma": 0.8,
                        "rmse_bma": 0.2,
                    },
                    {
                        "global_step": 1,
                        "step": 1,
                        "experiment": 1,
                        "pearson_r": nan,
                        "rmse": inf,
                        "pearson_r_bma": nan,
                        "rmse_bma": inf,
                    },
                ],
                baseline={"mean_r": 0.3, "per_model": {"s1": nan, "s2": 0.45}},
                fitted_baseline={
                    "per_model": {"s1": {"pearson_r": 0.5, "rmse": 0.3}},
                    "elpd_best_r": 0.5,
                    "elpd_best_rmse": 0.3,
                },
            )
        ]
    }
    run_ok = {
        "gt_runs": [
            _gt_run(
                "m",
                [
                    {
                        "global_step": 0,
                        "step": 0,
                        "experiment": 1,
                        "pearson_r": 0.6,
                        "rmse": 0.4,
                        "pearson_r_bma": 0.6,
                        "rmse_bma": 0.4,
                    },
                    {
                        "global_step": 1,
                        "step": 1,
                        "experiment": 1,
                        "pearson_r": 0.9,
                        "rmse": 0.1,
                        "pearson_r_bma": 0.9,
                        "rmse_bma": 0.1,
                    },
                ],
                baseline={"mean_r": 0.3, "per_model": {"s1": 0.35, "s2": 0.45}},
                fitted_baseline={
                    "per_model": {"s1": {"pearson_r": 0.5, "rmse": 0.3}},
                    "elpd_best_r": 0.5,
                    "elpd_best_rmse": 0.3,
                },
            )
        ]
    }
    agg = aggregate_holdout_trajectories(
        [run_bad, run_ok], metric="pearson_r", error="std"
    )
    best = {p["x"]: p for p in agg["gt_models"][0]["best"]}
    # Step 0 averages both finite values; step 1 keeps only the finite run.
    assert best[0]["mean"] == pytest.approx(0.7) and best[0]["n"] == 2
    assert best[1]["mean"] == pytest.approx(0.9) and best[1]["n"] == 1
    # The NaN sibling-seed value is dropped from the default-param baseline.
    baseline = agg["gt_models"][0]["baselines"]["baseline"]
    assert baseline["mean"] == pytest.approx(0.45)  # best of {NaN, 0.45} and {0.45}


def test_aggregate_baselines_use_the_elpd_best_seed_not_the_oracle_best():
    # The fitted baseline is each run's ELPD-best seed (chosen on training
    # data), pooled across runs — not the seed closest to the GT on the eval
    # pool. RMSE: fitted per run = [0.35 (s1), 0.40 (s2)] -> 0.375.
    agg = aggregate_holdout_trajectories([RUN_A, RUN_B], metric="rmse", error="std")
    baselines = agg["gt_models"][0]["baselines"]
    assert baselines["fitted_baseline"]["mean"] == pytest.approx(0.375)
    assert baselines["fitted_baseline"]["n"] == 2
    # The default-param baseline only stores Pearson r, so it has no RMSE.
    assert baselines["baseline"] is None

    # Pearson r at the end of the final experiment: RUN_A's loop r is
    # undefined there, so only RUN_B counts, for the loop and every baseline
    # alike: fitted 0.65; the default-params baseline (no ELPD) is still the
    # max-r sibling, 0.45.
    agg_r = aggregate_holdout_trajectories([RUN_A, RUN_B], metric="pearson_r")
    br = agg_r["gt_models"][0]["baselines"]
    assert br["fitted_baseline"]["mean"] == pytest.approx(0.65)
    assert br["fitted_baseline"]["n"] == 1
    assert br["baseline"]["mean"] == pytest.approx(0.45)


def test_aggregate_marks_outer_experiment_boundaries():
    agg = aggregate_holdout_trajectories([RUN_A, RUN_B], metric="pearson_r")
    assert agg["gt_models"][0]["experiment_boundaries"] == [2]


def test_aggregate_rejects_unknown_metric():
    with pytest.raises(ValueError, match="metric"):
        aggregate_holdout_trajectories([RUN_A], metric="nonsense")
