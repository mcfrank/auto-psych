"""Tests for the combined impossible-ground-truth recovery figure CLI.

The impossible test-retest sweep has the same on-disk shape as the holdout one —
``<runs_root>/run<r>/<gt_model>/holdout.json`` — but the ground truths are models
outside the seed hypothesis space (e.g. ``more_heads_more_random``). The script
reuses the holdout pooling/plotting, which is nan/inf-robust, so a degenerate
point (a constant prediction makes the correlation undefined -> NaN) is dropped
rather than crashing.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts" / "analysis"

GT_MODELS = ["more_heads_more_random", "longer_runs_more_random"]


def _gt_run(gt_model, traj_rows, *, baseline, fitted_baseline):
    return {
        "gt_model": gt_model,
        "trajectory": list(traj_rows),
        "baseline": baseline,
        "fitted_baseline": fitted_baseline,
    }


def _result(gt_model, *, r1):
    """One run's holdout result for ``gt_model``; ``r1`` may be NaN (degenerate)."""
    nan = float("nan")
    rmse1 = (1.0 - r1) if r1 == r1 else nan  # NaN propagates when r1 is NaN
    return {
        "gt_runs": [
            _gt_run(
                gt_model,
                [
                    {"global_step": 0, "step": 0, "experiment": 1,
                     "pearson_r": 0.1, "rmse": 0.9, "pearson_r_bma": 0.1, "rmse_bma": 0.9},
                    {"global_step": 1, "step": 1, "experiment": 1,
                     "pearson_r": r1, "rmse": rmse1, "pearson_r_bma": r1, "rmse_bma": nan},
                ],
                baseline={"mean_r": 0.0, "per_model": {"s1": 0.05, "s2": -0.10}},
                fitted_baseline={
                    "per_model": {
                        "s1": {"pearson_r": 0.10, "rmse": 0.40},
                        "s2": {"pearson_r": 0.20, "rmse": 0.35},
                    }
                },
            )
        ]
    }


def _write_tree(root):
    # run1 reaches a finite step-1 value; run2's is NaN and must be dropped.
    for run_name, r1 in (("run1", 0.9), ("run2", float("nan"))):
        for gt in GT_MODELS:
            dest = root / run_name / gt
            dest.mkdir(parents=True)
            (dest / "holdout.json").write_text(
                json.dumps(_result(gt, r1=r1)), encoding="utf-8"
            )


def _load_cli():
    spec = importlib.util.spec_from_file_location(
        "plot_impossible_combined", SCRIPTS / "plot_impossible_combined.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


def test_default_runs_root_points_at_impossible_results():
    assert "impossible" in str(cli.Args().runs_root)


def test_fitted_baseline_label_drops_the_other():
    # No seed is held out for impossible ground truths, so the baseline is just
    # the best seed model (not the best *other* seed model as in holdout recovery).
    assert cli.FITTED_BASELINE_LABEL == "best seed model"


def test_cli_pools_impossible_runs_into_figures(tmp_path):
    runs_root = tmp_path / "impossible_holdout_test_retest"
    _write_tree(runs_root)

    out_dir = tmp_path / "figs"
    cli.main(cli.Args(runs_root=runs_root, out_dir=out_dir, metric="both"))

    assert (out_dir / "impossible_combined_rmse.pdf").exists()
    assert (out_dir / "impossible_combined_pearson_r.pdf").exists()
    csv_path = out_dir / "impossible_combined.csv"
    assert csv_path.exists()
    text = csv_path.read_text(encoding="utf-8")
    # Both impossible ground truths are pooled into the CSV.
    assert "more_heads_more_random" in text and "longer_runs_more_random" in text


def test_cli_fails_loudly_when_no_runs_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="holdout.json"):
        cli.main(cli.Args(runs_root=tmp_path / "empty", out_dir=tmp_path / "out"))
