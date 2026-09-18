"""Tests for the compare_matched_cells CLI.

The CLI re-scores two sweep roots on a common held-out pool (exhaustive minus
the union of both cells' trained pairs) and writes per-cell-pair deltas.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

import src.subjective_randomness.holdout_eval as holdout_eval
import src.subjective_randomness.holdout_recovery as holdout_recovery
from src.subjective_randomness.holdout_eval import (
    _unordered_pair,
    collect_trained_pairs,
)


def _make_holdout_json(
    cell_dir: Path,
    gt_model: str,
    n_experiments: int,
    seed_models_dir: str,
    trained_pairs: list[tuple[str, str]],
) -> None:
    """Write a minimal holdout.json and the experiment tree it references."""
    run_root = cell_dir / "repo" / "_runs" / gt_model
    run_root.mkdir(parents=True)
    for exp_num in range(1, n_experiments + 1):
        data_dir = run_root / f"experiment{exp_num}" / "data"
        data_dir.mkdir(parents=True)
        loop_dir = run_root / f"experiment{exp_num}" / "model_loop"
        loop_dir.mkdir(parents=True)
        (loop_dir / "models").mkdir()
        (loop_dir / "responses.csv").write_text("chose_left\n1\n", encoding="utf-8")
        history = [
            {
                "step": 0,
                "iteration": None,
                "best_model": "seed_a",
                "posteriors": {"seed_a": 1.0},
                "elpd_loo": {"seed_a": -1.0},
            }
        ]
        (loop_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")
        with (data_dir / "responses.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sequence_a", "sequence_b", "chose_left"])
            for sa, sb in trained_pairs:
                writer.writerow([sa, sb, 1])

    result = {
        "project_id": "test",
        "seed_models_dir": seed_models_dir,
        "n_experiments": n_experiments,
        "n_participants": 1,
        "inner_loop": {"max_iterations": 1, "candidate_count": 1},
        "fit_kwargs": {},
        "seed": 42,
        "eval_pool": {
            "n_pairs": 0,
            "lengths": [3],
            "seed": 0,
            "min_remaining": 1,
            "exhaustive": True,
            "predict_max_draws": None,
        },
        "metrics_version": 2,
        "gt_runs": [
            {
                "gt_model": gt_model,
                "params": {"theta_alt": 0.65},
                "run_root": str(run_root),
                "n_eval_stimuli": 10,
                "n_eval_dropped": 0,
                "trajectory": [
                    {
                        "experiment": 1,
                        "step": 0,
                        "iteration": None,
                        "global_step": 0,
                        "best_model": "seed_a",
                        "pearson_r": 0.9,
                        "rmse": 0.1,
                        "kl_regret": 0.01,
                        "bias": 0.02,
                        "calib_slope": 1.0,
                        "calib_intercept": 0.0,
                        "pearson_r_bma": 0.9,
                        "rmse_bma": 0.1,
                        "kl_regret_bma": 0.01,
                        "bias_bma": 0.02,
                        "calib_slope_bma": 1.0,
                        "calib_intercept_bma": 0.0,
                    }
                ],
                "baseline": {"mean_r": 0.5, "per_model": {}},
                "fitted_baseline": {"mean_r": 0.6, "mean_rmse": 0.2, "per_model": {}, "n_responses": 10},
                "leakage": {},
                "experiments": [{"experiment": e, "manifest_models": ["seed_a"]} for e in range(1, n_experiments + 1)],
            }
        ],
    }
    (cell_dir / "holdout.json").write_text(json.dumps(result, indent=2), encoding="utf-8")


def test_compare_matched_cells_common_pool_excludes_union(tmp_path, monkeypatch):
    """The common pool excludes the union of both cells' trained pairs."""
    from scripts.subjective_randomness.compare_matched_cells import (
        _common_eval_pool,
    )

    sweep_a = tmp_path / "sweep_a"
    sweep_b = tmp_path / "sweep_b"
    gt = "gt_a"

    pairs_a = [("HHH", "TTT"), ("HHT", "TTH")]
    pairs_b = [("HHH", "TTT"), ("HTH", "THT")]

    cell_a = sweep_a / "run1" / gt
    cell_b = sweep_b / "run1" / gt
    _make_holdout_json(cell_a, gt, 1, "/fake", pairs_a)
    _make_holdout_json(cell_b, gt, 1, "/fake", pairs_b)

    result_a = json.loads((cell_a / "holdout.json").read_text(encoding="utf-8"))
    result_b = json.loads((cell_b / "holdout.json").read_text(encoding="utf-8"))

    pool = _common_eval_pool(result_a, result_b, gt)
    pool_pairs = {
        _unordered_pair(s["sequence_a"], s["sequence_b"]) for s in pool
    }
    union_trained = {_unordered_pair(*p) for p in pairs_a} | {_unordered_pair(*p) for p in pairs_b}
    assert pool_pairs & union_trained == set()


def _stub_eval_seams(monkeypatch, gt_p, pred):
    """Patch evaluation seams so no real models or manifests are needed."""
    monkeypatch.setattr(
        holdout_eval,
        "p_left_fixed_params",
        lambda model_name, models_dir, stimuli, params, **kw: gt_p[:len(stimuli)],
    )
    monkeypatch.setattr(
        holdout_eval, "make_stim_data", lambda model, rows: {"n": len(rows)}
    )
    monkeypatch.setattr(holdout_eval, "pm_data_inputs", lambda model: [])
    monkeypatch.setattr(
        holdout_eval, "seed_model_names",
        lambda pool_dir, *a, **kw: ["seed_a"],
    )

    class Fitted:
        model = None
        def predict_p_left(self, stim_data):
            return pred[:stim_data["n"]]

    monkeypatch.setattr(
        holdout_eval,
        "fit_model",
        lambda name, models_dir, responses_path, **kw: Fitted(),
    )

    monkeypatch.setattr(
        holdout_eval, "seed_baseline_correlation",
        lambda *a, **kw: {"mean_r": 0.5, "per_model": {}},
    )
    monkeypatch.setattr(
        holdout_eval, "fitted_seed_baseline_correlation",
        lambda *a, **kw: {"mean_r": 0.6, "mean_rmse": 0.2, "per_model": {}, "n_responses": 10},
    )


def test_compare_matched_cells_writes_paired_output(tmp_path, monkeypatch):
    """Full CLI test: paired.csv and paired.json are written with deltas."""
    from scripts.subjective_randomness.compare_matched_cells import Args, main

    sweep_a = tmp_path / "sweep_a"
    sweep_b = tmp_path / "sweep_b"
    gt = "gt_a"

    pairs_shared = [("HHH", "TTT")]
    for sweep in (sweep_a, sweep_b):
        cell = sweep / "run1" / gt
        _make_holdout_json(cell, gt, 1, str(tmp_path / "fake_seeds"), pairs_shared)

    fake_seeds = tmp_path / "fake_seeds"
    fake_seeds.mkdir()

    gt_p = np.array([0.3, 0.5, 0.7, 0.8])
    pred = np.array([0.35, 0.55, 0.65, 0.75])
    _stub_eval_seams(monkeypatch, gt_p, pred)

    out_dir = tmp_path / "out"
    main(Args(sweep_a=sweep_a, sweep_b=sweep_b, out=out_dir))

    assert (out_dir / "paired.json").exists()
    assert (out_dir / "paired.csv").exists()

    result = json.loads((out_dir / "paired.json").read_text(encoding="utf-8"))
    assert "pairs" in result
    assert len(result["pairs"]) >= 1

    pair = result["pairs"][0]
    assert "rmse_a" in pair
    assert "rmse_b" in pair
    assert "delta_rmse" in pair
    assert "kl_regret_a" in pair
    assert "pearson_r_a" in pair

    with (out_dir / "paired.csv").open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) >= 1
    assert "delta_rmse" in rows[0]

    per_gt = result["per_gt"]
    assert gt in per_gt
    assert "mean_delta_rmse" in per_gt[gt]
