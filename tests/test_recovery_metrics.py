"""Tests for src.subjective_randomness.recovery_metrics.

Analytic cases for kl_regret, bias, calibration (OLS of p on q), and rmse,
plus backward-compatibility tests for readers accepting legacy holdout.csv
without the new metric columns.
"""

from __future__ import annotations

import csv
import math
import sys

import numpy as np
import pytest
from pyprojroot import here

sys.path.insert(0, str(here()))

from src.subjective_randomness.recovery_metrics import (
    bias,
    calibration,
    kl_regret,
    rmse,
)


# ── kl_regret ──────────────────────────────────────────────


class TestKLRegret:
    def test_identical_gives_zero(self):
        q = [0.3, 0.5, 0.8]
        assert kl_regret(q, q) == pytest.approx(0.0)

    def test_symmetric_swap_gives_positive(self):
        q = [0.2, 0.8]
        p = [0.8, 0.2]
        assert kl_regret(q, p) > 0

    def test_exact_zero_and_one_are_clipped_and_finite(self):
        q = [0.0, 1.0, 0.5]
        p = [0.5, 0.5, 0.5]
        result = kl_regret(q, p)
        assert math.isfinite(result)
        assert result > 0

    def test_p_at_boundary_is_finite(self):
        q = [0.5, 0.5]
        p = [0.0, 1.0]
        result = kl_regret(q, p)
        assert math.isfinite(result)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="length"):
            kl_regret([0.3, 0.5], [0.5])


# ── bias ───────────────────────────────────────────────────


class TestBias:
    def test_identical_gives_zero(self):
        q = [0.3, 0.5, 0.8]
        assert bias(q, q) == pytest.approx(0.0)

    def test_overconfident(self):
        q = [0.5, 0.5]
        p = [0.6, 0.7]
        assert bias(q, p) == pytest.approx(0.15)

    def test_underconfident(self):
        q = [0.5, 0.5]
        p = [0.3, 0.4]
        assert bias(q, p) == pytest.approx(-0.15)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="length"):
            bias([0.3], [0.5, 0.5])


# ── calibration ────────────────────────────────────────────


class TestCalibration:
    def test_identical_gives_slope_1_intercept_0(self):
        q = [0.1, 0.5, 0.9]
        slope, intercept = calibration(q, q)
        assert slope == pytest.approx(1.0)
        assert intercept == pytest.approx(0.0)

    def test_anti_correlated_gives_slope_minus_1(self):
        q = [0.1, 0.5, 0.9]
        p = [0.9, 0.5, 0.1]
        slope, intercept = calibration(q, p)
        assert slope == pytest.approx(-1.0)
        assert intercept == pytest.approx(1.0)

    def test_constant_p_gives_zero_slope(self):
        q = [0.2, 0.5, 0.8]
        p = [0.5, 0.5, 0.5]
        slope, intercept = calibration(q, p)
        assert slope == pytest.approx(0.0)
        assert intercept == pytest.approx(0.5)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="length"):
            calibration([0.3], [0.5, 0.5])


# ── rmse ───────────────────────────────────────────────────


class TestRMSE:
    def test_identical_gives_zero(self):
        q = [0.3, 0.5, 0.8]
        assert rmse(q, q) == pytest.approx(0.0)

    def test_known_value(self):
        q = [0.0, 0.0]
        p = [1.0, 1.0]
        assert rmse(q, p) == pytest.approx(1.0)

    def test_matches_numpy(self):
        rng = np.random.default_rng(42)
        q = rng.uniform(0, 1, 100).tolist()
        p = rng.uniform(0, 1, 100).tolist()
        expected = float(np.sqrt(np.mean((np.array(q) - np.array(p)) ** 2)))
        assert rmse(q, p) == pytest.approx(expected)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="length"):
            rmse([0.3], [0.5, 0.5])


# ── Backward-compat: legacy CSV readers ───────────────────


def test_holdout_test_retest_reader_accepts_legacy_csv_without_new_columns(tmp_path):
    """holdout_test_retest's _final_rows_by_gt reports missing metrics as None."""
    from scripts.subjective_randomness.holdout_test_retest import _final_rows_by_gt

    legacy_csv = tmp_path / "holdout.csv"
    legacy_csv.write_text(
        "gt_model,experiment,step,iteration,global_step,best_model,pearson_r,rmse,pearson_r_bma,rmse_bma\n"
        "falk_konold_dp,1,0,,0,seed,0.70,0.15,0.72,0.14\n",
        encoding="utf-8",
    )
    rows_by_gt = _final_rows_by_gt(legacy_csv, "kl_regret")
    assert "falk_konold_dp" in rows_by_gt
    row = rows_by_gt["falk_konold_dp"]
    assert row.get("kl_regret") is None
    assert row["pearson_r"] == "0.70"


def test_holdout_test_retest_emits_rmse_and_kl_regret_summaries(tmp_path):
    """test_retest.json includes per_metric summaries for rmse and kl_regret."""
    from scripts.subjective_randomness.holdout_test_retest import Args, main

    runs_root = tmp_path / "runs"
    for r in (1, 2):
        cell = runs_root / f"run{r}" / "gt_a"
        cell.mkdir(parents=True)
        (cell / "holdout.csv").write_text(
            "gt_model,experiment,step,iteration,global_step,best_model,"
            "pearson_r,rmse,kl_regret,bias,calib_slope,calib_intercept,"
            "pearson_r_bma,rmse_bma,kl_regret_bma,bias_bma,"
            "calib_slope_bma,calib_intercept_bma\n"
            f"gt_a,1,0,,0,seed,0.{80+r},0.{10+r},0.0{r},0.01,1.0,0.0,"
            f"0.{80+r},0.{10+r},0.0{r},0.01,1.0,0.0\n",
            encoding="utf-8",
        )

    out_json = tmp_path / "test_retest.json"
    out_csv = tmp_path / "test_retest.csv"
    main(Args(
        runs_root=runs_root,
        out=out_json,
        csv=out_csv,
        metric="pearson_r",
    ))

    import json
    summary = json.loads(out_json.read_text(encoding="utf-8"))

    assert "per_metric" in summary
    assert "rmse" in summary["per_metric"]
    assert "kl_regret" in summary["per_metric"]
    rmse_summary = summary["per_metric"]["rmse"]["per_gt_model"]["gt_a"]
    assert rmse_summary["n_runs"] == 2
    assert rmse_summary["mean"] is not None

    with out_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        columns = reader.fieldnames
        assert "rmse" in columns
        assert "kl_regret" in columns
