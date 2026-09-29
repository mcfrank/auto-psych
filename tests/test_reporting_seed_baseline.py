"""The figures' fitted-seed baseline is the ELPD-best seed, not the oracle best.

The combined figure used to take, per cell, the seed with the best held-out
RMSE (or r) against the ground truth — a choice only an oracle can make. It now
plots the seed the loop itself would pick: the best by ELPD-LOO on the
training data (``elpd_best_*``, recorded by holdout_eval).
"""

from __future__ import annotations

from src.subjective_randomness.reporting import (
    _HOLDOUT_METRIC_SPECS as METRIC_SPECS,
    _best_seed_value,
)

FITTED = {
    "per_model": {
        "oracle_pick": {"pearson_r": 0.9, "rmse": 0.05},
        "elpd_pick": {"pearson_r": 0.6, "rmse": 0.12},
    },
    "elpd_best_model": "elpd_pick",
    "elpd_best_r": 0.6,
    "elpd_best_rmse": 0.12,
}


def test_the_fitted_baseline_is_the_elpd_best_seed():
    assert _best_seed_value(FITTED, "fitted_baseline", METRIC_SPECS["rmse"]) == 0.12
    assert _best_seed_value(FITTED, "fitted_baseline", METRIC_SPECS["pearson_r"]) == 0.6


def test_a_result_scored_before_the_elpd_best_field_has_no_fitted_baseline(capsys):
    old = {"per_model": FITTED["per_model"], "mean_r": 0.75, "mean_rmse": 0.085}
    assert _best_seed_value(old, "fitted_baseline", METRIC_SPECS["rmse"]) is None
    assert "no ELPD-best seed" in capsys.readouterr().err
