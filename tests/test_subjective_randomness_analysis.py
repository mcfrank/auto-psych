"""Fast tests for summarizing/analyzing recovery results.

`parameter_recovery_summary` condenses a per-repeat recovery report into one
row per parameter (bias, RMSE, calibration). `model_recovery_summary` turns a
closed-ended confusion result into per-model and overall recovery metrics.
"""

from __future__ import annotations

import pytest

from src.subjective_randomness.analysis import (
    model_recovery_summary,
    parameter_recovery_summary,
)

# A PyMC parameter-recovery report (posterior with credible intervals). The two
# repeats straddle the true value of 4.0, so bias is zero and RMSE is 0.5.
PYMC_PARAM_REPORT = {
    "model": "demo",
    "true_params": {"beta": 4.0},
    "runs": [
        {"repeat": 0, "posterior": {"beta": {"mean": 4.5, "q025": 3.5, "q975": 5.5}}},
        {"repeat": 1, "posterior": {"beta": {"mean": 3.5, "q025": 2.5, "q975": 4.5}}},
    ],
}

# A legacy PyMC report whose posterior summaries carry only means (written
# before q025/q975 were recorded), so CI coverage is undefined.
NO_INTERVAL_PARAM_REPORT = {
    "model": "demo",
    "true_params": {"beta": 4.0},
    "runs": [
        {"repeat": 0, "posterior": {"beta": {"mean": 4.2}}},
        {"repeat": 1, "posterior": {"beta": {"mean": 3.8}}},
    ],
}

CONFUSION = {
    "seed_models": ["A", "B"],
    "generating": [
        {
            "generating_model": "A",
            "best_model": "A",
            "recovered_correct": True,
            "posteriors": {"A": 0.8, "B": 0.2},
            "elpd_loo": {"A": -10.0, "B": -12.0},
        },
        {
            "generating_model": "B",
            "best_model": "A",
            "recovered_correct": False,
            "posteriors": {"A": 0.6, "B": 0.4},
            "elpd_loo": {"A": -9.0, "B": -9.5},
        },
    ],
}

# Same idea but carrying the inner loop's `comparison` table (elpd_diff/dse),
# so the summary can judge whether a recovery is statistically *clear* rather
# than a coin-flip between near-tied models.
CONFUSION_WITH_COMPARISON = {
    "seed_models": ["A", "B", "C"],
    "generating": [
        {  # correct AND clear: A wins, runner-up B is 30 elpd behind (dse 5).
            "generating_model": "A",
            "best_model": "A",
            "recovered_correct": True,
            "posteriors": {"A": 0.95, "B": 0.04, "C": 0.01},
            "elpd_loo": {"A": -100.0, "B": -130.0, "C": -160.0},
            "comparison": {
                "A": {"rank": 0, "elpd_diff": 0.0, "dse": 0.0, "weight": 0.95},
                "B": {"rank": 1, "elpd_diff": 30.0, "dse": 5.0, "weight": 0.04},
                "C": {"rank": 2, "elpd_diff": 60.0, "dse": 8.0, "weight": 0.01},
            },
        },
        {  # mis-recovered AND tied: B wins but true A is only 0.9 behind (dse 1.4).
            "generating_model": "A",
            "best_model": "B",
            "recovered_correct": False,
            "posteriors": {"A": 0.45, "B": 0.55, "C": 0.0},
            "elpd_loo": {"B": -200.0, "A": -200.9, "C": -260.0},
            "comparison": {
                "B": {"rank": 0, "elpd_diff": 0.0, "dse": 0.0, "weight": 0.55},
                "A": {"rank": 1, "elpd_diff": 0.9, "dse": 1.4, "weight": 0.45},
                "C": {"rank": 2, "elpd_diff": 60.0, "dse": 9.0, "weight": 0.0},
            },
        },
    ],
}


# ── parameter recovery ──────────────────────────────────────────────


def test_parameter_recovery_summary_computes_bias_rmse_and_coverage():
    rows = parameter_recovery_summary(PYMC_PARAM_REPORT)

    assert len(rows) == 1
    beta = rows[0]
    assert beta["model"] == "demo"
    assert beta["parameter"] == "beta"
    assert beta["true_value"] == 4.0
    assert beta["mean_estimate"] == pytest.approx(4.0)
    assert beta["bias"] == pytest.approx(0.0)
    assert beta["rmse"] == pytest.approx(0.5)
    assert beta["n_repeats"] == 2
    # True value (4.0) lies inside both 95% intervals -> full coverage.
    assert beta["ci_coverage_95"] == pytest.approx(1.0)


def test_parameter_recovery_summary_report_without_intervals_has_no_coverage():
    rows = parameter_recovery_summary(NO_INTERVAL_PARAM_REPORT)

    beta = rows[0]
    assert beta["bias"] == pytest.approx(0.0)
    assert beta["rmse"] == pytest.approx(0.2)
    # No credible intervals recorded, so coverage is undefined (not 0%).
    assert beta["ci_coverage_95"] is None


# A sampled-truth report (the pymc_recover.py default): each run carries its
# own ground truth, so the summary can correlate truth with estimate.
SAMPLED_TRUTH_REPORT = {
    "model": "demo",
    "param_ranges": {"beta": [0.2, 12.0]},
    "runs": [
        {
            "repeat": 0,
            "true_params": {"beta": 0.2},
            "posterior": {"beta": {"mean": 0.3, "q025": 0.1, "q975": 0.5}},
        },
        {
            "repeat": 1,
            "true_params": {"beta": 0.5},
            "posterior": {"beta": {"mean": 0.4, "q025": 0.2, "q975": 0.6}},
        },
        {
            "repeat": 2,
            "true_params": {"beta": 0.8},
            "posterior": {"beta": {"mean": 0.9, "q025": 0.7, "q975": 1.1}},
        },
    ],
}

# A sampled-truth PyMC-shaped report: per-run truths plus credible intervals,
# so CI coverage must be judged against each run's own truth.
SAMPLED_PYMC_REPORT = {
    "model": "demo",
    "param_ranges": {"beta": [0.2, 12.0]},
    "runs": [
        {
            "repeat": 0,
            "true_params": {"beta": 0.2},
            "posterior": {"beta": {"mean": 0.25, "q025": 0.1, "q975": 0.3}},
        },
        {
            "repeat": 1,
            "true_params": {"beta": 0.8},
            "posterior": {"beta": {"mean": 0.6, "q025": 0.5, "q975": 0.7}},
        },
    ],
}


def test_parameter_recovery_summary_sampled_report_correlates_truth_and_estimate():
    rows = parameter_recovery_summary(SAMPLED_TRUTH_REPORT)

    assert len(rows) == 1
    beta = rows[0]
    # Truths (0.2, 0.5, 0.8) vs estimates (0.3, 0.4, 0.9): strong but imperfect.
    assert beta["pearson_r"] == pytest.approx(0.93326, abs=1e-4)
    # No single true value exists when truths vary across repeats.
    assert beta["true_value"] is None
    assert beta["n_repeats"] == 3


def test_parameter_recovery_summary_fixed_report_has_no_correlation():
    rows = parameter_recovery_summary(PYMC_PARAM_REPORT)
    # A constant truth has zero variance, so the correlation is undefined.
    assert rows[0]["pearson_r"] is None
    assert rows[0]["true_value"] == 4.0


def test_parameter_recovery_summary_constant_truth_with_float_noise_no_correlation():
    # Regression: a constant truth of 0.4 is not exactly representable, so
    # naive std() accumulates ~1e-17 of float noise and looks nonzero; the
    # correlation must still be reported as undefined, not as ~0.
    report = {
        "model": "demo",
        "true_params": {"alt_prior": 0.4},
        "runs": [
            {"repeat": i, "posterior": {"alt_prior": {"mean": 0.3 + 0.01 * i}}}
            for i in range(20)
        ],
    }
    rows = parameter_recovery_summary(report)
    assert rows[0]["pearson_r"] is None
    assert rows[0]["true_value"] == 0.4


def test_parameter_recovery_summary_coverage_uses_each_runs_own_truth():
    rows = parameter_recovery_summary(SAMPLED_PYMC_REPORT)
    beta = rows[0]
    # Run 0's interval [0.1, 0.3] covers its truth 0.2; run 1's [0.5, 0.7]
    # misses its truth 0.8 -> coverage 1/2.
    assert beta["ci_coverage_95"] == pytest.approx(0.5)


# ── model recovery ──────────────────────────────────────────────────


def test_model_recovery_summary_overall_metrics():
    summary = model_recovery_summary(CONFUSION)

    assert summary["n_models"] == 2
    # A recovered (posterior best = A); B mis-recovered (posterior best = A).
    assert summary["posterior_accuracy"] == pytest.approx(0.5)
    # By ELPD-LOO, A's best is A (correct) but B's best is also A (incorrect).
    assert summary["elpd_accuracy"] == pytest.approx(0.5)
    # Posterior mass on the true model, averaged over generating models.
    assert summary["mean_true_posterior"] == pytest.approx((0.8 + 0.4) / 2)


def test_model_recovery_summary_per_model_rows():
    summary = model_recovery_summary(CONFUSION)
    by_model = {r["generating_model"]: r for r in summary["per_model"]}

    a = by_model["A"]
    assert a["true_posterior"] == pytest.approx(0.8)
    assert a["best_by_posterior"] == "A"
    assert a["best_by_elpd"] == "A"
    assert a["correct_posterior"] is True
    assert a["correct_elpd"] is True

    b = by_model["B"]
    assert b["true_posterior"] == pytest.approx(0.4)
    assert b["best_by_posterior"] == "A"
    assert b["best_by_elpd"] == "A"
    assert b["correct_posterior"] is False
    assert b["correct_elpd"] is False


def test_model_recovery_summary_rejects_empty_generating():
    with pytest.raises(ValueError, match="no generating"):
        model_recovery_summary({"seed_models": [], "generating": []})


def test_model_recovery_summary_without_comparison_marks_distinguishability_none():
    # Older confusion files carry no `comparison` table; distinguishability is
    # then unknown, not silently assumed.
    summary = model_recovery_summary(CONFUSION)
    assert summary["has_comparison"] is False
    assert summary["clear_recovery_rate"] is None
    for row in summary["per_model"]:
        assert row["winner_distinguishable"] is None
        assert row["winner_margin"] is None


def test_model_recovery_summary_with_comparison_flags_clear_vs_tied():
    summary = model_recovery_summary(CONFUSION_WITH_COMPARISON)
    by_idx = summary["per_model"]

    clear = by_idx[0]  # A recovered, runner-up 30 elpd behind (dse 5)
    assert clear["winner_by_elpd"] == "A"
    assert clear["winner_margin"] == pytest.approx(30.0)
    assert clear["winner_margin_dse"] == pytest.approx(5.0)
    assert clear["winner_distinguishable"] is True  # 30 > 2*5
    assert clear["true_model_elpd_diff"] == pytest.approx(0.0)
    assert clear["recovery_clear"] is True

    tied = by_idx[1]  # B "won" but true A is 0.9 behind (dse 1.4) -> a tie
    assert tied["winner_by_elpd"] == "B"
    assert tied["winner_margin"] == pytest.approx(0.9)
    assert tied["winner_distinguishable"] is False  # 0.9 < 2*1.4
    assert tied["true_model_elpd_diff"] == pytest.approx(0.9)
    assert tied["true_model_dse"] == pytest.approx(1.4)
    assert tied["recovery_clear"] is False


def test_model_recovery_summary_with_comparison_overall_clear_rate():
    summary = model_recovery_summary(CONFUSION_WITH_COMPARISON)
    assert summary["has_comparison"] is True
    # Only the first generating model is both correct and clearly separated.
    assert summary["clear_recovery_rate"] == pytest.approx(0.5)
