"""Tests for the ground-truth sampling utilities (`src/subjective_randomness/recover.py`).

These utilities are shared by every Bayesian recovery path: the PyMC
parameter recovery (`pymc_recover.py`) and the grid-posterior
stimulus-selection comparison (`adaptive_recovery.py`). Sampled-truth mode
draws each repeat's ground-truth vector uniformly from the family's
`PARAM_BOUNDS`, optionally narrowed by a `param_ranges` config entry.
"""

from __future__ import annotations

import random

import pytest

from src.subjective_randomness.model_families import prototype_similarity
from src.subjective_randomness.recover import (
    pearson_r,
    resolve_param_ranges,
    sample_true_params,
    summarize_paired_recovery,
)


# ── pearson_r (public: consumed by holdout_recovery) ────────────────


def test_pearson_r_exact_value_on_known_data():
    # Hand-computable case: r((1,2,3), (2,4,5)) = 3/sqrt(2*4.6667) ≈ 0.981981.
    assert pearson_r([1.0, 2.0, 3.0], [2.0, 4.0, 5.0]) == pytest.approx(
        0.981981, abs=1e-6
    )


def test_pearson_r_is_signed():
    assert pearson_r([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]) == pytest.approx(-1.0)
    assert pearson_r([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) == pytest.approx(1.0)


def test_pearson_r_undefined_returns_none_not_zero():
    # The undefined cases must be None (not 0.0): n < 2, constant xs, constant
    # ys. A constant like 0.4 accumulates ~1e-17 float noise in the moment
    # sums, so "constant" is judged by distinct values.
    assert pearson_r([1.0], [2.0]) is None
    assert pearson_r([0.4, 0.4, 0.4], [1.0, 2.0, 3.0]) is None
    assert pearson_r([1.0, 2.0, 3.0], [0.4, 0.4, 0.4]) is None

# ── resolve_param_ranges ────────────────────────────────────────────


def test_resolve_param_ranges_defaults_to_model_bounds():
    ranges = resolve_param_ranges({}, prototype_similarity)
    assert ranges == {
        name: (float(lo), float(hi))
        for name, (lo, hi) in prototype_similarity.PARAM_BOUNDS.items()
    }


def test_resolve_param_ranges_applies_config_override():
    config = {"param_ranges": {"beta": [1.0, 6.0]}}
    ranges = resolve_param_ranges(config, prototype_similarity)
    assert ranges["beta"] == (1.0, 6.0)
    # Untouched parameters keep the family bounds.
    assert ranges["theta_alt"] == prototype_similarity.PARAM_BOUNDS["theta_alt"]


def test_resolve_param_ranges_rejects_unknown_parameter():
    config = {"param_ranges": {"not_a_param": [0.0, 1.0]}}
    with pytest.raises(ValueError, match="not_a_param"):
        resolve_param_ranges(config, prototype_similarity)


def test_resolve_param_ranges_rejects_range_outside_fit_bounds():
    # Truths the fit can never reach would silently wreck recovery.
    config = {"param_ranges": {"beta": [0.0, 50.0]}}
    with pytest.raises(ValueError, match="beta"):
        resolve_param_ranges(config, prototype_similarity)


def test_resolve_param_ranges_rejects_inverted_range():
    config = {"param_ranges": {"beta": [6.0, 1.0]}}
    with pytest.raises(ValueError, match="beta"):
        resolve_param_ranges(config, prototype_similarity)


@pytest.mark.parametrize(
    "bad_value",
    [
        5.0,  # a bare number, not a pair
        "[1.0, 6.0]",  # a string that merely looks like a pair
        [1.0, 2.0, 3.0],  # too many endpoints
        [1.0],  # too few endpoints
    ],
)
def test_resolve_param_ranges_rejects_non_pair_values(bad_value):
    config = {"param_ranges": {"beta": bad_value}}
    with pytest.raises(ValueError, match="low, high"):
        resolve_param_ranges(config, prototype_similarity)


# ── sample_true_params ──────────────────────────────────────────────


def test_sample_true_params_draws_within_ranges_deterministically():
    ranges = {"a": (0.0, 1.0), "b": (2.0, 5.0)}

    first = sample_true_params(ranges, random.Random(3))
    again = sample_true_params(ranges, random.Random(3))

    assert first == again  # same seed, same draw
    assert set(first) == {"a", "b"}
    assert 0.0 <= first["a"] <= 1.0
    assert 2.0 <= first["b"] <= 5.0


# ── summarize_paired_recovery ───────────────────────────────────────


def test_summarize_paired_recovery_computes_bias_rmse_and_pearson():
    truths = [{"a": 1.0}, {"a": 8.0}]
    estimates = [{"a": 1.2}, {"a": 7.5}]

    summary = summarize_paired_recovery(truths, estimates)

    entry = summary["a"]
    # Errors are +0.2 and -0.5.
    assert entry["bias"] == pytest.approx((0.2 - 0.5) / 2, abs=1e-6)
    assert entry["rmse"] == pytest.approx(((0.2**2 + 0.5**2) / 2) ** 0.5, abs=1e-6)
    assert entry["mean_estimate"] == pytest.approx(4.35, abs=1e-6)
    assert entry["min_estimate"] == 1.2
    assert entry["max_estimate"] == 7.5
    # Two points with positive slope correlate perfectly.
    assert entry["pearson_r"] == pytest.approx(1.0, abs=1e-6)
    # Truths vary, so there is no single true value to report.
    assert entry["true"] is None


def test_summarize_paired_recovery_constant_truth_has_no_correlation():
    truths = [{"a": 0.4}, {"a": 0.4}, {"a": 0.4}]
    estimates = [{"a": 0.3}, {"a": 0.4}, {"a": 0.5}]

    summary = summarize_paired_recovery(truths, estimates)

    # A constant truth has zero variance: correlation undefined, truth reported.
    assert summary["a"]["pearson_r"] is None
    assert summary["a"]["true"] == 0.4


def test_summarize_paired_recovery_single_pair_has_no_correlation():
    summary = summarize_paired_recovery([{"a": 1.0}], [{"a": 1.1}])
    assert summary["a"]["pearson_r"] is None  # n < 2


def test_summarize_paired_recovery_constant_estimates_have_no_correlation():
    truths = [{"a": 1.0}, {"a": 8.0}]
    estimates = [{"a": 3.0}, {"a": 3.0}]
    summary = summarize_paired_recovery(truths, estimates)
    assert summary["a"]["pearson_r"] is None  # zero estimate variance
