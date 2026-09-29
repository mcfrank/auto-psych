"""Tests for `pearson_r` (`src/subjective_randomness/recover.py`)."""

from __future__ import annotations

import pytest

from src.subjective_randomness.recover import pearson_r


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
