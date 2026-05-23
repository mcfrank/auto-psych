"""Tests for shared correlation utilities (pearson_r, model_data_correlations)."""

import pytest
from pathlib import Path
from unittest.mock import patch

from src.stats.correlations import pearson_r, model_data_correlations


def test_pearson_r_perfect_positive():
    """Perfect positive correlation => r = 1."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert abs(pearson_r(x, y) - 1.0) < 1e-9


def test_pearson_r_perfect_negative():
    """Perfect negative correlation => r = -1."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [10.0, 8.0, 6.0, 4.0, 2.0]
    assert abs(pearson_r(x, y) + 1.0) < 1e-9


def test_pearson_r_uncorrelated():
    """Uncorrelated (orthogonal) => r ≈ 0."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 1.0, 1.0, 1.0, 1.0]  # constant
    assert pearson_r(x, y) == 0.0


def test_pearson_r_constant_input():
    """Constant input => r = 0 (undefined)."""
    assert pearson_r([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) == 0.0
    assert pearson_r([1.0, 2.0], [1.0, 1.0]) == 0.0


def test_pearson_r_length_mismatch():
    """Length mismatch => 0."""
    assert pearson_r([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_pearson_r_too_short():
    """n < 2 => 0."""
    assert pearson_r([1.0], [2.0]) == 0.0


def test_model_data_correlations_empty_aggregate():
    """Empty aggregate => all correlations 0."""
    lines = ["sequence_a,sequence_b,chose_left_pct,n\n"]
    result = model_data_correlations(lines, ["m1", "m2"], None, ["left", "right"])
    assert result == {"m1": 0.0, "m2": 0.0}


def test_model_data_correlations_synthetic():
    """Synthetic aggregate and mocked predictions => consistent correlations."""
    # Three stimuli; we mock get_model_predictions so m1 predicts [1,0,1], m2 [0,1,0]; observed [1,0,1] => m1 r=1, m2 r=-1
    lines = [
        "sequence_a,sequence_b,chose_left_pct,n\n",
        "A,B,1.0,10\n",
        "C,D,0.0,10\n",
        "E,F,1.0,10\n",
    ]
    aggregate_lines = [l.strip() for l in lines]

    def mock_predictions(stimulus, response_options, model_names, theorist_dir):
        # Return fixed predictions per stimulus index (we don't have index, so use stimulus as key)
        key = stimulus
        if key == ("A", "B"):
            return {"m1": {"left": 1.0, "right": 0.0}, "m2": {"left": 0.0, "right": 1.0}}
        if key == ("C", "D"):
            return {"m1": {"left": 0.0, "right": 1.0}, "m2": {"left": 1.0, "right": 0.0}}
        if key == ("E", "F"):
            return {"m1": {"left": 1.0, "right": 0.0}, "m2": {"left": 0.0, "right": 1.0}}
        return {"m1": {"left": 0.5, "right": 0.5}, "m2": {"left": 0.5, "right": 0.5}}

    with patch("src.stats.correlations.get_model_predictions", side_effect=mock_predictions):
        result = model_data_correlations(aggregate_lines, ["m1", "m2"], None, ["left", "right"])

    # Observed = [1, 0, 1]. m1 preds = [1, 0, 1] => r = 1. m2 preds = [0, 1, 0] => r = -1.
    assert abs(result["m1"] - 1.0) < 1e-5
    assert abs(result["m2"] + 1.0) < 1e-5
