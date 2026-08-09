"""Shared probability contracts fail loudly at every model boundary."""

from __future__ import annotations

import numpy as np
import pytest

from src.models.probability import (
    validate_probability_array,
    validate_probability_distribution,
)


def test_valid_distribution_is_returned_as_floats():
    assert validate_probability_distribution(
        {"left": 0.25, "right": 0.75}, ["left", "right"], context="model m"
    ) == {"left": 0.25, "right": 0.75}


@pytest.mark.parametrize(
    "distribution",
    [
        {"left": 0.5},
        {"left": 0.5, "right": 0.5, "abstain": 0.0},
        {"left": -0.1, "right": 1.1},
        {"left": float("nan"), "right": float("nan")},
        {"left": True, "right": False},
        {"left": 0.2, "right": 0.2},
    ],
)
def test_invalid_distributions_raise(distribution):
    with pytest.raises((TypeError, ValueError), match="model m"):
        validate_probability_distribution(
            distribution, ["left", "right"], context="model m"
        )


@pytest.mark.parametrize(
    "values", [np.array([0.2, np.nan]), np.array([-0.1, 0.5]), np.array([1.1])]
)
def test_invalid_probability_arrays_raise(values):
    with pytest.raises(ValueError, match="p_left"):
        validate_probability_array(values, context="p_left")
