"""Tests for shared correlation utilities (pearson_r, model_data_correlations)."""

import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.stats.correlations import model_data_correlations, pearson_r


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
    """Constant input => r = 0 (undefined)."""
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    y = [1.0, 1.0, 1.0, 1.0, 1.0]
    assert pearson_r(x, y) == 0.0


def test_pearson_r_constant_input():
    assert pearson_r([1.0, 1.0, 1.0], [2.0, 3.0, 4.0]) == 0.0
    assert pearson_r([1.0, 2.0], [1.0, 1.0]) == 0.0


def test_pearson_r_length_mismatch():
    assert pearson_r([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0


def test_pearson_r_too_short():
    assert pearson_r([1.0], [2.0]) == 0.0


def test_model_data_correlations_empty_models_returns_empty(tmp_path):
    csv_path = tmp_path / "responses.csv"
    csv_path.write_text("a,b,chose_left\n1,2,0\n")
    assert model_data_correlations([], tmp_path, csv_path) == {}


def test_model_data_correlations_with_mocked_fits(tmp_path):
    """Stub fit_models_cached + a fake FittedModel so we can assert the
    correlation logic without running MCMC."""
    csv_path = tmp_path / "responses.csv"
    rows = [
        {"x": "0", "chose_left": "1"},
        {"x": "0", "chose_left": "1"},  # stimulus 0: observed = 1.0
        {"x": "1", "chose_left": "0"},
        {"x": "1", "chose_left": "0"},  # stimulus 1: observed = 0.0
        {"x": "2", "chose_left": "1"},
        {"x": "2", "chose_left": "1"},  # stimulus 2: observed = 1.0
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["x", "chose_left"])
        w.writeheader()
        w.writerows(rows)

    import pymc as pm
    with pm.Model() as fake_model:
        x_data = pm.Data("x", np.zeros(1, dtype="int64"))
        y_data = pm.Data("chose_left", np.zeros(1, dtype="int64"))
        pm.Bernoulli("response", p=0.5, observed=y_data)

    class FakeFitted:
        def __init__(self, model, preds):
            self.model = model
            self._preds = preds

        def predict_p_left(self, stim_data, **kwargs):
            xs = list(stim_data["x"])
            return np.array([self._preds[int(v)] for v in xs])

    fake_m1 = FakeFitted(fake_model, {0: 1.0, 1: 0.0, 2: 1.0})  # matches observed → r=+1
    fake_m2 = FakeFitted(fake_model, {0: 0.0, 1: 1.0, 2: 0.0})  # anti-correlated → r=-1

    with patch(
        "src.models.pymc_inference.fit_models_cached",
        return_value={"m1": fake_m1, "m2": fake_m2},
    ):
        result = model_data_correlations(["m1", "m2"], tmp_path, csv_path)

    assert abs(result["m1"] - 1.0) < 1e-5
    assert abs(result["m2"] + 1.0) < 1e-5
