"""Unit tests for src/models/pymc_inference.py — fast, no MCMC."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from src.models import pymc_inference as pi

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pymc_models"


def test_load_pymc_model_returns_pm_model():
    import pymc as pm
    model = pi.load_pymc_model("bayesian_fair_coin", FIXTURE_DIR)
    assert isinstance(model, pm.Model)


def test_load_pymc_model_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        pi.load_pymc_model("does_not_exist", FIXTURE_DIR)


def test_pm_data_inputs_lists_all_data_containers():
    model = pi.load_pymc_model("bayesian_fair_coin", FIXTURE_DIR)
    names = pi.pm_data_inputs(model)
    assert set(names) == {"n_a", "h_a", "n_b", "h_b", "chose_left"}


def test_observed_response_data_identifies_y_via_graph():
    model = pi.load_pymc_model("bayesian_fair_coin", FIXTURE_DIR)
    assert pi.observed_response_data(model) == "chose_left"


def test_observed_response_data_works_for_second_fixture():
    model = pi.load_pymc_model("representativeness", FIXTURE_DIR)
    assert pi.observed_response_data(model) == "chose_left"


def test_extract_observed_pulls_columns_by_name_and_dtype(tmp_path):
    model = pi.load_pymc_model("bayesian_fair_coin", FIXTURE_DIR)
    csv_path = tmp_path / "responses.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_a", "h_a", "n_b", "h_b", "chose_left"])
        w.writeheader()
        w.writerow({"n_a": "10", "h_a": "5", "n_b": "10", "h_b": "5", "chose_left": "1"})
        w.writerow({"n_a": "12", "h_a": "8", "n_b": "8", "h_b": "4", "chose_left": "0"})

    observed = pi.extract_observed(csv_path, model)
    assert set(observed.keys()) == {"n_a", "h_a", "n_b", "h_b", "chose_left"}
    assert np.issubdtype(observed["n_a"].dtype, np.integer)
    assert observed["n_a"].tolist() == [10, 12]
    assert observed["chose_left"].tolist() == [1, 0]


def test_extract_observed_missing_column_raises(tmp_path):
    model = pi.load_pymc_model("bayesian_fair_coin", FIXTURE_DIR)
    csv_path = tmp_path / "responses.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["n_a", "chose_left"])  # missing h_a, n_b, h_b
        w.writeheader()
        w.writerow({"n_a": "10", "chose_left": "1"})
    with pytest.raises(ValueError, match="missing columns"):
        pi.extract_observed(csv_path, model)


def test_observed_response_data_zero_obs_rvs_raises():
    """A model with no observed RVs should fail loudly."""
    import pymc as pm
    with pm.Model() as bad:
        pm.Data("x", np.zeros(1, dtype="int64"))
        pm.Normal("z", mu=0, sigma=1)  # not observed
    with pytest.raises(ValueError, match="no observed RVs"):
        pi.observed_response_data(bad)


def test_observed_response_data_two_obs_rvs_raises():
    import pymc as pm
    with pm.Model() as bad:
        y1 = pm.Data("y1", np.zeros(1, dtype="int64"))
        y2 = pm.Data("y2", np.zeros(1, dtype="int64"))
        pm.Bernoulli("r1", p=0.5, observed=y1)
        pm.Bernoulli("r2", p=0.5, observed=y2)
    with pytest.raises(ValueError, match="expected exactly one"):
        pi.observed_response_data(bad)


def test_cache_key_changes_when_model_or_data_changes(tmp_path):
    """Two different responses CSVs must produce different cache keys."""
    csv1 = tmp_path / "a.csv"
    csv2 = tmp_path / "b.csv"
    csv1.write_text("col,col2\n1,2\n")
    csv2.write_text("col,col2\n3,4\n")
    k1 = pi._cache_key("bayesian_fair_coin", FIXTURE_DIR, csv1)
    k2 = pi._cache_key("bayesian_fair_coin", FIXTURE_DIR, csv2)
    assert k1 != k2


def test_prior_predict_p_left_returns_per_model_means():
    feature_row = {"n_a": 10, "h_a": 5, "n_b": 10, "h_b": 5, "chose_left": 0}
    pi.clear_model_cache()
    preds = pi.prior_predict_p_left(
        ["bayesian_fair_coin", "representativeness"], FIXTURE_DIR, feature_row, n_samples=50,
    )
    assert set(preds.keys()) == {"bayesian_fair_coin", "representativeness"}
    # Balanced stimulus → both models near 0.5 under their priors.
    for v in preds.values():
        assert 0.0 < v < 1.0
        assert abs(v - 0.5) < 0.2


def test_expected_information_gain_prior_pymc_nonneg():
    feature_row = {"n_a": 10, "h_a": 7, "n_b": 10, "h_b": 3, "chose_left": 0}
    pi.clear_model_cache()
    eig = pi.expected_information_gain_prior_pymc(
        feature_row, ["bayesian_fair_coin", "representativeness"], FIXTURE_DIR, n_samples=50,
    )
    assert eig >= 0.0
    assert eig <= 1.0  # EIG over 2 models is at most log2(2) = 1 bit
