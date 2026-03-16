"""Tests for expected information gain (EIG) computation."""

import math
import pytest
from pathlib import Path
from unittest.mock import patch

from src.agents.experiment_designer import expected_information_gain

RESPONSE_OPTIONS = ["left", "right"]


def _mock_predictions(model_name_to_p_left):
    """Return a get_model_predictions-like dict: model_name -> {'left': p, 'right': 1-p}."""

    def get_model_predictions(stimulus, response_options, model_names, theorist_dir):
        return {
            m: {"left": model_name_to_p_left[m], "right": 1.0 - model_name_to_p_left[m]}
            for m in model_names
            if m in model_name_to_p_left
        }

    return get_model_predictions


def test_eig_bounds_two_models():
    """EIG is in [0, log2(2)] = [0, 1] for two models."""
    stimulus = ("H", "T")
    model_names = ["m1", "m2"]
    with patch("src.agents.experiment_designer.get_model_predictions", side_effect=_mock_predictions({"m1": 0.9, "m2": 0.1})):
        eig = expected_information_gain(stimulus, model_names, theorist_dir=None)
    assert eig >= 0.0
    assert eig <= 1.0 + 1e-9  # log2(2) = 1


def test_eig_bounds_three_models():
    """EIG is in [0, log2(3)] for three models."""
    stimulus = ("HH", "TT")
    model_names = ["m1", "m2", "m3"]
    with patch("src.agents.experiment_designer.get_model_predictions", side_effect=_mock_predictions({"m1": 1.0, "m2": 0.5, "m3": 0.0})):
        eig = expected_information_gain(stimulus, model_names, theorist_dir=None)
    assert eig >= 0.0
    assert eig <= math.log2(3) + 1e-9


def test_eig_zero_when_all_models_agree():
    """When every model has the same P(left), EIG = 0 (no discrimination)."""
    stimulus = ("HTHT", "THTH")
    model_names = ["m1", "m2", "m3"]
    # All models say P(left) = 0.7
    with patch("src.agents.experiment_designer.get_model_predictions", side_effect=_mock_predictions({"m1": 0.7, "m2": 0.7, "m3": 0.7})):
        eig = expected_information_gain(stimulus, model_names, theorist_dir=None)
    assert eig == 0.0


def test_eig_two_model_analytic():
    """Two models: m1 always says left, m2 always says right => EIG = 1 bit."""
    stimulus = ("HH", "TT")
    model_names = ["m1", "m2"]
    with patch("src.agents.experiment_designer.get_model_predictions", side_effect=_mock_predictions({"m1": 1.0, "m2": 0.0})):
        eig = expected_information_gain(stimulus, model_names, theorist_dir=None)
    # p_left = 0.5, p_right = 0.5; H(M)=1; P(m1|left)=1, P(m2|right)=1 => H(M|R)=0 => EIG=1
    assert abs(eig - 1.0) < 1e-9


def test_eig_zero_when_single_model_prior():
    """When prior puts all mass on one model, EIG = 0 (no uncertainty to reduce)."""
    stimulus = ("HTHTHT", "HHHTTT")
    model_names = ["m1", "m2"]
    model_weights = {"m1": 1.0, "m2": 0.0}
    with patch("src.agents.experiment_designer.get_model_predictions", side_effect=_mock_predictions({"m1": 0.8, "m2": 0.2})):
        eig = expected_information_gain(stimulus, model_names, theorist_dir=None, model_weights=model_weights)
    assert eig == 0.0


def test_eig_positive_for_discriminating_stimulus():
    """EIG is positive when models disagree (using real MODEL_LIBRARY)."""
    model_names = ["bayesian_fair_coin", "representativeness", "alternation"]
    stimulus = ("HTHTHTHT", "HHHHHHHH")
    eig = expected_information_gain(stimulus, model_names, theorist_dir=None)
    assert eig >= 0.0
    assert eig <= math.log2(len(model_names)) + 1e-9
    assert eig > 0.01, "EIG should be positive when models differ"
