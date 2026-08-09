"""Test that project ground-truth models run and return a probability distribution."""

import sys
from pathlib import Path

import pytest

# Add project dir so we can load ground_truth_models
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.models.theorist.predictions import Stimulus

RESPONSE_OPTIONS = ["left", "right"]
TEST_STIMULI: list[Stimulus] = [
    ("HHTHTTHT", "HTHTHTHT"),
    ("TTTTTTTT", "HHHHHHHH"),
    ("HTHTHTHT", "HHHTTTTT"),
]


def _get_ground_truth_models():
    """Load subjective_randomness ground-truth models for testing."""
    from src.models.project.ground_truth import get_ground_truth_models

    return get_ground_truth_models("subjective_randomness")


def test_loader_sees_the_live_ground_truth_registry():
    """The loader must read the assets the outer loop actually runs.

    It resolved against the root ``projects/`` tree, which held a stale two-model
    shadow of the registry; the outer loop meanwhile ran the four-model file under
    ``src/pipelines/outer_loop/projects/``. The hidden models are the tell.
    """
    names = set(_get_ground_truth_models())
    assert {"length_sensitive_alternation", "recency_weighted_alternation"} <= names


@pytest.mark.parametrize("model_name", list(_get_ground_truth_models().keys()))
def test_model_returns_dict(model_name):
    models = _get_ground_truth_models()
    model_fn = models[model_name]
    for stim in TEST_STIMULI:
        result = model_fn(stim, RESPONSE_OPTIONS)
        assert isinstance(result, dict), f"{model_name} did not return a dict"


@pytest.mark.parametrize("model_name", list(_get_ground_truth_models().keys()))
def test_model_has_response_keys(model_name):
    models = _get_ground_truth_models()
    model_fn = models[model_name]
    for stim in TEST_STIMULI:
        result = model_fn(stim, RESPONSE_OPTIONS)
        for k in RESPONSE_OPTIONS:
            assert k in result, f"{model_name} missing key {k}"


@pytest.mark.parametrize("model_name", list(_get_ground_truth_models().keys()))
def test_model_probabilities_sum_to_one(model_name):
    models = _get_ground_truth_models()
    model_fn = models[model_name]
    for stim in TEST_STIMULI:
        result = model_fn(stim, RESPONSE_OPTIONS)
        total = sum(result[k] for k in RESPONSE_OPTIONS)
        assert abs(total - 1.0) < 1e-5, f"{model_name} probabilities sum to {total}"
