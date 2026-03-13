"""Test that each model in MODEL_LIBRARY runs and returns a probability distribution (score)."""

import pytest

from src.models.randomness import MODEL_LIBRARY, Stimulus

RESPONSE_OPTIONS = ["left", "right"]
TEST_STIMULI: list[Stimulus] = [
    ("HHTHTTHT", "HTHTHTHT"),
    ("TTTTTTTT", "HHHHHHHH"),
    ("HTHTHTHT", "HHHTTTTT"),
]


@pytest.mark.parametrize("model_name", list(MODEL_LIBRARY.keys()))
def test_model_returns_dict(model_name):
    model_fn = MODEL_LIBRARY[model_name]
    for stim in TEST_STIMULI:
        result = model_fn(stim, RESPONSE_OPTIONS)
        assert isinstance(result, dict), f"{model_name} did not return a dict"


@pytest.mark.parametrize("model_name", list(MODEL_LIBRARY.keys()))
def test_model_has_response_keys(model_name):
    model_fn = MODEL_LIBRARY[model_name]
    for stim in TEST_STIMULI:
        result = model_fn(stim, RESPONSE_OPTIONS)
        for k in RESPONSE_OPTIONS:
            assert k in result, f"{model_name} missing key {k}"


@pytest.mark.parametrize("model_name", list(MODEL_LIBRARY.keys()))
def test_model_probabilities_sum_to_one(model_name):
    model_fn = MODEL_LIBRARY[model_name]
    for stim in TEST_STIMULI:
        result = model_fn(stim, RESPONSE_OPTIONS)
        total = sum(result[k] for k in RESPONSE_OPTIONS)
        assert abs(total - 1.0) < 1e-5, f"{model_name} probabilities sum to {total}"
