"""Smoke tests for hand-authored subjective-randomness model families."""

from src.subjective_randomness.model_families import (
    bayesian_diagnosticity,
    encoding_compressibility,
    prototype_similarity,
    statistical_inference,
)


MODEL_MODULES = [
    bayesian_diagnosticity,
    encoding_compressibility,
    prototype_similarity,
    statistical_inference,
]

TEST_STIMULI = [
    ("HHTHTTHT", "HTHTHTHT"),
    ("TTTTTTTT", "HHTHTTHT"),
    ("HTHTHTHT", "HHHHTTTT"),
]


def test_model_family_predictions_are_distributions():
    for module in MODEL_MODULES:
        for stimulus in TEST_STIMULI:
            result = module.predict(stimulus, ["left", "right"], module.DEFAULT_PARAMS)
            assert set(result) == {"left", "right"}
            assert 0.0 <= result["left"] <= 1.0
            assert 0.0 <= result["right"] <= 1.0
            assert abs(sum(result.values()) - 1.0) < 1e-9


def test_default_models_prefer_irregular_balanced_sequence_over_perfect_alternation():
    stimulus = ("HHTHTTHT", "HTHTHTHT")
    for module in MODEL_MODULES:
        assert module.predict_left(stimulus, module.DEFAULT_PARAMS) > 0.5
