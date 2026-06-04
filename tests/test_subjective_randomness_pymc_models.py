"""Smoke tests for subjective-randomness PyMC adapter models."""

from pathlib import Path

import pytest
import yaml

from cc_pipeline.projects.subjective_randomness.preprocess_data import featurize_stimulus
from src.models.loader import get_model_names_from_manifest
from src.models.pymc_inference import (
    load_pymc_model,
    make_stim_data,
    observed_response_data,
    pm_data_inputs,
    prior_predict_p_left,
)


MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "cc_pipeline"
    / "projects"
    / "subjective_randomness"
    / "pymc_model_families"
)

EXPECTED_INPUTS = {
    "prototype_similarity": {
        "imbalance_a", "imbalance_b", "p_alts_a", "p_alts_b", "chose_left",
    },
    "encoding_compressibility": {
        "max_run_norm_a", "max_run_norm_b",
        "periodicity_a", "periodicity_b",
        "imbalance_a", "imbalance_b",
        "chose_left",
    },
    "bayesian_diagnosticity": {
        "n_a", "h_a", "alts_a", "n_b", "h_b", "alts_b", "chose_left",
    },
}


def test_subjective_randomness_manifest_lists_loadable_pymc_models():
    manifest = yaml.safe_load((MODEL_DIR / "models_manifest.yaml").read_text())
    assert get_model_names_from_manifest(manifest, MODEL_DIR) == [
        "prototype_similarity",
        "encoding_compressibility",
        "bayesian_diagnosticity",
    ]


@pytest.mark.parametrize("model_name,expected_inputs", EXPECTED_INPUTS.items())
def test_subjective_randomness_pymc_models_load(model_name, expected_inputs):
    model = load_pymc_model(model_name, MODEL_DIR)
    assert observed_response_data(model) == "chose_left"
    assert set(pm_data_inputs(model)) == expected_inputs


def test_featurize_stimulus_adds_pymc_adapter_features():
    features = featurize_stimulus("HTHT", "HHHT")
    assert features["n_a"] == 4
    assert features["h_a"] == 2
    assert features["alts_a"] == 3
    assert features["max_run_a"] == 1
    assert features["p_alts_a"] == 1.0
    assert features["imbalance_a"] == 0.0
    assert features["max_run_norm_a"] == 0.0
    assert features["periodicity_a"] == 1.0

    assert features["imbalance_b"] == 0.5
    assert features["max_run_norm_b"] == pytest.approx(2.0 / 3.0)
    assert features["periodicity_b"] == 0.5


@pytest.mark.parametrize("model_name", EXPECTED_INPUTS)
def test_featurized_stimuli_fill_pymc_data_containers(model_name):
    model = load_pymc_model(model_name, MODEL_DIR)
    row = featurize_stimulus("HHTHTTHT", "HTHTHTHT")
    row["chose_left"] = 0
    stim_data = make_stim_data(model, [row])
    assert set(stim_data) == set(pm_data_inputs(model))


def test_subjective_randomness_pymc_models_sample_prior_predictive():
    row = featurize_stimulus("HHTHTTHT", "HTHTHTHT")
    row["chose_left"] = 0
    preds = prior_predict_p_left(list(EXPECTED_INPUTS), MODEL_DIR, row, n_samples=10)
    assert set(preds) == set(EXPECTED_INPUTS)
    for p_left in preds.values():
        assert 0.0 <= p_left <= 1.0
