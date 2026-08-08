"""Smoke tests for subjective-randomness PyMC adapter models."""

from pathlib import Path

import pytest
import yaml

from src.subjective_randomness.features import featurize_stimulus
from src.models.theorist.loader import get_model_names_from_manifest
from src.models.pymc_inference import (
    load_pymc_model,
    make_stim_data,
    observed_response_data,
    pm_data_inputs,
    prior_predict_p_left,
)


MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "subjective_randomness"
    / "pymc_model_families"
)

EXPECTED_INPUTS = {
    "prototype_similarity": {
        "imbalance_a",
        "imbalance_b",
        "p_alts_a",
        "p_alts_b",
        "chose_left",
    },
    "encoding_compressibility": {
        "max_run_norm_a",
        "max_run_norm_b",
        "periodicity_a",
        "periodicity_b",
        "imbalance_a",
        "imbalance_b",
        "chose_left",
    },
    "bayesian_diagnosticity": {
        "n_a",
        "h_a",
        "rep_motifs_a",
        "alt_motifs_a",
        "n_b",
        "h_b",
        "rep_motifs_b",
        "alt_motifs_b",
        "chose_left",
    },
    "window_typicality": {
        "n_a",
        "n_b",
        "max_run_a",
        "max_run_b",
        "chose_left",
    },
    "falk_konold_dp": {
        "rep_motifs_a",
        "alt_motifs_a",
        "rep_motifs_b",
        "alt_motifs_b",
        "chose_left",
    },
    "motif_hmm": {
        "n_a",
        "n_b",
        *{f"sym{i}_{side}" for i in range(1, 9) for side in ("a", "b")},
        "chose_left",
    },
    "finite_experience_occurrence": {
        *{f"occ_n{w}_{side}" for w in (10, 20, 50) for side in ("a", "b")},
        "chose_left",
    },
    "local_representativeness": {
        "local_imbalance_a",
        "local_imbalance_b",
        "p_alts_a",
        "p_alts_b",
        "chose_left",
    },
}


def test_subjective_randomness_manifest_lists_loadable_pymc_models():
    # Active registry = the literature-faithful set only (2026-08
    # consolidation). The superseded originals remain in EXPECTED_INPUTS below
    # because their files must stay loadable for archival refits.
    manifest = yaml.safe_load((MODEL_DIR / "models_manifest.yaml").read_text())
    assert get_model_names_from_manifest(manifest, MODEL_DIR) == [
        "falk_konold_dp",
        "motif_hmm",
        "finite_experience_occurrence",
        "local_representativeness",
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
