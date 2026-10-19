"""Smoke tests for subjective-randomness PyMC adapter models."""

from pathlib import Path

import pytest

from src.models.model_manifest import read_loadable_model_names
from src.models.data_binding import make_stim_data
from src.models.model_loading import (
    load_pymc_model,
    observed_response_data,
    pm_data_inputs,
)
from src.models.pymc_inference import prior_predict_p_left
from tests.model_registry import faithful_model_names


MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "subjective_randomness"
    / "pymc_model_families"
)

EXPECTED_INPUTS = {
    "falk_konold_dp": {
        "rep_motifs_a",
        "alt_motifs_a",
        "rep_motifs_b",
        "alt_motifs_b",
        "chose_left",
    },
    # motif_stack does not read trial-aligned feature columns: it declares a
    # `prepare_observed` hook and binds a unique-sequence table (scored once per
    # DISTINCT sequence) plus per-trial gather indices. See
    # tests/test_motif_stack_unique_sequence_graph.py.
    "motif_stack": {
        "seq_len",
        "emission_mask",
        "mirror_symmetry",
        "complement_symmetry",
        "duplication",
        "idx_a",
        "idx_b",
        "chose_left",
    },
    "finite_experience_occurrence": {
        "n_a",
        "n_b",
        "occ_n20_a",
        "occ_n20_b",
        "chose_left",
    },
    "local_representativeness": {
        "multiscale_imbalance_a",
        "multiscale_imbalance_b",
        "p_alts_a",
        "p_alts_b",
        "periodicity_a",
        "periodicity_b",
        "chose_left",
    },
}


def test_subjective_randomness_manifest_lists_loadable_pymc_models():
    # Every model the manifest names must survive the loader's file check: the
    # theorist loader silently drops an entry whose `.py` is missing, which
    # would shrink the model set without a trace. *Which* models these are is
    # pinned once, in test_literature_faithful_pymc.py; here the manifest is
    # the reference.
    assert read_loadable_model_names(MODEL_DIR) == faithful_model_names()


@pytest.mark.parametrize("model_name,expected_inputs", EXPECTED_INPUTS.items())
def test_subjective_randomness_pymc_models_load(model_name, expected_inputs):
    model = load_pymc_model(model_name, MODEL_DIR)
    assert observed_response_data(model) == "chose_left"
    assert set(pm_data_inputs(model)) == expected_inputs


def _stimulus_row(sequence_a: str, sequence_b: str) -> dict:
    """A raw stimulus row, as every production caller builds it: the models
    compute their features from the sequences (``compute_features`` or
    ``prepare_observed``)."""
    return {"sequence_a": sequence_a, "sequence_b": sequence_b, "chose_left": 0}


@pytest.mark.parametrize("model_name", EXPECTED_INPUTS)
def test_raw_stimuli_fill_pymc_data_containers(model_name):
    model = load_pymc_model(model_name, MODEL_DIR)
    stim_data = make_stim_data(model, [_stimulus_row("HHTHTTHT", "HTHTHTHT")])
    assert set(stim_data) == set(pm_data_inputs(model))


def test_subjective_randomness_pymc_models_sample_prior_predictive():
    row = _stimulus_row("HHTHTTHT", "HTHTHTHT")
    preds = prior_predict_p_left(list(EXPECTED_INPUTS), MODEL_DIR, row, n_samples=10)
    assert set(preds) == set(EXPECTED_INPUTS)
    for p_left in preds.values():
        assert 0.0 <= p_left <= 1.0


def test_in_graph_assert_rejects_cross_length_data():
    """finite_experience_occurrence still guards same-length pairs in the graph."""
    row = _stimulus_row("HT", "HTHT")
    with pytest.raises(AssertionError, match="same-length"):
        prior_predict_p_left(
            ["finite_experience_occurrence"], MODEL_DIR, row, n_samples=1
        )


def test_motif_stack_rejects_cross_length_data_at_data_preparation_time():
    """motif_stack's guard moved out of the logp graph into `prepare_observed`,
    so it raises a ValueError naming the offending row instead of an opaque
    AssertionError re-checked on every logp evaluation."""
    row = _stimulus_row("HT", "HTHT")
    with pytest.raises(ValueError, match="same-length"):
        prior_predict_p_left(["motif_stack"], MODEL_DIR, row, n_samples=1)
