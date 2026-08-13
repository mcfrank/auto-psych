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
from tests.model_registry import faithful_model_names


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
    # the reference. The superseded originals remain in EXPECTED_INPUTS below
    # because their files must stay loadable for archival refits.
    manifest = yaml.safe_load((MODEL_DIR / "models_manifest.yaml").read_text())
    assert get_model_names_from_manifest(manifest, MODEL_DIR) == faithful_model_names()


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


def test_featurize_stimulus_marks_stack_automaton_patterns():
    mirror = featurize_stimulus("HHHTTHHH", "HHTTHHTT")
    complement = featurize_stimulus("TTTTHHHH", "HHTHTTHT")

    assert mirror["mirror_symmetry_a"] == 1
    assert mirror["duplication_a"] == 0
    assert mirror["duplication_b"] == 1
    assert complement["complement_symmetry_a"] == 1
    assert complement["mirror_symmetry_b"] == 0


def test_featurizer_exposes_multiscale_local_balance():
    features = featurize_stimulus("HHHHTTTT", "HHTTHHTT")

    assert features["multiscale_imbalance_a"] > features["multiscale_imbalance_b"]


def _stimulus_row(sequence_a: str, sequence_b: str) -> dict:
    """A feature row as every production caller builds it.

    The raw H/T sequences travel alongside the numeric features because models
    may declare a `compute_features` featurizer or a `prepare_observed` hook,
    both of which derive their inputs from the sequences rather than from the
    fixed feature columns (cf. `model_recovery.feature_rows`).
    """
    return {
        "sequence_a": sequence_a,
        "sequence_b": sequence_b,
        **featurize_stimulus(sequence_a, sequence_b),
        "chose_left": 0,
    }


@pytest.mark.parametrize("model_name", EXPECTED_INPUTS)
def test_featurized_stimuli_fill_pymc_data_containers(model_name):
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
