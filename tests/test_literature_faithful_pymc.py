"""Acceptance tests for the literature-faithful seed-model additions.

Driven by the 2026-08 seed-model fidelity review: the recovery registry gains
four models that implement their source theories exactly as published —

  * ``falk_konold_dp``: Falk & Konold (1997) Difficulty Predictor, minimal
    parse, unnormalised by length.
  * ``motif_stack``: Griffiths, Daniels, Austerweil & Tenenbaum (2018)
    four-motif stack automaton, maximised over paths and production methods.
  * ``finite_experience_occurrence``: Hahn & Warren (2009) probability of
    occurrence within a finite experienced global sequence.
  * ``local_representativeness``: Kahneman & Tversky (1972) local
    representativeness — multiscale balance plus explicit irregularity cues.

Each PyMC adapter must load from the registry manifest, expose the standard
``p_left`` / ``chose_left`` contract.
"""

from pathlib import Path

import numpy as np
import pytest

from src.models.model_manifest import read_loadable_model_names
from src.models.model_loading import load_pymc_model, observed_response_data
from src.models.pymc_inference import prior_predict_p_left
from src.subjective_randomness.holdout_data import p_left_fixed_params
from src.tasks.subjective_randomness import SUBJECTIVE_RANDOMNESS
from tests.test_ground_truth_params_in_config import SEED_PARAMS

MODEL_DIR = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "subjective_randomness"
    / "pymc_model_families"
)

NEW_MODELS = [
    "falk_konold_dp",
    "motif_stack",
    "finite_experience_occurrence",
    "local_representativeness",
]

TEST_STIMULI = [
    {"sequence_a": "HHTHTTHT", "sequence_b": "HTHTHTHT"},
    {"sequence_a": "TTTTTTTT", "sequence_b": "HHTHTTHT"},
    {"sequence_a": "HTHT", "sequence_b": "HHHT"},
    {"sequence_a": "HTH", "sequence_b": "THT"},
]


def test_registry_manifest_is_exactly_the_literature_faithful_set():
    # 2026-08 consolidation: each faithful model replaced its superseded
    # counterpart (falk_konold_dp <- encoding_compressibility, motif_stack <-
    # bayesian_diagnosticity, finite_experience_occurrence <-
    # window_typicality, local_representativeness <- prototype_similarity).
    assert read_loadable_model_names(MODEL_DIR) == NEW_MODELS


@pytest.mark.parametrize("model_name", NEW_MODELS)
def test_new_models_load_and_prior_predict(model_name):
    model = load_pymc_model(model_name, MODEL_DIR)
    assert observed_response_data(model) == "chose_left"

    # A raw row: every model computes its features from the sequences.
    row = {"sequence_a": "HHTHTTHT", "sequence_b": "HTHTHTHT", "chose_left": 0}
    preds = prior_predict_p_left([model_name], MODEL_DIR, row, n_samples=10)
    assert 0.0 <= preds[model_name] <= 1.0


@pytest.mark.parametrize("model_name", NEW_MODELS)
def test_the_generating_parameters_are_not_degenerate(model_name):
    # Ground-truth data come from these models at the config's seed_params;
    # p_left stuck at 0 or 1 would make recovery vacuous.
    params = SEED_PARAMS[model_name]
    p = p_left_fixed_params(model_name, MODEL_DIR, TEST_STIMULI, params, task=SUBJECTIVE_RANDOMNESS)
    assert np.all(p > 1e-4) and np.all(p < 1.0 - 1e-4)
