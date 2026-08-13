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
``p_left`` / ``chose_left`` contract, and agree with its pure-Python twin.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.models.theorist.loader import get_model_names_from_manifest
from src.models.pymc_inference import (
    load_pymc_model,
    observed_response_data,
    prior_predict_p_left,
)
from src.subjective_randomness.features import featurize_stimulus
from src.subjective_randomness.model_recovery import (
    p_left_fixed_params,
    p_left_model_family,
)

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
    manifest = yaml.safe_load((MODEL_DIR / "models_manifest.yaml").read_text())
    assert get_model_names_from_manifest(manifest, MODEL_DIR) == NEW_MODELS


@pytest.mark.parametrize("model_name", NEW_MODELS)
def test_new_models_load_and_prior_predict(model_name):
    model = load_pymc_model(model_name, MODEL_DIR)
    assert observed_response_data(model) == "chose_left"

    # The raw sequences travel with the numeric features, as every production
    # caller builds them (cf. `model_recovery.feature_rows`): motif_stack derives
    # its pm.Data containers from the sequences via a `prepare_observed` hook.
    row = {
        "sequence_a": "HHTHTTHT",
        "sequence_b": "HTHTHTHT",
        **featurize_stimulus("HHTHTTHT", "HTHTHTHT"),
        "chose_left": 0,
    }
    preds = prior_predict_p_left([model_name], MODEL_DIR, row, n_samples=10)
    assert 0.0 <= preds[model_name] <= 1.0


@pytest.mark.parametrize("model_name", NEW_MODELS)
def test_new_models_match_their_pure_python_twin(model_name):
    import importlib

    family = importlib.import_module(
        f"src.subjective_randomness.model_families.{model_name}"
    )
    params = dict(family.DEFAULT_PARAMS)
    from_family = p_left_model_family(model_name, TEST_STIMULI, params)
    from_pymc = p_left_fixed_params(model_name, MODEL_DIR, TEST_STIMULI, params)
    np.testing.assert_allclose(from_pymc, from_family, atol=1e-7)
    # The defaults must not sit in a degenerate corner: agreement between
    # constant-0/1 outputs would be vacuous.
    assert np.all(from_family > 1e-4) and np.all(from_family < 1.0 - 1e-4)
