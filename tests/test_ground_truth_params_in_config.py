"""The subjective-randomness configs carry the seeds' generating parameters.

They used to be the ``DEFAULT_PARAMS`` of pure-Python twins of the seed models
(``src/subjective_randomness/model_families``), read through a separate
pristine snapshot. The configs now give them (``seed_params``), like the
number-game and impossible configs' ``gt_models``; the values are pinned here,
so moving them changed no generated response.
"""

from __future__ import annotations

from src.subjective_randomness.config import load_config
from src.subjective_randomness.holdout_data import resolve_generating_params
from tests.paths import REPO_ROOT

CONFIGS = REPO_ROOT / "scripts" / "subjective_randomness" / "configs"
# The twins' DEFAULT_PARAMS as of 2026-10-18.
SEED_PARAMS = {
    "falk_konold_dp": {"beta": 1.0, "side_bias": 0.0},
    "motif_stack": {
        "delta": 0.5493, "alpha": 0.2073, "repetition_weight": 0.6839,
        "mirror_share": 0.4989, "complement_share": 0.3984, "beta": 1.0, "side_bias": 0.0,
    },
    "finite_experience_occurrence": {"beta": 4.0, "side_bias": 0.0},
    "local_representativeness": {
        "theta_alt": 0.65, "alt_weight": 0.75, "periodic_share": 0.45, "beta": 4.0, "side_bias": 0.0,
    },
}


def test_the_faithful_sweep_generates_at_the_pinned_parameters():
    config = load_config(CONFIGS / "holdout_recovery_faithful.yaml")
    assert resolve_generating_params(config["gt_models"], config["seed_params"]) == SEED_PARAMS


def test_the_impossible_sweep_scores_the_seeds_at_the_same_parameters():
    assert load_config(CONFIGS / "impossible_holdout_recovery.yaml")["seed_params"] == SEED_PARAMS


def test_a_ground_truth_with_no_parameters_anywhere_raises():
    import pytest

    with pytest.raises(KeyError, match="unknown_model"):
        resolve_generating_params({"unknown_model": None}, SEED_PARAMS)
