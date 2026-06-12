"""Tests for adaptive (sequential) EIG-driven recovery on the reference families.

Both loops keep an exact grid posterior over the pure-Python families, pick the
highest-EIG stimulus under the current posterior, simulate the true response,
and update — no MCMC. Tests assert the loop actually *learns*: posterior entropy
falls, estimates beat the uninformed prior, and a clearly-separable model is
recovered.
"""

from __future__ import annotations

import pytest

from src.subjective_randomness.adaptive_recovery import (
    run_adaptive_model_confusion,
    run_adaptive_model_recovery,
    run_adaptive_parameter_recovery,
)
from src.subjective_randomness.stimulus_design import generate_candidate_pool


def _abs_error_sum(estimate, true):
    return sum(abs(estimate[k] - true[k]) for k in true)


# ── adaptive parameter recovery ─────────────────────────────────────


def test_adaptive_parameter_recovery_learns_and_beats_prior():
    true = {"theta_alt": 0.80, "alt_weight": 0.30, "beta": 3.0, "side_bias": 0.5}
    pool = generate_candidate_pool(n_pairs=80, lengths=(6, 8), seed=3)

    result = run_adaptive_parameter_recovery(
        "prototype_similarity",
        true,
        pool,
        n_rounds=20,
        n_participants=30,
        points_per_dim=6,
        seed=0,
    )

    # One stimulus chosen per round, all distinct, all from the pool.
    assert len(result["selected_stimuli"]) == 20
    keys = {(s["sequence_a"], s["sequence_b"]) for s in result["selected_stimuli"]}
    assert len(keys) == 20
    assert result["selected_stimuli"][0]["eig"] > 0.0

    # The posterior concentrates (entropy falls) and the estimate beats the
    # uninformed prior mean.
    assert result["final_entropy_bits"] < result["prior_entropy_bits"]
    assert _abs_error_sum(result["posterior_mean"], true) < _abs_error_sum(
        result["prior_mean"], true
    )


def test_adaptive_parameter_recovery_is_deterministic():
    true = {"theta_alt": 0.80, "alt_weight": 0.30, "beta": 3.0, "side_bias": 0.5}
    pool = generate_candidate_pool(n_pairs=40, lengths=(6,), seed=3)
    kwargs = dict(n_rounds=10, n_participants=20, points_per_dim=5, seed=0)

    a = run_adaptive_parameter_recovery("prototype_similarity", true, pool, **kwargs)
    b = run_adaptive_parameter_recovery("prototype_similarity", true, pool, **kwargs)
    assert a["posterior_mean"] == b["posterior_mean"]
    assert [s["sequence_a"] for s in a["selected_stimuli"]] == [
        s["sequence_a"] for s in b["selected_stimuli"]
    ]


# ── adaptive model recovery ─────────────────────────────────────────


def test_adaptive_model_recovery_identifies_separable_model():
    # encoding_compressibility is well separated from the other families, so an
    # adaptive design should recover it and place most posterior mass on it.
    true_params = {
        "longrun_weight": 0.40,
        "periodic_share": 0.50,
        "beta": 4.0,
        "side_bias": 0.0,
    }
    pool = generate_candidate_pool(n_pairs=80, lengths=(6, 8), seed=5)

    result = run_adaptive_model_recovery(
        pool,
        true_model="encoding_compressibility",
        true_params=true_params,
        model_names=[
            "prototype_similarity",
            "encoding_compressibility",
            "bayesian_diagnosticity",
        ],
        n_rounds=25,
        n_participants=40,
        points_per_dim=5,
        seed=0,
    )

    assert result["recovered_model"] == "encoding_compressibility"
    assert result["recovered_correct"] is True
    assert result["model_posterior"]["encoding_compressibility"] > 0.5
    assert abs(sum(result["model_posterior"].values()) - 1.0) < 1e-9
    assert len(result["selected_stimuli"]) == 25
    assert result["selected_stimuli"][0]["eig"] > 0.0


def test_adaptive_model_confusion_assembles_one_entry_per_generating_model():
    pool = generate_candidate_pool(n_pairs=40, lengths=(6, 8), seed=7)
    generating_params = {
        "encoding_compressibility": {
            "longrun_weight": 0.40,
            "periodic_share": 0.50,
            "beta": 4.0,
            "side_bias": 0.0,
        },
        "prototype_similarity": {
            "theta_alt": 0.65,
            "alt_weight": 0.55,
            "beta": 4.0,
            "side_bias": 0.0,
        },
    }
    result = run_adaptive_model_confusion(
        pool,
        generating_params=generating_params,
        model_names=[
            "prototype_similarity",
            "encoding_compressibility",
            "bayesian_diagnosticity",
        ],
        n_rounds=15,
        n_participants=40,
        points_per_dim=4,
        seed=0,
    )

    assert [e["generating_model"] for e in result["generating"]] == [
        "encoding_compressibility",
        "prototype_similarity",
    ]
    for entry in result["generating"]:
        assert entry["recovered_model"] in result["model_names"]
        assert abs(sum(entry["model_posterior"].values()) - 1.0) < 1e-9
    # The cleanly-separable model is recovered correctly.
    enc = result["generating"][0]
    assert enc["recovered_model"] == "encoding_compressibility"
    assert enc["recovered_correct"] is True


def test_adaptive_model_recovery_rejects_unknown_true_model():
    pool = generate_candidate_pool(n_pairs=10, lengths=(6,), seed=0)
    with pytest.raises(ValueError, match="true_model"):
        run_adaptive_model_recovery(
            pool,
            true_model="not_a_model",
            true_params={},
            model_names=["prototype_similarity"],
            n_rounds=3,
            points_per_dim=4,
            seed=0,
        )
