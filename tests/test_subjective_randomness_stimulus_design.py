"""Fast tests for ranking/selecting stimuli by model-discrimination power.

The core metric is the mutual information between model identity and a single
binary response (bits), computed from each model's p_left. These tests pin the
information calculation with hand-checkable numbers, the ranking/selection
behavior with injected predictors, and a smoke check against the real families.
"""

from __future__ import annotations

import math

import pytest

from src.subjective_randomness.stimulus_design import (
    binary_entropy,
    default_model_family_names,
    family_predict_fns,
    generate_candidate_pool,
    model_discrimination_eig,
    rank_stimuli,
    select_discriminating_stimuli,
)

STIM = {"sequence_a": "HTHT", "sequence_b": "HHHH"}


# ── binary entropy ──────────────────────────────────────────────────


def test_binary_entropy_endpoints_and_max():
    assert binary_entropy(0.0) == 0.0
    assert binary_entropy(1.0) == 0.0
    assert binary_entropy(0.5) == pytest.approx(1.0)
    assert binary_entropy(0.9) == pytest.approx(0.468996, abs=1e-5)


# ── discrimination EIG ──────────────────────────────────────────────


def test_discrimination_eig_zero_when_models_agree():
    fns = {"a": lambda s: 0.7, "b": lambda s: 0.7}
    assert model_discrimination_eig(STIM, fns) == pytest.approx(0.0)


def test_discrimination_eig_positive_when_models_split():
    # p̄ = 0.5 -> H = 1; each model H(0.9) = 0.468996; MI = 1 - 0.468996.
    fns = {"a": lambda s: 0.9, "b": lambda s: 0.1}
    assert model_discrimination_eig(STIM, fns) == pytest.approx(1 - 0.468996, abs=1e-5)


def test_discrimination_eig_respects_model_weights():
    # With almost all weight on one model, the response is nearly certain and
    # carries little information about identity -> EIG near 0.
    fns = {"a": lambda s: 0.9, "b": lambda s: 0.1}
    eig = model_discrimination_eig(STIM, fns, model_weights={"a": 0.99, "b": 0.01})
    assert eig < 0.1


def test_discrimination_eig_rejects_no_models():
    with pytest.raises(ValueError, match="at least one model"):
        model_discrimination_eig(STIM, {})


# ── ranking / selection ─────────────────────────────────────────────

_RANK_FNS = {"a": lambda s: 0.5, "b": lambda s: s["p_b"]}
_CANDIDATES = [
    {"sequence_a": "x", "sequence_b": "y", "p_b": 0.5},   # models agree -> eig 0
    {"sequence_a": "u", "sequence_b": "v", "p_b": 0.95},  # split -> high eig
    {"sequence_a": "m", "sequence_b": "n", "p_b": 0.7},   # mild split
]


def test_rank_stimuli_sorts_by_discrimination_descending():
    ranked = rank_stimuli(_CANDIDATES, _RANK_FNS)
    eigs = [r["discrimination_eig"] for r in ranked]
    assert eigs == sorted(eigs, reverse=True)
    assert ranked[0]["p_b"] == 0.95  # most discriminating first
    assert ranked[-1]["p_b"] == 0.5  # agreement last (eig ~ 0)


def test_select_discriminating_stimuli_returns_top_k():
    top2 = select_discriminating_stimuli(_CANDIDATES, _RANK_FNS, 2)
    assert len(top2) == 2
    assert [s["p_b"] for s in top2] == [0.95, 0.7]


def test_select_discriminating_stimuli_rejects_bad_k_and_empty():
    with pytest.raises(ValueError, match="k must be"):
        select_discriminating_stimuli(_CANDIDATES, _RANK_FNS, 0)
    with pytest.raises(ValueError, match="No candidate"):
        select_discriminating_stimuli([], _RANK_FNS, 1)


# ── real model families ─────────────────────────────────────────────


def test_default_model_family_names_are_the_three_families():
    assert set(default_model_family_names()) == {
        "prototype_similarity",
        "encoding_compressibility",
        "bayesian_diagnosticity",
    }


def test_family_predict_fns_score_real_stimuli_nonnegative_and_discriminating():
    fns = family_predict_fns(default_model_family_names())
    # Stimuli engineered to load differently across the families.
    candidates = [
        {"sequence_a": "HHHHTTTT", "sequence_b": "HTHTHTHT"},
        {"sequence_a": "HHHHHHHH", "sequence_b": "HTTHTHHT"},
        {"sequence_a": "HHTHTTHT", "sequence_b": "HTHTHTHT"},
    ]
    ranked = rank_stimuli(candidates, fns)
    assert all(r["discrimination_eig"] >= 0.0 for r in ranked)
    # At least one stimulus genuinely separates the families.
    assert ranked[0]["discrimination_eig"] > 0.0


def test_generate_candidate_pool_is_diverse_valid_and_deterministic():
    pool = generate_candidate_pool(n_pairs=40, lengths=(6, 8), seed=1)
    assert len(pool) == 40
    # Every item is a pair of distinct H/T strings of an allowed length.
    for item in pool:
        a, b = item["sequence_a"], item["sequence_b"]
        assert a != b
        assert set(a) <= {"H", "T"} and set(b) <= {"H", "T"}
        assert len(a) == len(b) and len(a) in (6, 8)
    # Pairs are unique, and the same seed reproduces the same pool.
    keys = {(d["sequence_a"], d["sequence_b"]) for d in pool}
    assert len(keys) == 40
    again = generate_candidate_pool(n_pairs=40, lengths=(6, 8), seed=1)
    assert [(d["sequence_a"], d["sequence_b"]) for d in pool] == [
        (d["sequence_a"], d["sequence_b"]) for d in again
    ]


def test_generate_candidate_pool_rejects_oversized_request():
    # Only 2^4 = 16 sequences of length 4 -> C(16,2)=120 distinct pairs.
    with pytest.raises(ValueError, match="distinct pairs"):
        generate_candidate_pool(n_pairs=1000, lengths=(4,), seed=0)


def test_family_predict_fns_prior_predictive_is_valid():
    # Averaging over parameter draws (cheap prior-predictive) also yields valid,
    # non-negative discrimination scores.
    fns = family_predict_fns(default_model_family_names(), param_samples=8, seed=0)
    eig = model_discrimination_eig(
        {"sequence_a": "HHHHTTTT", "sequence_b": "HTHTHTHT"}, fns
    )
    assert eig >= 0.0 and math.isfinite(eig)
