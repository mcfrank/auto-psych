"""Tests for the EIG-optimized vs. random stimulus-set comparison.

`compare_parameter_recovery` and `compare_model_recovery` choose two fixed
stimulus sets from one candidate pool — the top-k by expected information gain
under the prior ("eig") and a uniform draw of the same size ("random") — and
evaluate both with the same grid-posterior recovery on the same sampled ground
truths, so the arms differ only in which stimulus set was chosen.
"""

from __future__ import annotations

import pytest

from src.subjective_randomness.adaptive_recovery import (
    compare_model_recovery,
    compare_parameter_recovery,
)
from src.subjective_randomness.model_families import prototype_similarity
from src.subjective_randomness.stimulus_design import generate_candidate_pool

MODEL_NAMES = ["prototype_similarity", "encoding_compressibility"]


def _stimulus_keys(stimuli):
    return {(s["sequence_a"], s["sequence_b"]) for s in stimuli}


# ── integration: parameter-recovery comparison ──────────────────────


def test_compare_parameter_recovery_pairs_arms_on_same_sampled_truths():
    pool = generate_candidate_pool(n_pairs=30, lengths=(6,), seed=3)

    report = compare_parameter_recovery(
        "prototype_similarity",
        pool,
        n_repeats=3,
        n_stimuli=6,
        n_participants=20,
        points_per_dim=4,
        seed=0,
    )

    assert report["model"] == "prototype_similarity"
    assert report["n_stimuli"] == 6
    assert set(report["arms"]) == {"eig", "random"}
    bounds = prototype_similarity.PARAM_BOUNDS
    pool_keys = _stimulus_keys(pool)

    for arm_name in ("eig", "random"):
        arm = report["arms"][arm_name]
        # A fixed set of distinct pool stimuli, chosen once up front.
        assert len(arm["stimuli"]) == 6
        assert _stimulus_keys(arm["stimuli"]) <= pool_keys
        assert len(_stimulus_keys(arm["stimuli"])) == 6
        assert all(s["eig"] >= 0.0 for s in arm["stimuli"])
        assert arm["mean_stimulus_eig"] >= 0.0
        runs = arm["runs"]
        assert len(runs) == 3
        for run in runs:
            assert set(run["true_params"]) == set(bounds)
            assert set(run["posterior_mean"]) == set(bounds)

    # The EIG set is optimized: it cannot carry less prior information per
    # stimulus than a random draw from the same pool.
    eig_arm = report["arms"]["eig"]
    assert eig_arm["mean_stimulus_eig"] >= report["arms"]["random"]["mean_stimulus_eig"]
    # ... and is reported most-informative-first.
    scores = [s["eig"] for s in eig_arm["stimuli"]]
    assert scores == sorted(scores, reverse=True)

    # Paired design: repeat i uses the same sampled truth in both arms, and
    # truths vary across repeats (sampled, not one fixed point).
    eig_runs = report["arms"]["eig"]["runs"]
    random_runs = report["arms"]["random"]["runs"]
    for e, r in zip(eig_runs, random_runs):
        assert e["true_params"] == r["true_params"]
    truths = [run["true_params"] for run in eig_runs]
    assert any(t != truths[0] for t in truths[1:])

    # Each arm summarizes recovery quality per parameter.
    for arm_name in ("eig", "random"):
        summary = report["arms"][arm_name]["summary"]
        assert set(summary) == set(bounds)
        for entry in summary.values():
            assert {"pearson_r", "rmse", "bias", "mean_posterior_sd"} <= set(entry)
            assert entry["rmse"] >= 0.0


def test_compare_parameter_recovery_is_deterministic():
    pool = generate_candidate_pool(n_pairs=20, lengths=(6,), seed=3)
    kwargs = dict(n_repeats=2, n_stimuli=4, n_participants=10, points_per_dim=3, seed=1)

    a = compare_parameter_recovery("prototype_similarity", pool, **kwargs)
    b = compare_parameter_recovery("prototype_similarity", pool, **kwargs)

    assert a == b


def test_compare_parameter_recovery_rejects_nonpositive_repeats():
    pool = generate_candidate_pool(n_pairs=10, lengths=(6,), seed=0)
    with pytest.raises(ValueError, match="n_repeats"):
        compare_parameter_recovery(
            "prototype_similarity", pool, n_repeats=0, n_stimuli=3, points_per_dim=3
        )


def test_compare_parameter_recovery_rejects_set_larger_than_pool():
    pool = generate_candidate_pool(n_pairs=5, lengths=(6,), seed=0)
    with pytest.raises(ValueError, match="n_stimuli"):
        compare_parameter_recovery(
            "prototype_similarity", pool, n_repeats=2, n_stimuli=6, points_per_dim=3
        )


# ── integration: model-recovery comparison ──────────────────────────


def test_compare_model_recovery_reports_accuracy_per_arm():
    pool = generate_candidate_pool(n_pairs=30, lengths=(6,), seed=5)

    report = compare_model_recovery(
        pool,
        model_names=MODEL_NAMES,
        n_repeats=2,
        n_stimuli=6,
        n_participants=20,
        points_per_dim=4,
        seed=0,
    )

    assert report["model_names"] == MODEL_NAMES
    assert report["n_stimuli"] == 6
    assert set(report["arms"]) == {"eig", "random"}
    pool_keys = _stimulus_keys(pool)

    for arm_name in ("eig", "random"):
        arm = report["arms"][arm_name]
        assert len(arm["stimuli"]) == 6
        assert _stimulus_keys(arm["stimuli"]) <= pool_keys
        runs = arm["runs"]
        assert len(runs) == 2 * len(MODEL_NAMES)  # n_repeats x generating models
        for run in runs:
            assert run["generating_model"] in MODEL_NAMES
            assert run["recovered_model"] in MODEL_NAMES
            assert abs(sum(run["model_posterior"].values()) - 1.0) < 1e-9
        assert 0.0 <= arm["accuracy"] <= 1.0
        assert 0.0 <= arm["mean_true_posterior"] <= 1.0
        # Mean-posterior confusion: one row per generating model, rows sum to 1.
        confusion = arm["confusion"]
        assert set(confusion) == set(MODEL_NAMES)
        for row in confusion.values():
            assert abs(sum(row.values()) - 1.0) < 1e-9

    # The EIG set maximizes model-discrimination information under the prior.
    assert (
        report["arms"]["eig"]["mean_stimulus_eig"]
        >= report["arms"]["random"]["mean_stimulus_eig"]
    )

    # Paired design: the same generating truths in both arms.
    pairs = zip(report["arms"]["eig"]["runs"], report["arms"]["random"]["runs"])
    for e, r in pairs:
        assert e["generating_model"] == r["generating_model"]
        assert e["true_params"] == r["true_params"]


def test_compare_model_recovery_rejects_nonpositive_repeats():
    pool = generate_candidate_pool(n_pairs=10, lengths=(6,), seed=0)
    with pytest.raises(ValueError, match="n_repeats"):
        compare_model_recovery(
            pool, model_names=MODEL_NAMES, n_repeats=0, n_stimuli=3, points_per_dim=3
        )
