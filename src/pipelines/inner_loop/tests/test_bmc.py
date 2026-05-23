"""Task 4: Bayesian model comparison with BIC."""

import json
import math

import numpy as np
import pytest

from src.pipelines.inner_loop.bmc import compute_bmc, write_bmc
from src.pipelines.inner_loop.fitting import FitResult
from src.pipelines.inner_loop.zoo import record_candidate, record_initial


def _fit(ll: float, n_params: int, n_trials: int = 100) -> FitResult:
    return FitResult(
        params=[0.0] * n_params,
        log_likelihood=ll,
        per_trial_ll=np.full(n_trials, ll / n_trials),
        n_samples=1,
        n_trials=n_trials,
        n_params=n_params,
    )


def _populate_zoo(tmp_path, fits: dict[tuple[int, str], FitResult]):
    """Populate a zoo with candidates indexed by (iteration, candidate_id)."""
    src = tmp_path / "model.py"
    src.write_text("def cognitive_model(s, t, p): return None\n")
    zoo = tmp_path / "zoo"
    for (it, cid), fit in fits.items():
        record_candidate(zoo, it, cid, src, fit)
    return zoo


# ---- Empty / degenerate -------------------------------------------------------


def test_compute_bmc_empty_zoo_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        compute_bmc(tmp_path / "no_zoo")


def test_compute_bmc_single_entry_yields_unit_posterior(tmp_path):
    zoo = _populate_zoo(tmp_path, {(0, "candidate_0"): _fit(ll=-50.0, n_params=2)})
    out = compute_bmc(zoo)

    assert set(out["posteriors"]) == {"iter_0__candidate_0"}
    assert pytest.approx(1.0) == out["posteriors"]["iter_0__candidate_0"]
    assert out["top_mass_set"] == ["iter_0__candidate_0"]


# ---- BIC penalty arithmetic --------------------------------------------------


def test_bic_penalty_matches_formula(tmp_path):
    zoo = _populate_zoo(
        tmp_path,
        {
            (0, "candidate_0"): _fit(ll=-100.0, n_params=0, n_trials=50),
            (0, "candidate_1"): _fit(ll=-95.0, n_params=4, n_trials=50),
        },
    )
    out = compute_bmc(zoo)

    assert out["bic_penalties"]["iter_0__candidate_0"] == 0.0
    expected = 0.5 * 4 * math.log(50)
    assert pytest.approx(expected) == out["bic_penalties"]["iter_0__candidate_1"]

    # Marginal LL = LL_max - penalty
    assert pytest.approx(-100.0) == out["marginal_log_likelihoods"]["iter_0__candidate_0"]
    assert pytest.approx(-95.0 - expected) == (
        out["marginal_log_likelihoods"]["iter_0__candidate_1"]
    )


def test_bic_penalty_zero_when_n_params_zero(tmp_path):
    """Subjective-randomness-style zero-param models pay no BIC penalty."""
    zoo = _populate_zoo(
        tmp_path,
        {
            (0, "candidate_0"): _fit(ll=-10.0, n_params=0),
            (0, "candidate_1"): _fit(ll=-12.0, n_params=0),
        },
    )
    out = compute_bmc(zoo)
    assert out["bic_penalties"]["iter_0__candidate_0"] == 0.0
    assert out["bic_penalties"]["iter_0__candidate_1"] == 0.0
    # Posterior reduces to softmax over raw LLs.
    diff = out["log_posteriors"]["iter_0__candidate_0"] - out["log_posteriors"]["iter_0__candidate_1"]
    assert pytest.approx(2.0) == diff


# ---- Posterior normalisation -------------------------------------------------


def test_posteriors_sum_to_one(tmp_path):
    zoo = _populate_zoo(
        tmp_path,
        {
            (0, "candidate_0"): _fit(ll=-50.0, n_params=1),
            (0, "candidate_1"): _fit(ll=-52.0, n_params=2),
            (1, "candidate_0"): _fit(ll=-55.0, n_params=0),
        },
    )
    out = compute_bmc(zoo)
    assert pytest.approx(1.0) == sum(out["posteriors"].values())


def test_higher_marginal_ll_yields_higher_posterior(tmp_path):
    zoo = _populate_zoo(
        tmp_path,
        {
            (0, "candidate_0"): _fit(ll=-50.0, n_params=2),
            (0, "candidate_1"): _fit(ll=-100.0, n_params=2),  # much worse
        },
    )
    out = compute_bmc(zoo)
    assert out["posteriors"]["iter_0__candidate_0"] > out["posteriors"]["iter_0__candidate_1"]
    # 50-nat advantage → essentially probability 1.
    assert out["posteriors"]["iter_0__candidate_0"] > 0.9999


def test_bic_can_flip_argmax(tmp_path):
    """A higher-LL model can lose to a simpler one once BIC is applied."""
    # log(N) = log(100) ≈ 4.605. Penalty for k=10: 0.5*10*4.605 ≈ 23.03.
    # If LL advantage is 5 but penalty differential is 23, simple model wins.
    zoo = _populate_zoo(
        tmp_path,
        {
            (0, "simple"): _fit(ll=-100.0, n_params=0, n_trials=100),
            (0, "complex"): _fit(ll=-95.0, n_params=10, n_trials=100),
        },
    )
    out = compute_bmc(zoo)
    assert out["ranking"][0] == "iter_0__simple"
    assert out["posteriors"]["iter_0__simple"] > out["posteriors"]["iter_0__complex"]


# ---- Prior -------------------------------------------------------------------


def test_uniform_prior_has_no_effect_when_omitted(tmp_path):
    zoo = _populate_zoo(
        tmp_path,
        {
            (0, "a"): _fit(ll=-50.0, n_params=0),
            (0, "b"): _fit(ll=-51.0, n_params=0),
        },
    )
    out_no_prior = compute_bmc(zoo)
    flat = {"iter_0__a": -2.5, "iter_0__b": -2.5}  # equal log priors
    out_with_flat = compute_bmc(zoo, log_prior=flat)
    for k in out_no_prior["posteriors"]:
        assert pytest.approx(out_no_prior["posteriors"][k]) == out_with_flat["posteriors"][k]


def test_log_prior_shifts_posterior(tmp_path):
    zoo = _populate_zoo(
        tmp_path,
        {
            (0, "a"): _fit(ll=-50.0, n_params=0),
            (0, "b"): _fit(ll=-50.0, n_params=0),
        },
    )
    # Equal LL → equal posterior under uniform prior.
    out_uniform = compute_bmc(zoo)
    assert pytest.approx(0.5) == out_uniform["posteriors"]["iter_0__a"]

    # Heavy prior favouring `a`.
    out_skewed = compute_bmc(zoo, log_prior={"iter_0__a": 3.0, "iter_0__b": 0.0})
    assert out_skewed["posteriors"]["iter_0__a"] > out_skewed["posteriors"]["iter_0__b"]
    # log p_a - log p_b = (LL_a - LL_b) + (prior_a - prior_b) = 0 + 3 = 3
    diff = out_skewed["log_posteriors"]["iter_0__a"] - out_skewed["log_posteriors"]["iter_0__b"]
    assert pytest.approx(3.0) == diff


# ---- top_mass_set ------------------------------------------------------------


def test_top_mass_set_is_smallest_set_above_threshold(tmp_path):
    # Construct posteriors ~ [0.6, 0.3, 0.1] via raw LLs.
    # softmax([log .6, log .3, log .1]) = [.6, .3, .1]
    p_targets = [0.6, 0.3, 0.1]
    fits = {
        (0, f"candidate_{i}"): _fit(ll=math.log(p), n_params=0)
        for i, p in enumerate(p_targets)
    }
    zoo = _populate_zoo(tmp_path, fits)

    out = compute_bmc(zoo, top_mass_threshold=0.5)
    assert out["top_mass_set"] == ["iter_0__candidate_0"]  # 0.6 ≥ 0.5

    out = compute_bmc(zoo, top_mass_threshold=0.7)
    assert out["top_mass_set"] == ["iter_0__candidate_0", "iter_0__candidate_1"]  # 0.9 ≥ 0.7


def test_ranking_is_descending(tmp_path):
    zoo = _populate_zoo(
        tmp_path,
        {
            (0, "low"): _fit(ll=-100.0, n_params=0),
            (0, "high"): _fit(ll=-50.0, n_params=0),
            (0, "mid"): _fit(ll=-75.0, n_params=0),
        },
    )
    out = compute_bmc(zoo)
    posts = [out["posteriors"][eid] for eid in out["ranking"]]
    assert posts == sorted(posts, reverse=True)


# ---- n_trials handling -------------------------------------------------------


def test_disagreeing_n_trials_requires_override(tmp_path):
    zoo = _populate_zoo(
        tmp_path,
        {
            (0, "a"): _fit(ll=-10.0, n_params=1, n_trials=50),
            (0, "b"): _fit(ll=-10.0, n_params=1, n_trials=80),
        },
    )
    with pytest.raises(ValueError, match="disagree on n_trials"):
        compute_bmc(zoo)

    # Override resolves it.
    out = compute_bmc(zoo, n_trials=100)
    assert out["n_trials"] == 100


def test_n_trials_must_be_positive(tmp_path):
    zoo = _populate_zoo(tmp_path, {(0, "a"): _fit(ll=-1.0, n_params=0, n_trials=10)})
    with pytest.raises(ValueError, match="positive"):
        compute_bmc(zoo, n_trials=0)


# ---- write_bmc round-trip ----------------------------------------------------


def test_write_bmc_persists_json(tmp_path):
    zoo = _populate_zoo(
        tmp_path,
        {
            (0, "a"): _fit(ll=-10.0, n_params=1),
            (0, "b"): _fit(ll=-12.0, n_params=2),
        },
    )
    out_path = tmp_path / "out" / "model_posterior.json"
    result = write_bmc(zoo, out_path)
    assert out_path.exists()
    on_disk = json.loads(out_path.read_text())
    assert on_disk["posteriors"] == result["posteriors"]
    assert on_disk["ranking"] == result["ranking"]
    assert pytest.approx(1.0) == sum(on_disk["posteriors"].values())
