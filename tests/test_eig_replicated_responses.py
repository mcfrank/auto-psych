"""Joint EIG when every selected stimulus is answered by n participants.

Each experiment shows every stimulus to all n simulated participants, who share
the ground truth's parameters, so a stimulus yields a count k ~ Binomial(n, p),
not one Bernoulli response. With ``n_responses=n`` the selection scores that
count; ``n_responses=1`` is the single-response objective, unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.models.eig_selection import estimate_joint_eig, select_n_joint_eig


def _point_models(**p_left):
    """Models without parameter uncertainty, each as two identical draws: a
    scenario's likelihood average leaves out its generating draw, and the copy
    keeps the point model's likelihood exact."""
    return {name: np.repeat(np.array([row]), 2, axis=0) for name, row in p_left.items()}


def _exact_binomial_mi(p_by_model, n):
    """I(M; K) in bits for one stimulus, K ~ Binomial(n, p_m), uniform prior."""
    prior = 1.0 / len(p_by_model)
    h_prior = math.log2(len(p_by_model))
    expected_h = 0.0
    for k in range(n + 1):
        lik = [math.comb(n, k) * p**k * (1 - p) ** (n - k) for p in p_by_model]
        p_k = prior * sum(lik)
        post = [prior * lk / p_k for lk in lik]
        expected_h += p_k * -sum(q * math.log2(q) for q in post if q > 0)
    return h_prior - expected_h


@pytest.mark.parametrize("n", [1, 5, 40])
def test_estimate_matches_the_exact_binomial_mutual_information(n):
    p = _point_models(a=[0.45], b=[0.60])
    estimate = estimate_joint_eig(p, [0], n_scenarios=40_000, seed=3, n_responses=n)
    assert estimate == pytest.approx(_exact_binomial_mi([0.45, 0.60], n), abs=0.01)


def test_forty_responses_carry_far_more_information_than_one():
    p = _point_models(a=[0.45], b=[0.60])
    one = estimate_joint_eig(p, [0], n_scenarios=20_000, seed=3, n_responses=1)
    forty = estimate_joint_eig(p, [0], n_scenarios=20_000, seed=3, n_responses=40)
    assert one < 0.05 < 0.4 < forty


def test_greedy_gain_matches_the_exact_value_for_the_first_pick():
    """The first greedy step's joint EIG is one stimulus's binomial MI."""
    p = _point_models(a=[0.45, 0.50, 0.30], b=[0.60, 0.50, 0.32])
    sel = select_n_joint_eig(p, 1, n_scenarios=40_000, seed=5, n_responses=40)
    assert sel.indices == [0]
    assert sel.joint_eig_bits[0] == pytest.approx(
        _exact_binomial_mi([0.45, 0.60], 40), abs=0.01
    )


def test_one_response_reproduces_the_single_response_selection_exactly():
    rng = np.random.default_rng(0)
    p = {m: rng.uniform(0.05, 0.95, size=(20, 60)) for m in ("a", "b", "c")}
    default = select_n_joint_eig(p, 8, n_scenarios=300, seed=7)
    explicit = select_n_joint_eig(p, 8, n_scenarios=300, seed=7, n_responses=1)
    assert explicit.indices == default.indices
    assert explicit.joint_eig_bits == default.joint_eig_bits


def test_n_responses_must_be_positive():
    p = {"a": np.array([[0.4]]), "b": np.array([[0.6]])}
    with pytest.raises(ValueError, match="n_responses"):
        select_n_joint_eig(p, 1, n_responses=0)
