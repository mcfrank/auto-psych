"""Tenenbaum's (2000) number-game model as a seed, with example noise.

"Rules and similarity in concept learning" (NIPS 12): 5,083 hypotheses, 33
mathematical properties (odd, even, squares, cubes, primes, multiples of
3-12, powers of 2-10, ending in 1-9) sharing prior lambda uniformly and the
5,050 intervals sharing 1 - lambda by an Erlang density on their size,
(|h| / sigma^2) exp(-|h| / sigma); strong sampling; generalisation by
posterior membership. His fit has lambda = 1/2 and sigma = 10. Without noise
every example's own target has p(yes) = 1, and 1,742 of the dataset's
responses to one are "no", so each example is drawn from the concept with
probability 0.9 and from 1..100 otherwise, as in Bigelow & Piantadosi (2016).

The seed collapses the intervals by (size, examples contained), which
determine an interval's likelihood. The reference below enumerates all 5,083.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytest
from scipy.special import logsumexp

from src.models.data_binding import make_stim_data
from src.models.model_loading import load_pymc_model
from src.number_game import data as ng_data
from src.runtime.config import REPO_ROOT

SEEDS_DIR = REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects" / "number_game" / "seed_models"
DATA_CSV = REPO_ROOT / "data" / "number-game" / "human experiment" / "numbergame_data.csv"
NUMBERS = np.arange(1, 101)


def _is_prime(n):
    return n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))


def reference_hypotheses():
    rules = [
        {n for n in NUMBERS if n % 2 == 1},
        {n for n in NUMBERS if n % 2 == 0},
        {n * n for n in range(1, 11)},
        {n**3 for n in range(1, 5)},
        {n for n in NUMBERS if _is_prime(n)},
    ]
    rules += [set(range(m, 101, m)) for m in range(3, 13)]
    rules += [{m**k for k in range(1, 8) if m**k <= 100} for m in range(2, 11)]
    rules += [{n for n in NUMBERS if n % 10 == d} for d in range(1, 10)]
    intervals = [set(range(a, b + 1)) for a in range(1, 101) for b in range(a, 101)]
    return rules, intervals


def reference_p_yes(number_set, lam, sigma, noise):
    """p(y in C | X) for y = 1..100 by enumerating every hypothesis."""
    rules, intervals = reference_hypotheses()
    sizes = np.array([len(h) for h in intervals])
    erlang = sizes / sigma**2 * np.exp(-sizes / sigma)
    log_prior = np.concatenate([
        np.full(len(rules), np.log(lam / len(rules))),
        np.log(1 - lam) + np.log(erlang / erlang.sum()),
    ])
    hypotheses = rules + intervals
    log_lik = np.array([
        sum(np.log((1 - noise) * (x in h) / len(h) + noise / 100) for x in number_set) for h in hypotheses
    ])
    log_post = log_prior + log_lik
    post = np.exp(log_post - logsumexp(log_post))
    membership = np.array([[y in h for y in NUMBERS] for h in hypotheses], dtype=float)
    return post @ membership


@pytest.fixture(scope="module")
def model():
    return load_pymc_model("tenenbaum_rules_similarity", SEEDS_DIR)


def seed_p_yes(model, number_sets, lam, sigma):
    """The seed's p(yes) for every target of every set, at fixed parameters."""
    rows = [
        {"number_set": ",".join(map(str, s)), "target": str(y), "rating": "0"}
        for s in number_sets for y in NUMBERS
    ]
    with model:
        pm.set_data(make_stim_data(model, rows))
    [graph] = model.replace_rvs_by_values([model["p_yes"]])
    fn = model.compile_fn(graph, inputs=model.value_vars, on_unused_input="ignore")
    constrained = {"lambda_rules": lam, "sigma": sigma}
    point = {}
    for rv in model.free_RVs:
        transform = model.rvs_to_transforms[rv]
        x = np.float64(constrained[rv.name])
        point[model.rvs_to_values[rv].name] = transform.forward(x, *rv.owner.inputs).eval() if transform else x
    return fn(point).reshape(len(number_sets), 100)


def test_hypothesis_space_is_tenenbaums_5083():
    rules, intervals = reference_hypotheses()
    assert len(rules) == 33 and len(intervals) == 5050


@pytest.mark.parametrize("lam, sigma", [(0.5, 10.0), (0.3, 4.0)])
def test_collapsed_seed_equals_enumerating_every_hypothesis(model, lam, sigma):
    sets = [(16,), (60, 80), (16, 8, 2, 64), (16, 23, 19, 20), (81, 25, 4)]
    got = seed_p_yes(model, sets, lam, sigma)
    expected = np.stack([reference_p_yes(s, lam, sigma, noise=0.1) for s in sets])
    np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-12)


def test_hypothesis_space_and_prior_reproduce_tenenbaums_patterns():
    """His model (no noise) at lambda = 1/2, sigma = 10 (paper, Figure 1); the
    seed equals this reference with noise 0.1 (test above)."""
    sets = [(16, 8, 2, 64), (60, 80, 10, 30), (16, 23, 19, 20), (16,)]
    p = {s: reference_p_yes(s, 0.5, 10.0, noise=1e-9) for s in sets}
    at = lambda s, y: p[s][y - 1]  # noqa: E731
    # A few examples of a small rule: generalise to that rule only.
    assert at((16, 8, 2, 64), 32) > 0.9 and at((16, 8, 2, 64), 24) < 0.05
    assert at((60, 80, 10, 30), 50) > 0.9 and at((60, 80, 10, 30), 55) < 0.1
    # Examples no rule covers: a magnitude gradient.
    gradient = [at((16, 23, 19, 20), y) for y in (22, 24, 26, 30, 40)]
    assert gradient == sorted(gradient, reverse=True) and gradient[0] > 0.99 and gradient[-1] < 0.1
    # One example: both rule-like and similarity-like generalisation.
    assert at((16,), 64) > at((16,), 87) and at((16,), 17) > at((16,), 87)


@pytest.mark.skipif(not DATA_CSV.exists(), reason="number-game dataset not present")
def test_in_set_no_answers_have_finite_likelihood(model):
    responses = ng_data.load_responses(DATA_CSV)
    in_set_no = responses[[t in s and r == 0 for s, t, r in zip(responses["set"], responses["target"], responses["rating"])]]
    rows = [{"number_set": ",".join(map(str, s)), "target": t, "rating": 0} for s, t in zip(in_set_no["set"], in_set_no["target"])]
    with model:
        pm.set_data(make_stim_data(model, rows))
    assert np.isfinite(model.compile_logp()(model.initial_point()))
