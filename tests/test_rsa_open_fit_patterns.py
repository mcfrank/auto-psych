"""The open fit evaluates each distinct display once and each distinct (display, response)
pattern once, weighted by its count (src/rsa_open/fit.py, OpenData; 2026-11-08). These tests hold
it to the computation it replaced, one evaluation per trial, reproduced here: the same per-trial
log-likelihoods, the same total, the same pointwise draws, for every link, on data whose displays
and responses repeat as real data do."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.rsa.loop.novelty import plain_pool
from tests.test_rsa_open_fit import _first_draw, theory  # noqa: F401 (fixture)

LINKS = ["choice", "rating", "allocation", "ordered_beta", "production", "binary_judgment"]


def _per_trial(theory, link, contexts, responses, params):
    """The replaced computation: every trial its own row of its shape group."""
    from src.rsa.context import group_by_shape
    from src.rsa_open.fit import _latent, _split

    tp, lp = _split(theory, link, params)
    per = [link.encode(c, r, i) for i, (c, r) in enumerate(zip(contexts, responses))]
    out = np.zeros(len(contexts))
    for group in group_by_shape(contexts).values():
        enc = link.stack([per[i] for i in group.indices])
        ll = link.log_lik(_latent(theory, link, tp, group), lp, enc, group.arrays, jnp.asarray(group.classes))
        out[group.indices] = np.asarray(ll)
    return out


@pytest.fixture(scope="module")
def repeated():
    """60 trials on 12 displays (each shown about 5 times), as a condition's participants see them."""
    pool = plain_pool()
    rng = np.random.default_rng(7)
    shown = [pool[i] for i in rng.choice(len(pool), 12, replace=False)]
    return [shown[i] for i in rng.integers(0, 12, 60)]


@pytest.mark.parametrize("name", LINKS)
def test_each_trial_scores_as_when_every_trial_was_evaluated(name, theory, repeated):  # noqa: F811
    from src.rsa_open.fit import OpenData, TheoryModel, simulate, trial_log_lik
    from src.rsa_open.links import likelihood

    link = likelihood(name, scale_points=5, total_points=10)
    tm = TheoryModel(theory)
    params = {**{k: float(v[0]) for k, v in _first_draw(tm.params).items()}, **link.reference_params}
    responses = simulate(tm, link, repeated, params, jax.random.PRNGKey(3))
    data = OpenData.build(repeated, responses, link)
    assert sum(len(g.indices) for g in data.groups) <= 12 and data.n_patterns < len(repeated)
    assert sum(float(c.sum()) for c in data.counts) == len(repeated)
    ours = np.asarray(trial_log_lik(tm, link, data, params))
    np.testing.assert_allclose(ours, _per_trial(tm, link, repeated, responses, params), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", ["choice", "rating", "production"])
def test_the_total_and_the_pointwise_draws_are_the_per_trial_ones(name, theory, repeated):  # noqa: F811
    from src.rsa.model_file import prior_draws
    from src.rsa_open.fit import OpenData, TheoryModel, _split, pointwise_log_lik, simulate, total_log_lik
    from src.rsa_open.links import likelihood

    link = likelihood(name, scale_points=5)
    tm = TheoryModel(theory)
    draws = [{**{k: float(v) for k, v in d.items()}, **link.reference_params} for d in prior_draws(tm.params, 3, 1)]
    responses = simulate(tm, link, repeated, draws[0], jax.random.PRNGKey(4))
    data = OpenData.build(repeated, responses, link)
    reference = np.stack([_per_trial(tm, link, repeated, responses, d) for d in draws])
    flat = {k: np.asarray([d[k] for d in draws], dtype=np.float32) for k in draws[0]}
    np.testing.assert_allclose(pointwise_log_lik(tm, link, data, flat), reference, rtol=1e-5, atol=1e-5)
    for d, ref in zip(draws, reference):
        tp, lp = _split(tm, link, d)
        assert float(total_log_lik(tm, link, data, tp, lp)) == pytest.approx(ref.sum(), rel=1e-5)


def test_repeats_collapse_to_displays_and_patterns(repeated):
    from src.rsa_open.fit import OpenData
    from src.rsa_open.links import likelihood

    choices = [{"choice": 0}] * len(repeated)  # every trial the same answer: one pattern per display
    data = OpenData.build(repeated, choices, likelihood("choice"))
    displays = sum(len(g.indices) for g in data.groups)
    assert displays == len({id(c) for c in repeated}) and data.n_patterns == displays
    assert data.trial_pattern.shape == (len(repeated),)
