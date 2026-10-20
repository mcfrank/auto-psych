"""Fitting theory x link on any paradigm's responses (docs/auto_open/PLAN.md §3)."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.rsa.loop.novelty import plain_pool
from src.rsa.model_file import RSAModel, prior_draws
from src.runtime.config import PROJECT_ASSETS_DIR

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"


@pytest.fixture(scope="module")
def contexts():
    pool = plain_pool()
    rng = np.random.default_rng(1)
    return [pool[i] for i in rng.choice(len(pool), 80, replace=False)]


@pytest.fixture(scope="module")
def theory(tmp_path_factory):
    from src.rsa_open.port import port

    path = tmp_path_factory.mktemp("theory") / "rsa_l1_salience.py"
    path.write_text(port((SEEDS / "rsa_l1_salience.py").read_text(), name="rsa_l1_salience.py"))
    return path


def test_the_choice_link_scores_trials_exactly_as_the_closed_fit_does(contexts, theory, tmp_path):
    from src.rsa import fit as closed
    from src.rsa_open.fit import OpenData, TheoryModel, trial_log_lik
    from src.rsa_open.links import compose, likelihood

    rng = np.random.default_rng(2)
    choices = [int(rng.integers(ctx.shape[0])) for ctx in contexts]
    composed = tmp_path / "composed.py"
    composed.write_text(compose(theory.read_text(), "choice"))
    model = RSAModel(composed)
    prep = closed.prepare(contexts, choices)
    data = OpenData.build(contexts, [{"choice": c} for c in choices], likelihood("choice"))
    for draw in prior_draws(model.params, 3, 4):
        flat = {k: np.asarray(v)[None] for k, v in draw.items()}
        theirs = closed.pattern_log_lik(model, prep, flat)[0][prep.trial_pattern]
        ours = np.asarray(trial_log_lik(TheoryModel(theory), likelihood("choice"), data, draw))
        np.testing.assert_allclose(ours, theirs, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("points", [2, 5, 7])
def test_rating_probabilities_over_the_scale_sum_to_one(points):
    from src.rsa_open.links import rating_probs

    p = jnp.asarray([0.0, 0.01, 0.2, 0.5, 0.97, 1.0])  # 0 and 1 are clipped, still finite
    probs = rating_probs(p, beta=1.3, center=0.2, spread=0.8, lapse=0.05, points=points)
    assert probs.shape == (6, points)
    np.testing.assert_allclose(np.asarray(probs.sum(axis=1)), 1.0, atol=1e-5)
    assert np.all(np.asarray(probs) > 0)
    # A higher probability never lowers the expected rating (it can saturate in float32 at the
    # ends of a short scale) and raises it across the range.
    means = np.asarray((probs * jnp.arange(1, points + 1)).sum(axis=1))
    assert np.all(np.diff(means) >= 0) and means[-1] > means[0] + 0.5 * (points - 1)


def test_allocation_probabilities_over_every_split_sum_to_one():
    from src.rsa_open.links import allocation_log_prob

    total, n = 4, 3
    splits = [s for s in itertools.product(range(total + 1), repeat=n) if sum(s) == total]
    latent = jnp.asarray([0.7, 0.3, 0.0])
    logp = allocation_log_prob(latent, jnp.asarray(splits, dtype=jnp.float32), total, kappa=5.0, lapse=0.05)
    assert float(jnp.exp(logp).sum()) == pytest.approx(1.0, abs=1e-5)
    best = splits[int(jnp.argmax(logp))]
    assert best[0] >= best[1] >= best[2]


@pytest.mark.parametrize("link, response, problem", [
    ("choice", {"choice": 9}, "object index"),
    ("choice", {"pick": 0}, "choice"),
    ("rating", {"probe": 0, "rating": 0}, "between 1 and 5"),
    ("rating", {"probe": 7, "rating": 3}, "probe"),
    ("allocation", {"points": [50, 40]}, "one entry per object"),
    ("allocation", {"points": [50, 40, 20]}, "sum to 100"),
    ("allocation", {"points": [60, 50, -10]}, "negative"),
])
def test_malformed_responses_fail_loudly(link, response, problem):
    from src.rsa.context import Context
    from src.rsa_open.fit import OpenData
    from src.rsa_open.links import likelihood

    ctx = Context(objects=((0, 1), (1, 1), (1, 0)), feature_names=("f0", "f1"), utterance=1)
    with pytest.raises(ValueError, match=problem):
        OpenData.build([ctx], [response], likelihood(link, scale_points=5, total_points=100))


def test_a_link_without_a_model_file_form_says_so():
    from src.rsa_open.links import compose

    with pytest.raises(ValueError, match="src.rsa_open.fit"):
        compose("PARAMS = {}\ndef listener(params, ctx):\n    return 0\n", "rating")


def test_simulated_responses_have_the_link_s_form(contexts, theory):
    from src.rsa_open.fit import OpenData, TheoryModel, simulate
    from src.rsa_open.links import likelihood

    for name in ("choice", "rating", "allocation"):
        link = likelihood(name, scale_points=7, total_points=100)
        params = {**{k: v[0] for k, v in _first_draw(TheoryModel(theory).params).items()},
                  **link.reference_params}
        responses = simulate(TheoryModel(theory), link, contexts, params, jax.random.PRNGKey(0))
        OpenData.build(contexts, responses, link)  # validates every response


def _first_draw(params):
    return {k: np.asarray(v)[None] for k, v in prior_draws(params, 1, 0)[0].items()}


@pytest.mark.slow  # NUTS on simulated data, once per link
@pytest.mark.parametrize("name", ["choice", "rating", "allocation"])
def test_each_link_recovers_known_parameters_from_simulated_data(name, theory):
    from src.rsa_open.fit import recovery_check
    from src.rsa_open.links import likelihood

    link = likelihood(name, scale_points=7, total_points=100)
    truth = {"alpha": 2.0, "w_features": 0.5, "w_familiar": 0.0, **link.reference_params}
    report = recovery_check(theory, link, truth, n_trials=1500, seed=0)
    assert report["converged"], report
    assert report["recovered"], report
