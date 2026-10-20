"""Speakers (docs/auto_open/PLAN.md §3.1, PI 2026-10-21): every theory has one, its own or the
speaker derived from its listener, and the production and judgment links score it."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.rsa.context import Context, group_by_shape
from src.rsa.model_file import prior_draws
from src.runtime.config import PROJECT_ASSETS_DIR

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"


@pytest.fixture(scope="module")
def theory(tmp_path_factory):
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.port import port

    path = tmp_path_factory.mktemp("theory") / "rsa_l1.py"
    path.write_text(port((SEEDS / "rsa_l1.py").read_text(), name="rsa_l1.py"))
    return TheoryModel(path)


# Object 2 has no feature: the scene needs a sink utterance (the last), which no speaker says.
SCENE = Context(objects=((1, 0), (1, 1), (0, 0)), feature_names=("hat", "glasses"), utterance=0)


def test_the_derived_speaker_is_proportional_to_the_listener_over_real_messages(theory):
    group = group_by_shape([SCENE])[SCENE.shape]
    draw = prior_draws(theory.params, 1, 0)[0]
    speaker = np.asarray(theory.group_speaker(draw, group))[0]  # (N_UTT, N_OBJ)
    n_utt = SCENE.shape[1]
    assert speaker.shape == (n_utt, 3) and SCENE.needs_sink
    np.testing.assert_allclose(speaker[-1], 0.0)  # the sink is never said
    listener = np.stack([np.asarray(theory.group_probs(draw, group_by_shape([ctx])[ctx.shape]))[0]
                         for ctx in (Context(SCENE.objects, SCENE.feature_names, utterance=u) for u in (0, 1))])
    for r in (0, 1):
        np.testing.assert_allclose(speaker[:2, r], listener[:, r] / listener[:, r].sum(), rtol=1e-5)
    np.testing.assert_allclose(speaker[:, 2], [0.5, 0.5, 0.0])  # nothing true of it: indifferent
    np.testing.assert_allclose(speaker.sum(axis=0), 1.0, rtol=1e-6)


def test_a_theory_s_own_speaker_is_used_when_it_defines_one(tmp_path):
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.port import port

    source = port((SEEDS / "rsa_l1.py").read_text(), name="rsa_l1.py") + '''

def speaker(params, ctx):
    real = 1.0 - ctx.is_sink
    return jnp.broadcast_to((real / real.sum())[:, None], ctx.lex.shape)
'''
    path = tmp_path / "own.py"
    path.write_text(source)
    model = TheoryModel(path)
    group = group_by_shape([SCENE])[SCENE.shape]
    speaker = np.asarray(model.group_speaker(prior_draws(model.params, 1, 0)[0], group))[0]
    np.testing.assert_allclose(speaker, [[0.5] * 3, [0.5] * 3, [0.0] * 3])


def test_production_probabilities_cover_the_real_messages_only():
    from src.rsa_open.links import likelihood

    link = likelihood("production")
    speaker = jnp.asarray([[[0.7, 0.2, 0.5], [0.3, 0.8, 0.5], [0.0, 0.0, 0.0]]])  # (1, N_UTT, N_OBJ)
    arrays = group_by_shape([SCENE])[SCENE.shape].arrays
    total = 0.0
    for u in (0, 1):
        enc = {"target": jnp.asarray([1]), "utterance": jnp.asarray([u])}
        total += float(jnp.exp(link.log_lik(speaker, {"lapse": 0.1, "temperature": 1.5}, enc, arrays, None))[0])
    assert total == pytest.approx(1.0, abs=1e-6)


def test_a_judgment_is_more_likely_yes_the_likelier_the_speaker_says_it():
    from src.rsa_open.links import likelihood

    link = likelihood("binary_judgment")
    speaker = jnp.asarray([[[0.9, 0.1, 0.5], [0.1, 0.9, 0.5], [0.0, 0.0, 0.0]]])
    arrays = group_by_shape([SCENE])[SCENE.shape].arrays
    params = {"center": 0.0, "lapse": 0.05, "temperature": 1.0}

    def p_yes(target, u):
        enc = {"target": jnp.asarray([target]), "utterance": jnp.asarray([u]), "answer": jnp.asarray([1])}
        return float(jnp.exp(link.log_lik(speaker, params, enc, arrays, None))[0])

    assert p_yes(0, 0) > 0.9 > 0.1 > p_yes(0, 1)
    no = {"target": jnp.asarray([0]), "utterance": jnp.asarray([0]), "answer": jnp.asarray([0])}
    assert p_yes(0, 0) + float(jnp.exp(link.log_lik(speaker, params, no, arrays, None))[0]) == pytest.approx(1.0)


@pytest.mark.parametrize("link, response, problem", [
    ("production", {"target": 0, "utterance": 2}, "sink"),
    ("production", {"target": 5, "utterance": 0}, "target"),
    ("production", {"target": 0}, "utterance"),
    ("binary_judgment", {"target": 0, "utterance": 1, "answer": "yes"}, "answer"),
])
def test_malformed_speaker_responses_fail_loudly(link, response, problem):
    from src.rsa_open.fit import OpenData
    from src.rsa_open.links import likelihood

    with pytest.raises(ValueError, match=problem):
        OpenData.build([SCENE], [response], likelihood(link))


def test_simulated_speaker_responses_have_the_link_s_form(theory):
    from src.rsa.loop.novelty import plain_pool
    from src.rsa_open.fit import OpenData, simulate
    from src.rsa_open.links import likelihood

    pool = plain_pool()
    scenes = [pool[i] for i in np.random.default_rng(3).choice(len(pool), 60)]
    for name in ("production", "binary_judgment"):
        link = likelihood(name)
        params = {**{k: float(np.asarray(v)) for k, v in prior_draws(theory.params, 1, 0)[0].items()},
                  **link.reference_params}
        OpenData.build(scenes, simulate(theory, link, scenes, params, jax.random.PRNGKey(0)), link)


def test_speaker_links_are_in_the_library_and_paradigms_can_use_them():
    from src.rsa_open.links import LINKS

    assert LINKS["production"].response_type == "production"
    assert LINKS["binary_judgment"].response_type == "binary_judgment"


@pytest.mark.slow  # NUTS on simulated data, 3 replicates per link
@pytest.mark.parametrize("name", ["production", "binary_judgment"])
def test_each_speaker_link_recovers_known_parameters_from_simulated_data(name, tmp_path):
    from src.rsa_open.fit import recovery_check
    from src.rsa_open.links import likelihood
    from src.rsa_open.port import port

    path = tmp_path / "rsa_l1_salience.py"
    path.write_text(port((SEEDS / "rsa_l1_salience.py").read_text(), name="rsa_l1_salience.py"))
    link = likelihood(name)
    truth = {"alpha": 2.0, "w_features": 0.5, "w_familiar": 0.0, **link.reference_params}
    report = recovery_check(path, link, truth, n_trials=1500, seed=0)
    assert report["converged"], report["convergence_problems"]
    assert report["recovered"], report["misses"]
