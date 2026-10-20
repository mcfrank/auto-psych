"""The designer's EIG tool (docs/auto_open/PLAN.md §4): expected information gain over a
paradigm's trial units, under any categorical link."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.rsa.context import Context, group_by_shape
from src.rsa.model_file import prior_draws
from src.runtime.config import PROJECT_ASSETS_DIR

SEEDS = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
SCENE = Context(objects=((1, 0), (1, 1), (0, 0)), feature_names=("hat", "glasses"), utterance=0)


@pytest.mark.parametrize("name", ["choice", "rating", "production", "binary_judgment"])
def test_outcome_probabilities_are_the_likelihood_of_each_outcome(name):
    from src.rsa_open.links import likelihood

    link = likelihood(name, scale_points=5)
    group = group_by_shape([SCENE])[SCENE.shape]
    rng = np.random.default_rng(0)
    params = {k: float(v) for k, v in link.reference_params.items()}
    params["temperature"], params["lapse"] = 1.7, 0.08
    if name in ("production", "binary_judgment"):
        latent = jnp.asarray(rng.dirichlet(np.ones(3), size=3).T[None] * np.array([1, 1, 0])[None, :, None])
        latent = latent / jnp.maximum(latent.sum(axis=1, keepdims=True), 1e-9)
        unit = {"target": jnp.asarray([1]), "utterance": jnp.asarray([0])}
    else:
        latent = jnp.asarray(rng.dirichlet(np.ones(3))[None])
        unit = {"probe": jnp.asarray([2])}
    classes = jnp.asarray(group.classes)
    probs = np.asarray(link.outcome_probs(latent, params, unit, group.arrays, classes))[0]
    valid = np.asarray(link.outcome_valid(group.arrays, classes, unit))[0]
    assert probs[~valid].max(initial=0.0) == 0.0
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)
    for k in np.where(valid)[0]:
        enc = {**unit, **link.outcome_response(int(k))}
        ll = float(link.log_lik(latent, params, enc, group.arrays, classes)[0])
        assert np.exp(ll) == pytest.approx(probs[k], rel=1e-5)


def test_slider_bins_are_a_proper_distribution_that_moves_with_the_probability():
    from src.rsa_open.links import ordered_beta_bin_probs

    params = {"cut_low": -3.0, "cut_gap": 6.0, "phi": 10.0, "lapse": 0.05}
    probs = np.asarray(ordered_beta_bin_probs(jnp.asarray([0.05, 0.3, 0.6, 0.95]), 100, params))
    assert probs.shape == (4, 12)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    means = (probs * np.arange(12)).sum(axis=1)
    assert np.all(np.diff(means) > 0)


def test_trial_units_are_one_per_distinguishable_question():
    from src.rsa_open.design import trial_units
    from src.rsa_open.links import likelihood

    scenes = [SCENE, Context(objects=((1, 0), (1, 0), (0, 1)), feature_names=("hat", "glasses"), utterance=None)]
    assert len(trial_units(scenes, likelihood("choice"))) == 2
    # A probe or target per class of identical objects: 3 classes, then 2.
    assert len(trial_units(scenes, likelihood("rating", scale_points=5))) == 5
    assert len(trial_units(scenes, likelihood("ordered_beta", total_points=100))) == 5
    assert len(trial_units(scenes, likelihood("production"))) == 5
    # Judgments: each target with each real message (2 real messages in each scene).
    assert len(trial_units(scenes, likelihood("binary_judgment"))) == 10
    # A speaker's trial ignores the heard message: the same game heard otherwise adds nothing.
    reheard = Context(objects=SCENE.objects, feature_names=SCENE.feature_names, utterance=1)
    assert len(trial_units([SCENE, reheard], likelihood("production"))) == 3
    assert len(trial_units([SCENE, reheard], likelihood("choice"))) == 2


def test_the_tool_refuses_allocation_until_it_can_score_it():
    from src.rsa_open.design import trial_units
    from src.rsa_open.links import likelihood

    with pytest.raises(NotImplementedError, match="allocation"):
        trial_units([SCENE], likelihood("allocation", total_points=100))  # a bet


@pytest.fixture(scope="module")
def context(tmp_path_factory):
    """Two ported theories with prior draws standing in for posterior draws."""
    from src.rsa_open.design import write_context
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.port import port

    root = tmp_path_factory.mktemp("context")
    theories = {}
    for name in ("literal_listener", "rsa_l1"):
        path = root / f"{name}.py"
        path.write_text(port((SEEDS / f"{name}.py").read_text(), name=f"{name}.py"))
        draws = prior_draws(TheoryModel(path).params, 40, 0)
        theories[name] = (path, {k: np.asarray([d[k] for d in draws]) for k in draws[0]})
    return write_context(root / "ctx", theories)


def test_a_context_with_draws_that_do_not_match_its_theory_is_refused(context, tmp_path):
    from src.rsa_open.design import ContextInvalid, load_context

    bad = tmp_path / "bad"
    import shutil

    shutil.copytree(context, bad)
    with np.load(bad / "draws" / "rsa_l1.npz") as d:
        arrays = {k: d[k] for k in d.files}
    arrays["not_a_param"] = arrays.pop("alpha")
    np.savez(bad / "draws" / "rsa_l1.npz", **arrays)
    with pytest.raises(ContextInvalid, match="alpha"):
        load_context(bad)


def response_block(kind):
    from tests.test_rsa_open_paradigm import response

    block = response(kind)
    if kind in ("production", "binary_judgment"):
        block.pop("prior_prompt", None)
    return block


def small_paradigm(response):
    from tests.test_rsa_open_paradigm import closed_like_spec
    from src.rsa_open.paradigm import parse_paradigm

    spec = closed_like_spec()
    spec["scene_space"]["sizes"] = [[2, 2], [3, 2]]
    spec["response"] = response
    spec["design"].update(n_participants=120, trials_per_participant=8, distinct_trials=12, minutes=4,
                          pay_usd=0.70)
    return parse_paradigm(spec)


@pytest.mark.parametrize("response", [response_block(k) for k in
                                      ("choice", "rating", "production", "binary_judgment", "slider")],
                         ids=lambda r: r["type"])
def test_the_tool_designs_a_paradigm_and_reports_its_information_per_dollar(context, response, tmp_path):
    from src.rsa_open.design import load_context, score_paradigm

    paradigm = small_paradigm(response)
    report = score_paradigm(paradigm, load_context(context), n_draws=16, n_scenarios=200, seed=0,
                            reference=False)
    assert report["link"] == response["link"]
    assert len(report["design"]) == 12
    assert report["responses_per_unit"] == 120 * 8 // 12
    assert 0 < report["joint_eig_bits"] <= report["max_bits"] + 1e-9
    assert report["bits_per_dollar"] == pytest.approx(report["joint_eig_bits"] / report["cost_usd"])
    json.dumps(report)  # the designer reads it as JSON


@pytest.mark.slow  # scores the closed paradigm over its 1,597 scenes
def test_the_floor_is_half_the_closed_paradigm_s_information_per_dollar(context):
    from src.rsa_open.design import FLOOR_SHARE, load_context, score_paradigm

    paradigm = small_paradigm(response_block("choice"))
    report = score_paradigm(paradigm, load_context(context), n_draws=16, n_scenarios=200, seed=0)
    ref = report["reference"]
    assert ref["design_size"] == 52 and ref["responses_per_unit"] == 50
    assert report["floor_bits_per_dollar"] == pytest.approx(FLOOR_SHARE * ref["bits_per_dollar"])
    assert report["meets_floor"] == (report["bits_per_dollar"] >= report["floor_bits_per_dollar"])
