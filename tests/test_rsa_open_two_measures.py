"""Two-measure paradigms (PI 2026-10-26): a second question about the same screen, scored by the
pair link (src/rsa_open/links.py `PairLikelihood`), through the spec, the design tool, the page's
lists, simulation, the pilot and the critic."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.test_rsa_open_design import SEEDS
from tests.test_rsa_open_paradigm import response
from tests.test_rsa_open_screens import paradigm_of, units_of


def pair_paradigm(first="choice", second="rating", **design):
    from src.rsa_open.paradigm import parse_paradigm

    spec = paradigm_of(first, **design).model_dump(mode="json")
    spec["second_response"] = response(second) if isinstance(second, str) else second
    return parse_paradigm(spec)


@pytest.fixture(scope="module")
def theory(tmp_path_factory):
    from src.rsa_open.fit import TheoryModel
    from src.rsa_open.port import port

    path = tmp_path_factory.mktemp("t") / "rsa_l1.py"
    path.write_text(port((SEEDS / "rsa_l1.py").read_text(), name="rsa_l1.py"))
    return TheoryModel(path)


# --------------------------------------------------------------------------- the link

def test_the_pair_link_is_the_sum_of_its_parts_with_the_second_s_parameters_prefixed(theory):
    from src.rsa_open.fit import OpenData, simulate
    from src.rsa_open.links import likelihood
    from src.rsa_open.paradigm import scene_pool

    pair = likelihood("pair", first={"name": "choice"}, second={"name": "rating", "scale_points": 5})
    assert set(pair.params) == {"lapse", "temperature", "q2_lapse", "q2_temperature", "q2_center", "q2_spread"}
    scenes = scene_pool(pair_paradigm())[:40]
    truth = {"alpha": 1.5, **pair.reference_params}
    responses = simulate(theory, pair, scenes, truth, jax.random.PRNGKey(0))
    assert all(set(r) == {"choice", "second"} and set(r["second"]) == {"probe", "rating"} for r in responses)
    data = OpenData.build(scenes, responses, pair)
    one = OpenData.build(scenes, [{k: v for k, v in r.items() if k != "second"} for r in responses], pair.first)
    two = OpenData.build(scenes, [r["second"] for r in responses], pair.second)
    from src.rsa_open.fit import pointwise_log_lik

    flat = {k: np.full(3, v, dtype=np.float32) for k, v in truth.items()}
    both = pointwise_log_lik(theory, pair, data, flat)
    first = pointwise_log_lik(theory, pair.first, one, {k: v for k, v in flat.items() if not k.startswith("q2_")})
    second = pointwise_log_lik(theory, pair.second, two,
                               {**{k: v for k, v in flat.items() if k == "alpha"},
                                **{k[3:]: v for k, v in flat.items() if k.startswith("q2_")}})
    np.testing.assert_allclose(both, first + second, rtol=1e-5)


def test_the_pair_s_outcomes_are_the_product_of_its_parts(theory):
    from src.rsa.context import group_by_shape
    from src.rsa_open.links import likelihood
    from src.rsa_open.paradigm import scene_pool

    pair = likelihood("pair", first={"name": "choice"}, second={"name": "rating", "scale_points": 5})
    group = next(iter(group_by_shape(scene_pool(pair_paradigm())[:20]).values()))
    latent = theory.group_probs({"alpha": jnp.asarray(1.5)}, group)
    unit = {"second_probe": jnp.zeros(len(group.indices), dtype=int)}
    p = {k: jnp.asarray(v) for k, v in pair.reference_params.items()}
    probs = pair.outcome_probs(latent, p, unit, group.arrays, jnp.asarray(group.classes))
    valid = pair.outcome_valid(group.arrays, jnp.asarray(group.classes), unit)
    assert probs.shape == valid.shape == (len(group.indices), group.shape[0] * 5)
    np.testing.assert_allclose(np.asarray(probs).sum(-1), 1.0, rtol=1e-5)


def test_a_pair_reads_one_side_of_the_theory_and_its_second_measure_is_small():
    from src.rsa_open.links import likelihood

    with pytest.raises(ValueError, match="one side"):
        likelihood("pair", first={"name": "choice"}, second={"name": "binary_judgment"})
    with pytest.raises(ValueError, match="the second is one of"):
        likelihood("pair", first={"name": "choice"}, second={"name": "ordered_beta", "total_points": 100})


# --------------------------------------------------------------------------- the spec

def test_a_second_response_is_validated_like_the_first_and_by_its_own_rules(tmp_path, monkeypatch):
    import src.rsa_open.links as links
    from src.rsa_open.paradigm import ParadigmInvalid

    p = pair_paradigm()
    assert p.second_response.type == "rating"
    for second, problem in [(response("slider"), "the second is one of"),
                            (response("choice"), "repeats the first"),
                            (response("rating", scale_points=9), "at most 7 points"),
                            ({**response("rating"), "prompt": "No message here."}, "second_response.prompt must show")]:
        with pytest.raises(ParadigmInvalid, match=problem):
            pair_paradigm("choice", second)
    pair_paradigm("production", response("binary_judgment"))  # a speaker measure, then a judgment
    monkeypatch.setattr(links, "MAX_PAIR_OUTCOMES", 10)
    with pytest.raises(ParadigmInvalid, match="joint outcomes"):
        pair_paradigm("choice", "rating")  # 3 objects x 5 points


# --------------------------------------------------------------------------- design and lists

def test_the_design_tool_scores_every_pairing_of_the_two_measures_units():
    from src.rsa_open.design import trial_units
    from src.rsa_open.paradigm import paradigm_likelihood, scene_pool

    p = pair_paradigm()
    scenes = scene_pool(p)
    units = trial_units(scenes, paradigm_likelihood(p))
    per_scene = {i: sum(1 for u in units if u.scene == i) for i in range(len(scenes))}
    assert all(n == len(set(scenes[i].choice_classes())) for i, n in per_scene.items())
    assert all(dict(u.fields).keys() == {"second_probe"} for u in units)


def test_lists_carry_both_questions_and_catch_trials_their_second_fields():
    from src.rsa_open.screens import trial_lists

    p = pair_paradigm()
    lists = trial_lists(p, units_of(p), seed=0, n_lists=3)
    assert lists["response_type"] == "choice+rating"
    for t in (t for lst in lists["lists"] for t in lst["trials"]):
        assert t["widget"]["type"] == "choice" and t["widget2"]["type"] == "rating"
        assert t["widget2"]["marked"] == t["display_order"].index(t["second_probe"])
        assert "prompt2_html" in t


def test_the_pilot_asks_the_second_question_and_records_both():
    from src.rsa_open.pilot import gate, run_participant
    from src.rsa_open.screens import scenes_and_responses, trial_lists
    from src.rsa_open.fit import OpenData
    from src.rsa_open.paradigm import paradigm_likelihood
    from tests.test_rsa_open_pilot import Scripted

    p = pair_paradigm()
    lists = trial_lists(p, units_of(p), seed=0, n_lists=4)
    sessions = [run_participant(lists, i, Scripted(lists, i)) for i in range(4)]
    rows = sessions[0].rows
    assert all(set(r["response"]) == {"choice", "second"} for r in rows)
    scenes, responses = scenes_and_responses(rows)
    OpenData.build(scenes, responses, paradigm_likelihood(p))  # every response encodes
    assert gate(p, lists, sessions)["catch_pass"] == 1.0


def test_the_critic_sees_both_responses_as_columns(theory):
    from src.rsa_open.corpus import Block
    from src.rsa_open.fit import simulate
    from src.rsa_open.links import likelihood
    from src.rsa_open.loop.critique import critique_frame, outcome_columns
    from src.rsa_open.paradigm import scene_pool

    pair = likelihood("pair", first={"name": "choice"}, second={"name": "rating", "scale_points": 5})
    scenes = scene_pool(pair_paradigm())[:30]
    responses = simulate(theory, pair, scenes, {"alpha": 1.5, **pair.reference_params}, jax.random.PRNGKey(1))
    block = Block("exp1", pair, tuple(scenes), tuple(responses), tuple(f"u{i}" for i in range(30)))
    df = critique_frame(block)
    assert {"choice", "second_probe", "second_rating"} <= set(df.columns)
    assert outcome_columns(pair) == ["choice", "second_rating"]


@pytest.mark.slow  # the EIG pick over every pairing of units
def test_a_two_measure_paradigm_is_scored_for_information(tmp_path, theory):
    from src.rsa.model_file import prior_draws
    from src.rsa_open.design import load_context, score_paradigm, write_context

    draws = prior_draws(theory.params, 40, 0)
    write_context(tmp_path / "ctx", {"rsa_l1": (theory.path, {k: np.asarray([d[k] for d in draws]) for k in draws[0]}),
                                     "rsa_l1b": (theory.path, {k: np.asarray([d[k] for d in draws]) for k in draws[0]})})
    report = score_paradigm(pair_paradigm(), load_context(tmp_path / "ctx"), n_draws=16, n_scenarios=100, seed=0,
                            reference=False)
    assert report["link"] == "pair" and len(report["design"]) == 10
    assert all("second_probe" in u for u in report["design"])
