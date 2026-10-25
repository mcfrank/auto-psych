"""Trial lists from a paradigm and its design (src/rsa_open/screens.py), and the data back."""

import collections
import json

import numpy as np
import pytest

from tests.test_rsa_open_design import response_block
from tests.test_rsa_open_paradigm import closed_like_spec

KINDS = ("choice", "rating", "slider", "bet", "production", "binary_judgment")


def paradigm_of(kind, **design):
    from src.rsa_open.paradigm import parse_paradigm

    spec = closed_like_spec()
    spec["scene_space"]["sizes"] = [[2, 2], [3, 2], [3, 3]]
    spec["response"] = response_block(kind)
    spec["design"].update(n_participants=120, trials_per_participant=6, distinct_trials=10, catch_trials=2,
                          minutes=4, pay_usd=0.70)
    spec["design"].update(design)
    return parse_paradigm(spec)


def units_of(paradigm, seed=0):
    """A design of the paradigm's distinct_trials units (random units stand in for the EIG pick)."""
    from src.rsa_open.design import describe_unit, trial_units
    from src.rsa_open.paradigm import paradigm_likelihood, scene_pool
    from src.rsa_open.screens import design_units

    scenes = scene_pool(paradigm)
    if paradigm.response.type == "bet":
        all_units = [{"objects": [list(o) for o in s.objects], "heard": s.utterance} for s in scenes]
    else:
        all_units = [describe_unit(scenes, u) for u in trial_units(scenes, paradigm_likelihood(paradigm))]
    pick = np.random.default_rng(seed).choice(len(all_units), paradigm.design.distinct_trials, replace=False)
    return design_units({"design": [all_units[i] for i in pick]})


def answer(trial, rng, good=True):
    """A response as the page records it, for the trial's widget (scene indices)."""
    w, n = trial["widget"], len(trial["objects"])
    if w["type"] == "choice":
        return {"choice": int(rng.integers(n))}
    if w["type"] == "rating":
        return {"probe": trial["probe"], "rating": int(rng.integers(1, w["points"] + 1))}
    if w["type"] == "slider":
        return {"probe": trial["probe"], "value": int(rng.integers(0, w["max"] + 1))}
    if w["type"] == "bet":
        points = rng.multinomial(w["total"], np.ones(n) / n)
        return {"points": [int(x) for x in points]}
    if w["type"] == "production":
        return {"target": trial["target"], "utterance": w["options"][int(rng.integers(len(w["options"])))]["utterance"]}
    return {"target": trial["target"], "utterance": trial["utterance"], "answer": bool(rng.integers(2))}


@pytest.mark.parametrize("kind", KINDS)
def test_lists_balance_the_design_and_add_practice_and_catch_trials(kind):
    from src.rsa_open.screens import trial_lists

    p = paradigm_of(kind)
    units = units_of(p)
    doc = trial_lists(p, units, seed=3)
    assert len(doc["lists"]) == 120 and doc["paradigm_sha256"] == p.sha256
    counts = collections.Counter()
    for lst in doc["lists"]:
        phases = [t["phase"] for t in lst["trials"]]
        assert phases[0] == "practice" and phases.count("test") == 6 and phases.count("catch") == 2
        assert [t["trial_number"] for t in lst["trials"]] == list(range(9))
        tests = [t["unit"] for t in lst["trials"] if t["phase"] == "test"]
        assert len(set(tests)) == 6  # no unit twice in one list
        counts.update(tests)
        for t in lst["trials"]:
            assert sorted(t["display_order"]) == list(range(len(t["objects"])))
            assert len(t["items_html"]) == len(t["objects"])
    assert set(counts) == set(range(10)) and max(counts.values()) - min(counts.values()) <= 1


def test_lists_are_a_function_of_the_seed():
    from src.rsa_open.screens import trial_lists

    p = paradigm_of("choice")
    units = units_of(p)
    assert trial_lists(p, units, seed=1, n_lists=3) == trial_lists(p, units, seed=1, n_lists=3)
    assert trial_lists(p, units, seed=1, n_lists=3) != trial_lists(p, units, seed=2, n_lists=3)


def test_a_design_of_another_size_than_the_paradigm_s_is_refused():
    from src.rsa_open.screens import trial_lists

    p = paradigm_of("choice")
    with pytest.raises(ValueError, match="distinct_trials"):
        trial_lists(p, units_of(p)[:-1], seed=0)


def test_the_prompt_shows_the_heard_word_or_the_prior_prompt():
    from src.rsa_open.screens import trial_lists

    p = paradigm_of("choice")
    for lst in trial_lists(p, units_of(p), seed=0, n_lists=20)["lists"]:
        for t in lst["trials"]:
            if t["heard"] is None:
                assert "could not hear" in t["prompt_html"]
            else:
                assert f"“{t['words'][t['heard']]}”" in t["prompt_html"]


def test_objects_are_drawn_with_their_features_in_the_trial_s_words():
    from src.rsa_open.screens import trial_lists

    p = paradigm_of("choice")
    domain = p.materials.domains[0]
    t = trial_lists(p, units_of(p), seed=0, n_lists=1)["lists"][0]["trials"][1]
    for pos, obj in enumerate(t["display_order"]):
        n_feats = sum(t["objects"][obj])
        assert t["items_html"][pos].startswith("<svg") and domain.base in t["items_html"][pos]
        drawn = [j for j in range(len(t["words"])) if domain.features[t["feature_map"][j]].svg in t["items_html"][pos]]
        assert drawn == [j for j, v in enumerate(t["objects"][obj]) if v]  # exactly the object's features
        assert n_feats == len(drawn)
        named = [t["words"][j] for j, v in enumerate(t["objects"][obj]) if v]
        assert all(w in t["item_labels"][pos] for w in named)


def test_text_materials_write_each_object():
    from src.rsa_open.paradigm import parse_paradigm
    from src.rsa_open.screens import trial_lists

    spec = paradigm_of("choice").model_dump(mode="json")
    spec["materials"] = {"kind": "text", "domains": [{
        "name": "houses", "base": "a house",
        "features": [{"word": w, "phrase": f"with a {w}"} for w in ("garden", "garage", "porch", "pool")]}]}
    p = parse_paradigm(spec)
    t = trial_lists(p, units_of(p), seed=0, n_lists=1)["lists"][0]["trials"][1]
    assert all(h.startswith('<span class="ro-text">a house') for h in t["items_html"])


@pytest.mark.parametrize("kind", KINDS)
def test_recorded_responses_come_back_as_the_link_s_data(kind):
    from src.rsa_open.fit import OpenData
    from src.rsa_open.paradigm import paradigm_likelihood
    from src.rsa_open.screens import page_rows, scenes_and_responses, trial_lists

    p = paradigm_of(kind)
    doc = trial_lists(p, units_of(p), seed=0, n_lists=12)
    rng = np.random.default_rng(0)
    data = []
    for lst in doc["lists"]:
        data.append({"task": "welcome"})
        for t in lst["trials"]:
            data.append({"task": "rsa_open_trial", **{k: t.get(k) for k in (
                "phase", "unit", "objects", "heard", "probe", "target", "utterance", "expect")},
                "response": answer(t, rng)})
    rows = page_rows(json.loads(json.dumps(data)))  # through JSON, as the page sends it
    scenes, responses = scenes_and_responses(rows)
    assert len(scenes) == 12 * 6
    OpenData.build(scenes, responses, paradigm_likelihood(p))


@pytest.mark.parametrize("kind", KINDS)
def test_catch_trials_pass_the_right_answer_and_fail_a_wrong_one(kind):
    from src.rsa_open.screens import catch_passed, trial_lists

    p = paradigm_of(kind)
    doc = trial_lists(p, units_of(p), seed=0, n_lists=10)
    catches = [t for lst in doc["lists"] for t in lst["trials"] if t["phase"] in ("catch", "practice")]
    for t in catches:
        e, n = t["expect"], len(t["objects"])
        if kind == "choice":
            right, wrong = {"choice": e["choice"]}, {"choice": (e["choice"] + 1) % n}
        elif kind == "rating":
            right, wrong = {"probe": t["probe"], "rating": 5}, {"probe": t["probe"], "rating": 1}
        elif kind == "slider":
            right, wrong = {"probe": t["probe"], "value": 90}, {"probe": t["probe"], "value": 50}
        elif kind == "bet":
            pts = [0] * n
            pts[e["object"]] = 100
            right = {"points": pts}
            wrong = {"points": [100 if i == (e["object"] + 1) % n else 0 for i in range(n)]}
        elif kind == "production":
            false = [f for f, v in enumerate(t["objects"][t["target"]]) if not v]
            right = {"target": t["target"], "utterance": e["utterance_in"][0]}
            wrong = {"target": t["target"], "utterance": false[0]} if false else None
        else:
            right = {"target": t["target"], "utterance": t["utterance"], "answer": e["answer"]}
            wrong = {**right, "answer": not e["answer"]}
        assert catch_passed({"expect": e, "response": right})
        if wrong is not None:
            assert not catch_passed({"expect": e, "response": wrong})


def test_judgment_catch_trials_ask_for_yes_and_for_no():
    from src.rsa_open.screens import trial_lists

    p = paradigm_of("binary_judgment")
    doc = trial_lists(p, units_of(p), seed=0, n_lists=10)
    answers = collections.Counter(t["expect"]["answer"] for lst in doc["lists"] for t in lst["trials"]
                                  if t["phase"] == "catch")
    assert answers[True] == answers[False] == 10


def test_a_production_trial_offers_every_word_of_its_scene():
    from src.rsa_open.screens import trial_lists

    p = paradigm_of("production")
    for t in trial_lists(p, units_of(p), seed=0, n_lists=5)["lists"][0]["trials"]:
        options = t["widget"]["options"]
        assert sorted(o["utterance"] for o in options) == list(range(len(t["objects"][0])))
        assert [o["label"] for o in options] == [t["words"][o["utterance"]] for o in options]


def test_catch_scenes_are_the_closed_campaign_s_drawn_alike():
    """The open code's copy of `_catch_spec` (the agents' tree leaves src/rsa/experiment out)."""
    from src.rsa.experiment.design import _catch_spec
    from src.rsa_open.screens import catch_scene

    for shape in ((2, 2), (3, 2), (3, 4), (4, 3), (6, 4)):
        for seed in range(3):
            a, b = np.random.default_rng(seed), np.random.default_rng(seed)
            spec, target = _catch_spec(shape, a)
            assert catch_scene(shape, b) == (spec.objects, spec.utterance, target)
            assert a.integers(10**9) == b.integers(10**9)  # the generator is left in the same state
