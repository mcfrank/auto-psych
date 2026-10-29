"""Participants' experience in a designed paradigm (PI 2026-10-29, after run C′'s designer chose one
robot, a bet typed per object and 24 near-identical trials): at least 3 domains for pictures, at
most 30 test trials, a brief that asks about variety and effort, and an end-of-study enjoyment
rating that reaches the next designers."""

import copy
import json

import pytest

from tests.test_rsa_open_paradigm import closed_like_spec, varied, write


def test_a_designed_spec_needs_three_domains_and_at_most_thirty_test_trials(tmp_path):
    from src.rsa_open.paradigm import MAX_TEST_TRIALS, MIN_DOMAINS, ParadigmInvalid, load_designed, load_paradigm

    assert (MIN_DOMAINS, MAX_TEST_TRIALS) == (3, 30)
    one = closed_like_spec()
    load_paradigm(write(tmp_path, one))  # any spec the harness reads is still valid
    with pytest.raises(ParadigmInvalid, match="at least 3"):
        load_designed(write(tmp_path, one, name="one.json"))
    load_designed(write(tmp_path, varied(copy.deepcopy(one)), name="three.json"))
    long = varied(copy.deepcopy(one))
    long["design"].update(trials_per_participant=31, distinct_trials=62, n_participants=100)
    with pytest.raises(ParadigmInvalid, match="at most 30 test trials"):
        load_designed(write(tmp_path, long, name="long.json"))


def test_the_fixed_arm_s_spec_meets_them():
    from src.rsa_open.fixed import PAIR_SPEC_PATH, SPEC_PATH
    from src.rsa_open.paradigm import load_designed

    for path in (SPEC_PATH, PAIR_SPEC_PATH):
        p = load_designed(path)
        assert len(p.materials.domains) == 6 and p.design.trials_per_participant == 13


def test_the_brief_asks_about_variety_effort_and_length():
    src = open("src/rsa_open/designer.py").read()
    for words in ("People's experience of the study", "effort per answer", "at least {MIN_DOMAINS} domains",
                  "experience.md", "how your design keeps people engaged"):
        assert words in src


def test_earlier_experiments_ratings_reach_the_designer():
    from src.rsa_open.designer import experience_md

    assert "No live experiment has finished yet" in experience_md([])
    md = experience_md([{"source": "chain_0, experiment 1", "paradigm": "bets", "response": "bet", "domains": 3,
                         "trials": 24, "minutes": 6.0, "n": 140, "mean": 2.4,
                         "counts": {"1": 30, "2": 50, "3": 40, "4": 15, "5": 5}}])
    assert "| chain_0, experiment 1 | bets | bet | 3 | 24 | 6 | 140 | 2.40 | 30 / 50 / 40 / 15 / 5 |" in md


def test_the_rating_is_collected_aggregated_and_recorded_for_later_designs(tmp_path):
    from src.rsa_open.chain import Chain, ChainConfig
    from src.rsa_open.live import monitor_report, person
    from tests.test_rsa_open_live import frozen_experiment, records, stored

    p, lists, run = frozen_experiment(tmp_path / "run")
    people = []
    for k, rating in enumerate((5, 4, None)):
        recs = records(lists["lists"][k])
        if rating is not None:
            recs.append({"task": "experience", "rating": rating, "comment": "fine"})
        people.append(person(f"p{k}", stored(f"x{k}", lists["lists"][k], recs), lists))
    assert [q.enjoyment for q in people] == [5, 4, None]
    report = monitor_report(people, {})
    assert report["enjoyment"] == {"mean": 4.5, "n": 2, "counts": {"1": 0, "2": 0, "3": 0, "4": 1, "5": 1}}
    assert "fine" not in json.dumps(report)  # comments stay in the private raw data
    data = run / "experiment1" / "data"
    data.mkdir(parents=True)
    (data / "experience.json").write_text(json.dumps({"source": "s", **report["enjoyment"]}))
    chain = Chain(ChainConfig(chain="chain_0", seeds_dir=tmp_path, literature=None, out_dir=run, ground_truth=None),
                  designer_spawn=None, theorist_spawn=None)
    assert chain.experience(2)[0]["mean"] == 4.5 and chain.experience(1) == []
