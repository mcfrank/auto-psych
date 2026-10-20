"""The simulated pilot and its gate (src/rsa_open/pilot.py), with scripted participants (no API)."""

import json

import numpy as np
import pytest

from tests.test_rsa_open_screens import KINDS, paradigm_of, units_of


def reply_for(trial, good=True, rng=None):
    """The ANSWER line a participant would give: right on catch trials when ``good``."""
    rng = rng or np.random.default_rng(0)
    w, n, order = trial["widget"], len(trial["objects"]), trial["display_order"]
    e = trial.get("expect") or {}
    pos = {obj: p for p, obj in enumerate(order)}
    t = w["type"]
    if t == "choice":
        k = pos[e["choice"]] if e and good else (pos[e["choice"]] + 1) % n if e else int(rng.integers(n))
        return f"ANSWER: {k + 1}"
    if t == "rating":
        return f"ANSWER: {w['points'] if good or not e else 1}"
    if t == "slider":
        return f"ANSWER: {w['max'] if good or not e else 0}"
    if t == "bet":
        pts = [0] * n
        target = e.get("object", order[0])
        pts[pos[target] if good or not e else (pos[target] + 1) % n] = w["total"]
        return "ANSWER: " + ", ".join(map(str, pts))
    if t == "production":
        if e:
            ok = [o["label"] for o in w["options"] if o["utterance"] in e["utterance_in"]]
            bad = [o["label"] for o in w["options"] if o["utterance"] not in e["utterance_in"]]
            return f"ANSWER: {(ok if good or not bad else bad)[0]}"
        return f"ANSWER: {w['options'][0]['label']}"
    answer = e.get("answer", True) if good else not e.get("answer", True)
    return f"ANSWER: {'yes' if answer else 'no'}"


class Scripted:
    """Answers the screens of one list in order: checks (``check_errors`` readings wrong), then trials."""

    name = "scripted"

    def __init__(self, lists, index, *, good=True, check_errors=0, mumble=0):
        self.lists, self.trials = lists, lists["lists"][index]["trials"]
        self.good, self.check_errors, self.mumble = good, check_errors, mumble
        self.readings, self.check_i, self.trial_i = 0, 0, 0
        self.seen = []

    def reply(self, text, images=()):
        self.seen.append((text, len(images)))
        checks = self.lists["intro"]["checks"]
        if "option's number" in text:
            c = checks[self.check_i % len(checks)]
            wrong = self.check_i // len(checks) < self.check_errors
            self.check_i += 1
            return f"I think it is this one.\nANSWER: {(c['correct'] + (1 if wrong else 0)) % len(c['options']) + 1}"
        if self.mumble > 0:
            self.mumble -= 1
            return "Hmm, hard to say."
        trial = self.trials[self.trial_i]
        self.trial_i += 1
        return reply_for(trial, self.good)


def run(kind, minutes=None, **behaviour):
    from src.rsa_open.pilot import estimated_minutes, gate, run_participant
    from src.rsa_open.screens import trial_lists

    p = paradigm_of(kind)
    units = units_of(p)
    lists = trial_lists(p, units, seed=0, n_lists=8)
    if minutes is None:  # a paradigm that declares its estimated time
        minutes = round(estimated_minutes(lists), 1)
        p = paradigm_of(kind, minutes=minutes, pay_usd=round(minutes * 0.18, 2))
        lists = trial_lists(p, units, seed=0, n_lists=8)
    sessions = [run_participant(lists, i, Scripted(lists, i, **behaviour)) for i in range(8)]
    return gate(p, lists, sessions), sessions, lists


@pytest.mark.parametrize("kind", KINDS)
def test_attentive_participants_who_understand_pass_the_gate(kind):
    report, sessions, _ = run(kind)
    assert report["passed"], report["problems"]
    assert report["catch_pass"] == 1.0 and report["readable"] == 1.0 and report["data_parse"] == "ok"


def test_inattentive_participants_fail_on_attention():
    report, _, _ = run("choice", good=False)
    assert not report["passed"] and any(p.startswith("attention") for p in report["problems"])


def test_a_wrong_check_rereads_the_instructions_and_counts_only_if_wrong_twice():
    report, sessions, _ = run("rating", check_errors=1)
    assert report["comprehension_first"] == 0.0 and report["comprehension_final"] == 1.0
    assert all(len(s.checks) == 2 for s in sessions)
    assert any("read the instructions again" in line for line in sessions[0].transcript)
    report, _, _ = run("rating", check_errors=2)
    assert any(p.startswith("comprehension") for p in report["problems"])


def test_an_unreadable_reply_is_asked_again_then_recorded_as_unreadable():
    report, sessions, _ = run("choice", mumble=1)  # the first trial's first reply is unreadable
    assert report["readable"] == 1.0
    assert any("could not be read" in line for line in sessions[0].transcript)
    report, _, _ = run("choice", mumble=4)  # two trials of each participant go unanswered
    assert report["readable"] < 0.95 and any(p.startswith("readability") for p in report["problems"])


def test_a_paradigm_whose_time_is_far_from_its_estimate_fails():
    report, _, _ = run("choice", minutes=10)
    assert any(p.startswith("time") for p in report["problems"])


@pytest.mark.parametrize("kind, answer, ok", [
    ("choice", "ANSWER: 2", True), ("choice", "ANSWER: 9", False), ("choice", "answer: 1.", True),
    ("rating", "ANSWER: 5", True), ("rating", "ANSWER: 6", False), ("rating", "ANSWER: four", False),
    ("slider", "ANSWER: 100", True), ("slider", "ANSWER: 101", False),
    ("bet", "ANSWER: 100, 0, 0", True), ("bet", "ANSWER: 50, 50", False), ("bet", "ANSWER: 60, 60, -20", False),
    ("binary_judgment", "ANSWER: Yes", True), ("binary_judgment", "ANSWER: maybe", False),
    ("choice", "I pick the second one", False),
])
def test_answers_are_read_into_the_page_s_responses(kind, answer, ok):
    from src.rsa_open.pilot import parse_answer
    from src.rsa_open.screens import trial_lists

    p = paradigm_of(kind)
    trials = trial_lists(p, units_of(p), seed=0, n_lists=1)["lists"][0]["trials"]
    trial = next(t for t in trials if len(t["objects"]) == 3)
    response, why = parse_answer(trial, f"Some thinking.\n{answer}")
    assert (response is not None) == ok, why
    if ok and kind == "bet":  # points by screen position, recorded by object
        assert response["points"][trial["display_order"][0]] == 100
    if ok and kind == "choice" and answer.endswith("2"):
        assert response["choice"] == trial["display_order"][1]


def test_a_production_answer_is_one_of_its_words():
    from src.rsa_open.pilot import parse_answer
    from src.rsa_open.screens import trial_lists

    p = paradigm_of("production")
    trial = trial_lists(p, units_of(p), seed=0, n_lists=1)["lists"][0]["trials"][1]
    word = trial["widget"]["options"][0]
    assert parse_answer(trial, f"ANSWER: “{word['label'].upper()}”")[0]["utterance"] == word["utterance"]
    assert parse_answer(trial, "ANSWER: banana")[0] is None


def test_an_svg_trial_is_shown_as_one_image_with_the_marked_object_framed():
    from src.rsa_open.pilot import trial_screen
    from src.rsa_open.screens import trial_lists

    p = paradigm_of("rating")
    trial = trial_lists(p, units_of(p), seed=0, n_lists=1)["lists"][0]["trials"][1]
    text, images = trial_screen(trial, "svg")
    assert len(images) == 1 and images[0][:8] == b"\x89PNG\r\n\x1a\n"
    assert f"object {trial['widget']['marked'] + 1}, which is with an orange frame" in text
    assert "ANSWER: <1 to 5>" in text
    assert not any(w in text for w in trial["words"] if w not in _plain_prompt(trial))  # no feature words leak


def _plain_prompt(trial):
    from src.rsa_open.pilot import _plain

    return _plain(trial["prompt_html"])


def test_run_pilot_writes_transcripts_rows_and_the_report(tmp_path):
    from src.rsa_open.pilot import estimated_minutes, run_pilot
    from src.rsa_open.screens import trial_lists

    p = paradigm_of("choice")
    lists = trial_lists(p, units_of(p), seed=0, n_lists=6)
    minutes = round(estimated_minutes(lists), 1)
    p = paradigm_of("choice", minutes=minutes, pay_usd=round(minutes * 0.18, 2))
    lists = trial_lists(p, units_of(p), seed=0, n_lists=6)
    report = run_pilot(p, lists, tmp_path, n_participants=4, participant_factory=lambda i: Scripted(lists, i))
    assert report["passed"] and report["participant_model"] == "scripted" and report["n_participants"] == 4
    assert len(list((tmp_path / "transcripts").glob("*.md"))) == 4
    assert json.loads((tmp_path / "pilot.json").read_text())["passed"]
    assert len((tmp_path / "rows.jsonl").read_text().splitlines()) == 4 * 9
    with pytest.raises(ValueError, match="not built from this paradigm"):
        run_pilot(paradigm_of("rating"), lists, tmp_path, participant_factory=lambda i: Scripted(lists, i))
