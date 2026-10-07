"""jsPsych data of the reference-game page -> canonical trial rows (src.rsa.experiment.convert).

The participant is simulated in Python: records shaped as the page writes them
(`src/rsa/experiment/template.html`, `choiceTrial` and `addProperties`). The
browser test (`test_rsa_experiment_browser.py`) checks the page itself.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.rsa.dataset import context_from_row, load_forced_choice
from src.rsa.experiment.convert import convert
from src.rsa.experiment.design import EXPERIMENT_ASSETS_DIR, Design, trial_lists
from src.rsa.ingest.common import COLUMNS, check_frame

DEMO = Design.load(EXPERIMENT_ASSETS_DIR / "demo_design.json")
RECORD_KEYS = (
    "phase", "trial_number", "n_test_trials", "condition", "is_catch", "catch_target", "spec_index", "item",
    "feature_names", "objects", "roles", "utterance", "word", "query", "messages", "display_order", "bases",
)  # fmt: skip


def simulate(doc: dict, list_index: int, positions) -> str:
    """jsPsych.data.get().json() of a participant who clicks ``positions[i]`` on choice screen i."""
    lst = doc["lists"][list_index]
    common = {
        "participant_id": "PROLIFIC123",
        "list_index": list_index,
        "list_seed": lst["seed"],
        "list_assignment": "url",
        "design_name": doc["design_name"],
        "design_sha256": doc["design_sha256"],
    }
    records = [dict(common, task=task, response=0, rt=900) for task in ("welcome", "consent", "instructions")]
    for i, (trial, position) in enumerate(zip(lst["trials"], positions)):
        record = dict(common, task="rsa_choice", **{k: trial[k] for k in RECORD_KEYS})
        record.update(
            response=position,
            rt=1000 + 10 * i,
            choice_position=position,
            choice=trial["display_order"][position],
            chosen_alt=trial["screen"][position]["alt"],
            trial_type="html-button-response",
        )
        records.append(record)
        if trial["phase"] == "practice":
            records.append(dict(common, task="practice_done", response=0, rt=500))
    for j, r in enumerate(records):
        r["trial_index"] = j
        r["time_elapsed"] = 1000 * j
    return json.dumps(records)


def _doc(n_lists=4):
    return trial_lists(DEMO, seed=21, n_lists=n_lists, n_catch=2)


def test_every_row_loads_as_a_context_and_the_choice_is_the_clicked_object(tmp_path):
    doc = _doc()
    rng = np.random.default_rng(0)
    frames = []
    for k, lst in enumerate(doc["lists"]):
        positions = [int(rng.integers(len(t["objects"]))) for t in lst["trials"]]
        frame = convert(simulate(doc, k, positions), participant_id=f"p{k}", experiment="rsa_demo")
        assert list(frame.columns) == COLUMNS
        tests = lst["trials"][1:]
        assert len(frame) == len(tests)  # the practice trial is not a row
        for (_, row), trial, position in zip(frame.iterrows(), tests, positions[1:]):
            ctx = context_from_row(row)
            assert row["choice"] == trial["display_order"][position]
            assert ctx.objects == tuple(tuple(r) for r in trial["objects"])
            assert ctx.utterance == trial["utterance"]
            assert list(ctx.feature_names) == trial["feature_names"]
            assert json.loads(row["display_order"]) == trial["display_order"]
            assert row["condition"] == trial["condition"]
            assert row["participant_id"] == f"auto_psych:p{k}"
            cov = json.loads(row["covariates"])
            assert cov["rt"] > 0 and cov["choice_position"] == position
            if trial["is_catch"]:
                assert cov["catch_correct"] == (row["choice"] == trial["catch_target"])
        frames.append(frame)
    path = tmp_path / "rows.csv"
    pd.concat(frames).to_csv(path, index=False, lineterminator="\n")
    check_frame(pd.read_csv(path), "auto_psych")
    loaded = load_forced_choice(path)
    assert len(loaded.contexts) == sum(len(f) for f in frames)
    assert loaded.choices == [int(c) for f in frames for c in f["choice"]]


def test_the_page_identity_is_not_carried_into_the_rows():
    doc = _doc(1)
    data = simulate(doc, 0, [0] * len(doc["lists"][0]["trials"]))
    frame = convert(data, participant_id="anon7", experiment="rsa_demo")
    assert "PROLIFIC123" not in frame.to_csv()


def test_prior_and_catch_rows_are_labelled():
    doc = _doc(1)
    frame = convert(simulate(doc, 0, [0] * 20), participant_id="a", experiment="e")
    prior = frame[frame["query"] == "prior"]
    assert (prior["query_detail"] == "mumble_one_word").all() and (prior["utterance"] == "").all()
    assert (frame["condition"] == "catch").sum() == 2
    assert (frame["framing"] == "one_word").all() and (frame["dv"] == "forced_choice").all()
    assert all(json.loads(m) == list(range(len(json.loads(f)))) for m, f in zip(frame["messages"], frame["feature_names"]))


def _records(doc):
    return json.loads(simulate(doc, 0, [0] * 20))


def test_missing_or_repeated_trials_raise():
    doc = _doc(1)
    records = _records(doc)
    dropped = [r for r in records if not (r.get("phase") == "test" and r["trial_number"] == 3)]
    with pytest.raises(ValueError, match="expected test trials"):
        convert(dropped, participant_id="a", experiment="e")
    repeated = records + [next(r for r in records if r.get("phase") == "test")]
    with pytest.raises(ValueError, match="expected test trials"):
        convert(repeated, participant_id="a", experiment="e")
    with pytest.raises(ValueError, match="no test choices"):
        convert([r for r in records if r.get("phase") != "test"], participant_id="a", experiment="e")


def test_inconsistent_records_raise():
    doc = _doc(1)
    records = _records(doc)
    first = next(r for r in records if r.get("phase") == "test")
    first["choice"] = (first["choice"] + 1) % len(first["objects"])
    with pytest.raises(ValueError, match="page recorded choice"):
        convert(records, participant_id="a", experiment="e")
    records = _records(doc)
    next(r for r in records if r.get("phase") == "test")["response"] = 9
    with pytest.raises(ValueError, match="not a screen position"):
        convert(records, participant_id="a", experiment="e")
    records = _records(doc)
    next(r for r in records if r.get("phase") == "test")["list_index"] = 5
    with pytest.raises(ValueError, match="disagree on list_index"):
        convert(records, participant_id="a", experiment="e")
    records = _records(doc)
    del next(r for r in records if r.get("phase") == "test")["rt"]
    with pytest.raises(ValueError, match="lacks"):
        convert(records, participant_id="a", experiment="e")


def test_a_participant_id_is_required():
    doc = _doc(1)
    with pytest.raises(ValueError, match="participant_id"):
        convert(_records(doc), participant_id="", experiment="e")
    with pytest.raises(ValueError, match="participant_id"):
        convert(_records(doc), participant_id="x:y", experiment="e")
