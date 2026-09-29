"""Live data correctness (first audit R8).

- ``participant_id`` restarted at 0 in every experiment, so the loop's pooled
  data gave different people the same id (and the same participant random
  effect). It is now unique across a run's experiments; a participant seen in
  an earlier experiment keeps their id. The Prolific IDs, and so the mapping,
  stay in the researchers' raw collected files.
- The Firebase collector kept every ``/results`` row when none belonged to
  this collection's participants, mixing other collections' data in. It now
  raises.

``/results``-shaped fixtures with fake 24-hex Prolific IDs; nothing is
contacted.
"""

from __future__ import annotations

import csv
import io

import pytest

from src.pipelines.outer_loop import collect
from src.pipelines.outer_loop import model_loop_runner as mlr
from src.pipelines.outer_loop import orchestrator

PROJECT = "subjective_randomness"
HEADER = "participant_id,participant_id_str,trial_index,sequence_a,sequence_b,chose_left,chose_right,model"
ALICE = "5f8a1b2c3d4e5f6a7b8c9d0e"
BOB = "60a1b2c3d4e5f6a7b8c9d0e1"
CAROL = "61b2c3d4e5f6a7b8c9d0e1f2"
DAVE = "62c3d4e5f6a7b8c9d0e1f2a3"


def _results(*people: str) -> str:
    """What /results returns for one experiment: participants numbered from 0,
    two trials each, with varied answers."""
    lines = [HEADER]
    for index, person in enumerate(people):
        lines.append(f"{index},{person},0,HHT,THT,1,0,")
        lines.append(f"{index},{person},1,HTHT,HHHH,0,1,")
    return "\n".join(lines) + "\n"


def _collect(monkeypatch, exp_num: int, results_csv: str):
    exp_dir = orchestrator.experiment_dir(PROJECT, exp_num)
    (exp_dir / "experiment").mkdir(parents=True)
    (exp_dir / "experiment" / "config.json").write_text(
        '{"prolific_study_id": "study%d", "results_api_url": "http://results.invalid",'
        ' "total_available_places": 2}' % exp_num,
        encoding="utf-8",
    )
    monkeypatch.setattr(collect, "_poll_prolific_until_target", lambda *a, **k: 2)
    monkeypatch.setattr(
        collect.urllib.request,
        "urlopen",
        lambda *a, **k: io.BytesIO(results_csv.encode("utf-8")),
    )
    orchestrator.run_collect_programmatic(
        exp_dir, mode="live", n_participants=2, project_id=PROJECT, prolific_mode="live"
    )
    return exp_dir


def _ids_by_person(path) -> dict:
    with open(path, encoding="utf-8", newline="") as f:
        return {
            row["participant_id_str"]: int(row["participant_id"])
            for row in csv.DictReader(f)
        }


def test_live_participant_ids_are_unique_across_a_runs_experiments(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("AUTO_PSYCH_OUTPUT_DIR", str(tmp_path / "output"))
    exp1 = _collect(monkeypatch, 1, _results(ALICE, BOB))
    # Experiment 2: two new people, and Bob again. /results numbers them 0-2.
    exp2 = _collect(monkeypatch, 2, _results(CAROL, BOB, DAVE))

    first = _ids_by_person(orchestrator.raw_collected_responses_path(exp1))
    second = _ids_by_person(orchestrator.raw_collected_responses_path(exp2))
    assert first == {ALICE: 0, BOB: 1}
    assert second == {CAROL: 2, BOB: 1, DAVE: 3}

    # The loop's pooled data: one id per person, and no Prolific ID.
    pooled = mlr._pooled_response_rows(exp2)
    assert sorted({int(row["participant_id"]) for row in pooled}) == [0, 1, 2, 3]
    assert len(pooled) == 10
    for exp_dir in (exp1, exp2):
        text = (exp_dir / "data" / "responses.csv").read_text(encoding="utf-8")
        assert not any(person in text for person in (ALICE, BOB, CAROL, DAVE))


def test_simulated_run_py_participants_are_unique_across_experiments_too(
    tmp_path, monkeypatch
):
    """Rows without a Prolific ID (browser-free simulation) are renumbered
    after the earlier experiments' ids."""
    monkeypatch.setenv("AUTO_PSYCH_OUTPUT_DIR", str(tmp_path / "output"))
    earlier = orchestrator.experiment_dir(PROJECT, 1) / "data"
    earlier.mkdir(parents=True)
    (earlier / "responses.csv").write_text(
        "sequence_a,sequence_b,participant_id,trial_index,chose_left\nHT,TH,0,0,1\nHT,TH,1,0,0\n",
        encoding="utf-8",
    )
    exp2 = orchestrator.experiment_dir(PROJECT, 2)
    rows = [
        {
            "sequence_a": "HT",
            "sequence_b": "TH",
            "participant_id": p,
            "trial_index": t,
            "chose_left": 1,
        }
        for p in (0, 1)
        for t in (0, 1)
    ]
    renumbered = orchestrator.run_unique_participant_ids(rows, exp2)
    assert [row["participant_id"] for row in renumbered] == [2, 2, 3, 3]


# ── The Firebase collector: other collections' rows ───────────────────────


def _firebase(monkeypatch, tmp_path, results_csv: str):
    monkeypatch.setattr(collect, "_unique_batch_id", lambda: "b1")
    monkeypatch.setattr(
        collect.urllib.request,
        "urlopen",
        lambda *a, **k: io.BytesIO(results_csv.encode("utf-8")),
    )
    return collect._collect_from_firebase(
        {"project_id": PROJECT, "run_id": 1},
        {"project_id": PROJECT, "run_id": 1},
        "http://results.invalid",
        2,
        tmp_path,
        tmp_path,
    )


OURS = [f"{PROJECT}_run1_b1_p0", f"{PROJECT}_run1_b1_p1"]


def test_firebase_collection_raises_when_no_row_is_this_collections(
    tmp_path, monkeypatch
):
    with pytest.raises(
        RuntimeError, match="none from this collection's 2 participants"
    ):
        _firebase(monkeypatch, tmp_path, _results(ALICE, BOB))


def test_firebase_collection_raises_when_rows_cannot_be_attributed(
    tmp_path, monkeypatch
):
    no_ids = (
        "participant_id,trial_index,sequence_a,sequence_b,chose_left\n0,0,HT,TH,1\n"
    )
    with pytest.raises(RuntimeError, match="without a participant_id_str"):
        _firebase(monkeypatch, tmp_path, no_ids)


def test_firebase_collection_keeps_only_this_collections_rows(tmp_path, monkeypatch):
    rows = _firebase(monkeypatch, tmp_path, _results(ALICE, OURS[1], OURS[0]))
    assert {row["participant_id_str"] for row in rows} == set(OURS)
    assert {(row["participant_id_str"], row["participant_id"]) for row in rows} == {
        (OURS[0], 0),
        (OURS[1], 1),
    }
