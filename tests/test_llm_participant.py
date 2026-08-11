"""Tests for the LLM-as-participant (no-browser) collection path.

These exercise the participant-model abstraction and the wiring into the active
programmatic collector *offline* — a fake participant model stands in for any
real backend (Gemini / Hugging Face), so no API calls or model downloads happen.
"""

from __future__ import annotations

import csv
import json
import threading
import time

import pytest

from src.pipelines.outer_loop import participants
from src.pipelines.outer_loop.collect import generate_llm_participant_rows
from src.pipelines.outer_loop.participants import get_participant_model


class FakeParticipantModel:
    """Deterministic stand-in: alternates left/right across calls."""

    def __init__(self, name="fake:test"):
        self.name = name
        self._n = 0

    def answer(self, system, user):
        self._n += 1
        return "ANSWER: left" if self._n % 2 else "ANSWER: right"


STIMULI = [
    {"sequence_a": "HHTHT", "sequence_b": "HTHTH"},
    {"sequence_a": "HHHHH", "sequence_b": "HTHTT"},
]

REQUIRED_COLUMNS = {
    "participant_id",
    "trial_index",
    "sequence_a",
    "sequence_b",
    "chose_left",
}


def test_generate_rows_shape_and_schema(tmp_path):
    rows, stats = generate_llm_participant_rows(
        STIMULI,
        n_participants=3,
        participant_model=FakeParticipantModel(),
        prompt_text="You are a participant.",
        transcripts_dir=tmp_path / "transcripts",
    )
    assert len(rows) == 3 * len(STIMULI)
    assert stats == {
        "n_participants": 3,
        "n_stimuli": 2,
        "n_rows": 6,
        "n_unparseable": 0,
        "n_errors": 0,
    }
    assert REQUIRED_COLUMNS <= set(rows[0].keys())
    assert rows[0]["model"] == "fake:test"
    # one transcript per participant
    assert len(list((tmp_path / "transcripts").glob("*.md"))) == 3


def test_unparseable_and_errors_are_counted(tmp_path):
    # Content-based (not call-order-based) so it is deterministic even though
    # participants run concurrently: stimulus 0 -> unparseable, stimulus 1 -> error.
    class Flaky:
        name = "fake:flaky"

        def answer(self, system, user):
            if "HHTHT" in user:  # STIMULI[0].sequence_a -> unparseable reply
                return "i refuse to answer"
            raise RuntimeError("boom")  # STIMULI[1] -> model error

    rows, stats = generate_llm_participant_rows(
        STIMULI, n_participants=2, participant_model=Flaky(), prompt_text="x"
    )
    # 2 participants x 2 stimuli: stimulus 0 unparseable, stimulus 1 errors.
    assert stats["n_unparseable"] == 2
    assert stats["n_errors"] == 2
    assert stats["n_rows"] == len(rows) == 0


def test_transient_backend_error_is_retried_once_within_a_trial():
    """A single flaky API call must not cost the trial its response.

    The downstream completeness gate rejects any collection with a missing
    response, so without a retry one transient blip aborts the whole (paid)
    collection. Each trial gets exactly one second chance.
    """

    class TransientlyFailing:
        name = "fake:transient"

        def __init__(self):
            self.calls: dict[str, int] = {}

        def answer(self, system, user):
            attempt = self.calls[user] = self.calls.get(user, 0) + 1
            if attempt == 1:
                raise RuntimeError("transient backend blip")
            return "ANSWER: left"

    rows, stats = generate_llm_participant_rows(
        STIMULI,
        n_participants=1,
        participant_model=TransientlyFailing(),
        prompt_text="x",
    )
    assert stats["n_errors"] == 0
    assert stats["n_rows"] == len(rows) == len(STIMULI)


def test_uncommitted_reply_is_retried_once_within_a_trial():
    class WaffleThenCommit:
        name = "fake:waffle"

        def __init__(self):
            self.calls: dict[str, int] = {}

        def answer(self, system, user):
            attempt = self.calls[user] = self.calls.get(user, 0) + 1
            if attempt == 1:
                return "hmm, hard to say really"
            return "ANSWER: right"

    rows, stats = generate_llm_participant_rows(
        STIMULI,
        n_participants=1,
        participant_model=WaffleThenCommit(),
        prompt_text="x",
    )
    assert stats["n_unparseable"] == 0
    assert stats["n_rows"] == len(rows) == len(STIMULI)
    assert all(row["chose_left"] == 0 for row in rows)


def test_model_concurrency_limit_is_respected():
    class SingleThreadedModel:
        name = "fake:single-threaded"
        max_concurrency = 1

        def __init__(self):
            self.active_calls = 0
            self.high_water_mark = 0
            self.lock = threading.Lock()

        def answer(self, system, user):
            with self.lock:
                self.active_calls += 1
                self.high_water_mark = max(self.high_water_mark, self.active_calls)
            time.sleep(0.005)
            with self.lock:
                self.active_calls -= 1
            return "ANSWER: left"

    model = SingleThreadedModel()
    rows, stats = generate_llm_participant_rows(
        STIMULI,
        n_participants=4,
        participant_model=model,
        prompt_text="x",
        max_workers=4,
    )

    assert stats["n_rows"] == len(rows) == 8
    assert model.high_water_mark == 1


def test_only_committed_answers_parse_loose_mentions_are_unparseable():
    from src.pipelines.outer_loop.collect import _parse_participant_answer

    # The committed forms parse.
    assert _parse_participant_answer("ANSWER: left") == "left"
    assert _parse_participant_answer("answer:  RIGHT") == "right"
    assert _parse_participant_answer("Left") == "left"
    # A reply that merely mentions a side without committing must NOT be coerced
    # into a choice (it would bias the data given fixed presentation side).
    assert _parse_participant_answer("I'd lean left but it's really close") is None
    assert (
        _parse_participant_answer(
            "the left column looks less random, so I pick the other"
        )
        is None
    )
    assert _parse_participant_answer("i refuse to answer") is None


def test_factory_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown participant backend"):
        get_participant_model("banana")


def test_open_backend_requires_model_name_without_importing_torch():
    # Must raise on the missing id *before* attempting any heavy import, so this
    # passes whether or not torch/transformers are installed.
    with pytest.raises(ValueError, match="requires a Hugging Face model id"):
        get_participant_model("open", None)


def test_run_collect_programmatic_nobrowser_writes_csv(tmp_path, monkeypatch):
    """End-to-end through the active collector with a fake backend."""
    from src.pipelines.outer_loop import orchestrator

    # Patch the factory so no real backend is constructed.
    monkeypatch.setattr(
        participants, "get_participant_model", lambda b, m=None: FakeParticipantModel()
    )

    exp_dir = tmp_path / "experiment1"
    (exp_dir / "design").mkdir(parents=True)
    (exp_dir / "design" / "stimuli.json").write_text(
        json.dumps(STIMULI), encoding="utf-8"
    )

    csv_path = orchestrator.run_collect_programmatic(
        exp_dir,
        mode="simulated_participants_nobrowser",
        n_participants=2,
        project_id="subjective_randomness",
        participant_backend="closed",
    )
    assert csv_path.exists()
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 2 * len(STIMULI)
    assert REQUIRED_COLUMNS <= set(rows[0].keys())


def test_programmatic_nobrowser_rejects_partial_collection(tmp_path, monkeypatch):
    """A few valid rows must not disguise failed or unparseable trials."""
    from src.pipelines.outer_loop import orchestrator

    class PartialParticipant:
        name = "fake:partial"

        def answer(self, system, user):
            if "HHHHH" in user:
                raise RuntimeError("backend failure")
            return "ANSWER: left"

    monkeypatch.setattr(
        participants, "get_participant_model", lambda b, m=None: PartialParticipant()
    )
    exp_dir = tmp_path / "experiment1"
    (exp_dir / "design").mkdir(parents=True)
    (exp_dir / "design" / "stimuli.json").write_text(
        json.dumps(STIMULI), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="incomplete.*expected 4.*received 2"):
        orchestrator.run_collect_programmatic(
            exp_dir,
            mode="simulated_participants_nobrowser",
            n_participants=2,
            project_id="subjective_randomness",
        )

    stats = json.loads(
        (exp_dir / "data" / "collection_stats.json").read_text(encoding="utf-8")
    )
    assert stats["n_errors"] == 2
    assert not (exp_dir / "data" / "responses.csv").exists()
    # The rows that WERE collected are paid LLM output: they must be preserved
    # for diagnosis under a name modeling never reads, not discarded.
    rejected = list(
        csv.DictReader(
            (exp_dir / "data" / "responses_rejected.csv").open(encoding="utf-8")
        )
    )
    assert len(rejected) == 2
    assert all(row["sequence_a"] != "HHHHH" != row["sequence_b"] for row in rejected)
