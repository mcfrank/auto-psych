"""Tests for the LLM-as-participant (no-browser) collection path.

These exercise the participant-model abstraction and the wiring into the active
programmatic collector *offline* — a fake participant model stands in for any
real backend (Gemini / Hugging Face), so no API calls or model downloads happen.
"""

from __future__ import annotations

import csv
import json

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
