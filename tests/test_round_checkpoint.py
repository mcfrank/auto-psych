"""The model loop's per-round checkpoint: what is saved, and what a restore puts back."""

from __future__ import annotations

import json

from src.pipelines.inner_loop.hypothesis_ledger import LEDGER_FILENAME
from src.pipelines.inner_loop.round_checkpoint import (
    restore_round_checkpoint,
    save_round_checkpoint,
)


def _loop_state(results_dir, notes_dir, tag):
    (results_dir / "models").mkdir(parents=True, exist_ok=True)
    (results_dir / "models" / "models_manifest.yaml").write_text(tag)
    (results_dir / LEDGER_FILENAME).write_text(tag)
    (results_dir / "history.json").write_text(json.dumps([{"step": tag}]))
    notes_dir.mkdir(exist_ok=True)
    (notes_dir / "note.md").write_text(tag)


def test_a_restore_puts_back_the_state_of_the_last_checkpoint(tmp_path):
    results_dir, notes_dir = tmp_path / "model_loop", tmp_path / "notes"
    _loop_state(results_dir, notes_dir, "after round 1")
    save_round_checkpoint(results_dir, notes_dir, next_round=2, rounds_abandoned=1)
    _loop_state(results_dir, notes_dir, "half of round 2")
    (notes_dir / "extra.md").write_text("written in round 2")

    checkpoint = restore_round_checkpoint(results_dir, notes_dir)

    assert checkpoint == {"next_round": 2, "rounds_abandoned": 1}
    assert (results_dir / "models" / "models_manifest.yaml").read_text() == "after round 1"
    assert (results_dir / LEDGER_FILENAME).read_text() == "after round 1"
    assert json.loads((results_dir / "history.json").read_text()) == [{"step": "after round 1"}]
    assert sorted(p.name for p in notes_dir.iterdir()) == ["note.md"]
    assert (notes_dir / "note.md").read_text() == "after round 1"


def test_a_restore_removes_the_working_dirs_of_rounds_not_yet_completed(tmp_path):
    results_dir, notes_dir = tmp_path / "model_loop", tmp_path / "notes"
    _loop_state(results_dir, notes_dir, "x")
    save_round_checkpoint(results_dir, notes_dir, next_round=2, rounds_abandoned=0)
    for name in ("iter_0", "iter_1", "iter_1_retry_1", "iter_2", "iter_2_retry_1", "iter_10"):
        (results_dir / name).mkdir()

    restore_round_checkpoint(results_dir, notes_dir)

    assert sorted(p.name for p in results_dir.glob("iter_*")) == [
        "iter_0", "iter_1", "iter_1_retry_1"
    ]


def test_notes_absent_at_the_checkpoint_are_absent_after_a_restore(tmp_path):
    results_dir, notes_dir = tmp_path / "model_loop", tmp_path / "notes"
    _loop_state(results_dir, notes_dir, "x")
    (notes_dir / "note.md").unlink()
    notes_dir.rmdir()
    save_round_checkpoint(results_dir, notes_dir, next_round=0, rounds_abandoned=0)
    notes_dir.mkdir()
    (notes_dir / "later.md").write_text("later")

    restore_round_checkpoint(results_dir, notes_dir)

    assert not notes_dir.exists()


def test_without_a_checkpoint_there_is_nothing_to_restore(tmp_path):
    assert restore_round_checkpoint(tmp_path / "model_loop", tmp_path / "notes") is None
