"""A model loop stopped mid-experiment resumes from its last completed round.

A loop used to restart from the set it started from, so a killed cell redid
every round. After each round the loop now checkpoints its state, and a rerun
into the same results dir continues with the next round.
"""

from __future__ import annotations

import json

import pytest

import src.pipelines.inner_loop.pymc_orchestrator as pymc_orchestrator
from src.pipelines.inner_loop.hypothesis_ledger import LEDGER_FILENAME
from src.pipelines.inner_loop.model_zoo import _retire
from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop
from src.pipelines.inner_loop.round_checkpoint import checkpoint_before_round
from tests.inner_loop_fixtures import write_responses, write_seed_models
from tests.test_empty_round_retry import _ledger_rows, _manifest_names, _patch_scoring


class Crash(RuntimeError):
    """A simulated kill of the process."""


def _agents_writing(spawned, notes_dir):
    """Fake candidate agents: round k admits ``round<k>_model`` and leaves a note."""

    def fake_spawn(candidate_dir, docs, **kwargs):
        k = int(candidate_dir.parent.name.removeprefix("iter_"))
        spawned.append(k)
        (candidate_dir / "candidate.py").write_text("# ok\n", encoding="utf-8")
        (candidate_dir / "hypothesis.md").write_text(f"hyp {k}\n", encoding="utf-8")
        (candidate_dir / "model_name.txt").write_text(f"round{k}_model\n", encoding="utf-8")
        notes_dir.mkdir(exist_ok=True)
        (notes_dir / f"round{k}_attempt{spawned.count(k)}.md").write_text("n\n")
        return True

    return fake_spawn


def _run(tmp_path, seed_dir, responses, notes_dir):
    return run_pymc_inner_loop(
        responses,
        tmp_path / "model_loop",
        seed_models_dir=seed_dir,
        max_iterations=3,
        candidate_count=1,
        enable_critique=False,
        agent_notes_dir=notes_dir,
    )


def test_a_loop_killed_in_a_round_resumes_from_the_last_completed_round(
    tmp_path, monkeypatch
):
    seed_dir = write_seed_models(tmp_path)
    responses = write_responses(tmp_path)
    notes_dir = tmp_path / "agent_notes"
    _patch_scoring(monkeypatch)
    spawned = []
    monkeypatch.setattr(
        pymc_orchestrator, "_spawn_candidate_agent", _agents_writing(spawned, notes_dir)
    )
    # Killed at the end of round 2, after its admission and its agent's note.
    record_step = pymc_orchestrator._record_history_step

    def crash_once_in_round_2(history, results_dir, *args, iteration, **kwargs):
        if iteration == 2 and spawned.count(2) == 1:
            raise Crash
        record_step(history, results_dir, *args, iteration=iteration, **kwargs)

    monkeypatch.setattr(pymc_orchestrator, "_record_history_step", crash_once_in_round_2)
    with pytest.raises(Crash):
        _run(tmp_path, seed_dir, responses, notes_dir)

    result = _run(tmp_path, seed_dir, responses, notes_dir)

    loop_dir = tmp_path / "model_loop"
    assert spawned == [0, 1, 2, 2]  # rounds 0 and 1 are not redone
    history = json.loads((loop_dir / "history.json").read_text())
    assert [step["iteration"] for step in history] == [None, 0, 1, 2]
    ledger = _ledger_rows(loop_dir / LEDGER_FILENAME)
    assert [r["name"] for r in ledger if r["outcome"] == "admitted"] == [
        "round0_model", "round1_model", "round2_model"
    ]
    assert _manifest_names(loop_dir / "models") == [
        "model_a", "model_b", "round0_model", "round1_model", "round2_model"
    ]
    # The killed attempt's note is gone; the redone round's is kept.
    assert sorted(p.name for p in notes_dir.iterdir()) == [
        "round0_attempt1.md", "round1_attempt1.md", "round2_attempt2.md"
    ]
    assert result["best_model"] == "model_a"


def test_a_finished_loop_rewound_to_its_first_abandoned_round_redoes_only_those(
    tmp_path, monkeypatch
):
    # The 2026-10-30 nov26b cells: rounds abandoned to a usage limit, the
    # experiment then ended (pruned, exported) as if they had run.
    seed_dir = write_seed_models(tmp_path)
    responses = write_responses(tmp_path)
    notes_dir = tmp_path / "agent_notes"
    _patch_scoring(monkeypatch)
    monkeypatch.setattr(
        pymc_orchestrator,
        "_prune_losers",
        lambda models_dir, *a, ledger, ledger_context, **k: _retire(
            models_dir, {"model_b": "behind"}, ledger=ledger, ledger_context=ledger_context
        ) or ["model_b"],
    )
    spawned = []
    write_round_0 = _agents_writing(spawned, notes_dir)

    def limited_after_round_0(candidate_dir, docs, **kwargs):
        if candidate_dir.parent.name == "iter_0":
            return write_round_0(candidate_dir, docs, **kwargs)
        return True  # the agent wrote nothing

    monkeypatch.setattr(pymc_orchestrator, "_spawn_candidate_agent", limited_after_round_0)
    _run(tmp_path, seed_dir, responses, notes_dir)
    loop_dir = tmp_path / "model_loop"

    checkpoint_before_round(loop_dir, 1, notes_dir, ledger_context="")
    spawned.clear()
    monkeypatch.setattr(
        pymc_orchestrator, "_spawn_candidate_agent", _agents_writing(spawned, notes_dir)
    )
    _run(tmp_path, seed_dir, responses, notes_dir)

    assert spawned == [1, 2]
    history = json.loads((loop_dir / "history.json").read_text())
    assert [step["iteration"] for step in history] == [None, 0, 1, 2]
    assert history[-1]["retired_at_experiment_end"] == ["model_b"]
    ledger = _ledger_rows(loop_dir / LEDGER_FILENAME)
    assert [r["name"] for r in ledger if r["outcome"] == "admitted"] == [
        "round0_model", "round1_model", "round2_model"
    ]
    assert not [r for r in ledger if r["outcome"] == "round_abandoned"]
    assert [r["name"] for r in ledger if r["outcome"] == "pruned"] == ["model_b"]
    assert _manifest_names(loop_dir / "models") == [
        "model_a", "round0_model", "round1_model", "round2_model"
    ]
