"""Unit tests for the incumbent record (``src.subjective_randomness.incumbent``).

The loop-improvement plan's primary metric is whether the exported best model
("the incumbent") ever changes across a cell's scoring steps, and whether it
is ever a model the loop discovered rather than one the cell was seeded with.
These tests pin the pure bookkeeping: the per-step flags, the per-cell
summary, the definition of the starting set, and the readers that assemble a
cell's scoring steps from its per-experiment ``history.json`` files (live tree
or ``agent_runs.tar.gz`` archive).
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from src.subjective_randomness.incumbent import (
    INCUMBENT_COLUMNS,
    annotate_incumbents,
    histories_from_archive,
    histories_from_run_tree,
    incumbent_summary_for_histories,
    starting_model_set,
    steps_from_histories,
    summarise_incumbents,
)


def _step(global_step, best, *, experiment=1, step=None, iteration=None):
    return {
        "experiment": experiment,
        "step": global_step if step is None else step,
        "iteration": iteration,
        "global_step": global_step,
        "best_model": best,
        "rmse": 0.1,
    }


def _history_entry(step, iteration, best, posteriors=None):
    return {
        "step": step,
        "iteration": iteration,
        "best_model": best,
        "posteriors": posteriors if posteriors is not None else {best: 1.0},
        "elpd_loo": {best: -1.0},
    }


# ── per-step flags ───────────────────────────────────────────────────


def test_annotate_flags_a_change_and_a_discovered_incumbent():
    steps = [
        _step(0, "seed_a"),
        _step(1, "seed_a", step=1, iteration=0),
        _step(2, "new_model", experiment=2, step=0),
        _step(3, "seed_b", experiment=2, step=1, iteration=0),
    ]
    rows = annotate_incumbents(steps, {"seed_a", "seed_b"})
    assert [r["incumbent_changed"] for r in rows] == [False, False, True, True]
    assert [r["incumbent_is_discovered"] for r in rows] == [False, False, True, False]
    # Every original field survives, and the two new ones are exactly INCUMBENT_COLUMNS.
    assert all(set(r) == set(steps[0]) | set(INCUMBENT_COLUMNS) for r in rows)
    assert INCUMBENT_COLUMNS == ("incumbent_changed", "incumbent_is_discovered")


def test_annotate_first_step_is_never_a_change_even_when_discovered():
    # A cell whose very first incumbent is not a starting model would be a
    # pipeline bug elsewhere, but the record must still describe it honestly:
    # nothing precedes step 0, so it is not a change.
    rows = annotate_incumbents([_step(0, "x")], {"seed"})
    assert rows[0]["incumbent_changed"] is False
    assert rows[0]["incumbent_is_discovered"] is True


def test_annotate_returns_copies_and_leaves_the_input_alone():
    steps = [_step(0, "seed"), _step(1, "seed")]
    rows = annotate_incumbents(steps, {"seed"})
    assert rows is not steps and rows[0] is not steps[0]
    assert "incumbent_changed" not in steps[0]


def test_annotate_requires_consecutive_global_steps_from_zero():
    with pytest.raises(ValueError, match="global_step"):
        annotate_incumbents([_step(0, "a"), _step(2, "a")], {"a"})
    with pytest.raises(ValueError, match="global_step"):
        annotate_incumbents([_step(1, "a")], {"a"})


def test_annotate_refuses_an_empty_starting_set():
    # With no starting models every incumbent would read "discovered" — a
    # caller bug, not a finding.
    with pytest.raises(ValueError, match="starting"):
        annotate_incumbents([_step(0, "a")], set())


def test_annotate_accepts_an_empty_trajectory():
    assert annotate_incumbents([], {"a"}) == []


# ── per-cell summary ─────────────────────────────────────────────────


def test_summary_counts_changes_and_discovered_incumbent_steps():
    steps = [
        _step(0, "seed_a"),
        _step(1, "seed_a", step=1, iteration=0),
        _step(2, "new_model", experiment=2, step=0),
        _step(3, "new_model", experiment=2, step=1, iteration=0),
        _step(4, "seed_a", experiment=3, step=0),
    ]
    summary = summarise_incumbents(annotate_incumbents(steps, {"seed_a", "seed_b"}), {"seed_b", "seed_a"})
    assert summary == {
        "starting_models": ["seed_a", "seed_b"],
        "n_steps": 5,
        "n_incumbent_changes": 2,
        "n_steps_discovered_incumbent": 2,
        "final_incumbent": "seed_a",
        "changes": [
            {"global_step": 2, "experiment": 2, "step": 0, "from": "seed_a", "to": "new_model"},
            {"global_step": 4, "experiment": 3, "step": 0, "from": "new_model", "to": "seed_a"},
        ],
    }


def test_summary_of_the_baseline_shape_is_all_zeros():
    # The archived motif_stack cells: one seed at every one of nine steps.
    steps = [_step(i, "local_representativeness", experiment=1 + i // 3, step=i % 3) for i in range(9)]
    summary = summarise_incumbents(
        annotate_incumbents(steps, {"local_representativeness", "falk_konold_dp"}),
        {"local_representativeness", "falk_konold_dp"},
    )
    assert summary["n_steps"] == 9
    assert summary["n_incumbent_changes"] == 0
    assert summary["n_steps_discovered_incumbent"] == 0
    assert summary["changes"] == []


def test_summary_requires_annotated_rows():
    with pytest.raises(KeyError, match="incumbent_changed"):
        summarise_incumbents([_step(0, "a")], {"a"})


def test_summary_of_an_empty_trajectory_has_no_final_incumbent():
    summary = summarise_incumbents([], {"a"})
    assert summary["n_steps"] == 0
    assert summary["final_incumbent"] is None


# ── the starting set ─────────────────────────────────────────────────


def test_starting_model_set_is_the_seed_step_model_set():
    history = [
        _history_entry(0, None, "a", posteriors={"a": 0.6, "b": 0.4}),
        _history_entry(1, 0, "c", posteriors={"a": 0.1, "b": 0.1, "c": 0.8}),
    ]
    assert starting_model_set(history) == frozenset({"a", "b"})


def test_starting_model_set_rejects_a_history_that_does_not_open_with_the_seed_step():
    with pytest.raises(ValueError, match="seed step"):
        starting_model_set([_history_entry(1, 0, "a")])
    with pytest.raises(ValueError, match="seed step"):
        starting_model_set([_history_entry(0, 0, "a")])
    with pytest.raises(ValueError, match="empty"):
        starting_model_set([])
    with pytest.raises(ValueError, match="no models"):
        starting_model_set([_history_entry(0, None, "a", posteriors={})])


# ── assembling steps from histories ──────────────────────────────────


def test_steps_from_histories_numbers_experiments_and_global_steps():
    histories = [
        [_history_entry(0, None, "a"), _history_entry(1, 0, "b")],
        [_history_entry(0, None, "b")],
    ]
    steps = steps_from_histories(histories)
    assert steps == [
        {"experiment": 1, "step": 0, "iteration": None, "global_step": 0, "best_model": "a"},
        {"experiment": 1, "step": 1, "iteration": 0, "global_step": 1, "best_model": "b"},
        {"experiment": 2, "step": 0, "iteration": None, "global_step": 2, "best_model": "b"},
    ]


def test_steps_from_histories_rejects_an_empty_history():
    with pytest.raises(ValueError, match="experiment 2"):
        steps_from_histories([[_history_entry(0, None, "a")], []])


def test_incumbent_summary_for_histories_uses_the_first_seed_step_as_the_starting_set():
    histories = [
        [_history_entry(0, None, "a", posteriors={"a": 0.5, "b": 0.5}), _history_entry(1, 0, "a")],
        [_history_entry(0, None, "a"), _history_entry(1, 0, "new")],
    ]
    summary = incumbent_summary_for_histories(histories)
    assert summary["starting_models"] == ["a", "b"]
    assert summary["n_steps"] == 4
    assert summary["n_incumbent_changes"] == 1
    assert summary["n_steps_discovered_incumbent"] == 1
    assert summary["final_incumbent"] == "new"


def _write_history(root: Path, exp_num: int, history) -> None:
    loop_dir = root / f"experiment{exp_num}" / "model_loop"
    loop_dir.mkdir(parents=True)
    (loop_dir / "history.json").write_text(json.dumps(history), encoding="utf-8")


def test_histories_from_run_tree_reads_experiments_in_order(tmp_path):
    # Written out of order and past single digits, to pin numeric ordering.
    for n in (10, 2, 1, 3, 4, 5, 6, 7, 8, 9):
        _write_history(tmp_path, n, [_history_entry(0, None, f"m{n}")])
    histories = histories_from_run_tree(tmp_path)
    assert [h[0]["best_model"] for h in histories] == [f"m{n}" for n in range(1, 11)]


def test_histories_from_run_tree_rejects_a_gap_or_no_experiments(tmp_path):
    with pytest.raises(FileNotFoundError, match="experiment"):
        histories_from_run_tree(tmp_path)
    _write_history(tmp_path, 1, [_history_entry(0, None, "a")])
    _write_history(tmp_path, 3, [_history_entry(0, None, "a")])
    with pytest.raises(FileNotFoundError, match="experiment2"):
        histories_from_run_tree(tmp_path)


def _archive(tmp_path: Path, members: dict) -> Path:
    staging = tmp_path / "staging"
    for rel, payload in members.items():
        path = staging / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    tar_path = tmp_path / "agent_runs.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(staging)))
    return tar_path


def test_histories_from_archive_reads_the_cell_histories_in_order(tmp_path):
    tar_path = _archive(tmp_path, {
        "_runs/gt/experiment2/model_loop/history.json": [_history_entry(0, None, "b")],
        "_runs/gt/experiment1/model_loop/history.json": [_history_entry(0, None, "a")],
        "_runs/gt/experiment1/cognitive_models/models_manifest.yaml": {},
        "_runs/gt/experiment1/model_loop/iter_0/candidate_0/history.json": [{"decoy": True}],
    })
    histories = histories_from_archive(tar_path)
    assert [h[0]["best_model"] for h in histories] == ["a", "b"]


def test_histories_from_archive_rejects_ambiguous_or_gapped_archives(tmp_path):
    two_cells = _archive(tmp_path / "two", {
        "_runs/gt1/experiment1/model_loop/history.json": [_history_entry(0, None, "a")],
        "_runs/gt2/experiment1/model_loop/history.json": [_history_entry(0, None, "a")],
    })
    with pytest.raises(ValueError, match="more than one run"):
        histories_from_archive(two_cells)
    gapped = _archive(tmp_path / "gap", {
        "_runs/gt/experiment1/model_loop/history.json": [_history_entry(0, None, "a")],
        "_runs/gt/experiment3/model_loop/history.json": [_history_entry(0, None, "a")],
    })
    with pytest.raises(FileNotFoundError, match="experiment2"):
        histories_from_archive(gapped)
    empty = _archive(tmp_path / "empty", {"_runs/gt/nothing.json": {}})
    with pytest.raises(FileNotFoundError, match="history.json"):
        histories_from_archive(empty)
