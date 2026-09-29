"""The snapshot scorer finds the latest experiment of an in-progress cell."""

from __future__ import annotations

import json

import pytest

from scripts.subjective_randomness.snapshot_recovery import latest_loop


def _history(tree, exp, steps):
    loop = tree / f"experiment{exp}" / "model_loop"
    loop.mkdir(parents=True)
    (loop / "history.json").write_text(json.dumps(steps), encoding="utf-8")
    return loop


def test_latest_loop_is_the_highest_experiment_with_a_step(tmp_path):
    _history(tmp_path, 1, [{"best_model": "a"}])
    loop2 = _history(tmp_path, 2, [{"best_model": "b"}])
    _history(tmp_path, 10, [])  # started, no step yet: not scoreable
    assert latest_loop(tmp_path) == (loop2, 2)


def test_experiment_numbers_sort_numerically_not_as_text(tmp_path):
    _history(tmp_path, 2, [{"best_model": "b"}])
    loop10 = _history(tmp_path, 10, [{"best_model": "c"}])
    assert latest_loop(tmp_path) == (loop10, 10)


def test_a_cell_with_no_step_yet_fails_loudly(tmp_path):
    _history(tmp_path, 1, [])
    with pytest.raises(FileNotFoundError, match="no inner-loop step"):
        latest_loop(tmp_path)
