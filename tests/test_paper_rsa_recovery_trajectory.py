"""Unit tests for ``scripts/paper/rsa_recovery_trajectory.py`` (no fitting):
the guard keeping trial-level data out of the checkout, and what it reads
from a cell's record."""

from __future__ import annotations

import json

import pytest
from pyprojroot import here

from scripts.paper import rsa_recovery_trajectory as rt


def test_a_work_directory_inside_the_checkout_is_refused(tmp_path):
    with pytest.raises(ValueError, match="inside the checkout"):
        rt.check_outside_checkout(here() / "data" / "paper" / "work")
    rt.check_outside_checkout(tmp_path)


def test_a_step_best_model_is_found_among_the_pruned_models(tmp_path):
    (tmp_path / "models" / "pruned").mkdir(parents=True)
    (tmp_path / "models" / "pruned" / "rsa_l2.py").write_text("")
    assert rt.model_file(tmp_path, "rsa_l2") == tmp_path / "models" / "pruned" / "rsa_l2.py"
    with pytest.raises(FileNotFoundError):
        rt.model_file(tmp_path, "missing")


def test_steps_read_each_steps_best_model_from_the_committed_history():
    cell = rt.RUN2_RECORDS / "recovery_literal_rep1"
    steps = rt.steps(cell)
    assert [s["step"] for s in steps] == list(range(6))
    export = json.loads((cell / "export.json").read_text())
    assert steps[-1]["best_model"] == export["best_model"]
