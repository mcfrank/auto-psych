"""A coding-stage validation failure must feed the error back to the agent so it
can repair its output in place — not crash the whole pipeline. Only an exhausted
repair budget aborts the run (you must never deploy an unfixable experiment).
"""

from __future__ import annotations

import pytest

from src.pipelines.outer_loop import run


def _run_implement(monkeypatch, tmp_path, validate_results, *, max_repairs, validate=True):
    """Drive _run_agent for the 3_implement coding stage with its seams stubbed.

    ``validate_results`` is the sequence of booleans ``validate_cc_output`` returns.
    Returns the list of ``repair_feedback`` values each spawn was called with.
    """
    monkeypatch.setattr(run, "write_context", lambda **k: None)
    monkeypatch.setattr(run, "outer_project_dir", lambda project_id: tmp_path)

    spawn_feedback = []

    def fake_spawn(agent_key, exp_dir, allowed_dirs=None, backend=None, repair_feedback=None, **k):
        spawn_feedback.append(repair_feedback)
        return True, ""

    monkeypatch.setattr(run, "spawn_cc_agent", fake_spawn)

    remaining = list(validate_results)

    def fake_validate(agent_key, exp_dir):
        ok = remaining.pop(0)
        return ok, "ok" if ok else "formatting: literal Markdown bold (`**...**`)"

    monkeypatch.setattr(run, "validate_cc_output", fake_validate)

    run._run_agent(
        agent_key="3_implement",
        exp_dir=tmp_path,
        project_id="p",
        exp_num=1,
        mode="live",
        n_participants=1,
        prev_exp_dir=None,
        validate=validate,
        backend="opencode",
        max_validation_repairs=max_repairs,
    )
    return spawn_feedback


def test_validation_failure_then_pass_repairs_without_crashing(tmp_path, monkeypatch):
    feedback = _run_implement(monkeypatch, tmp_path, [False, True], max_repairs=2)
    assert len(feedback) == 2  # re-spawned once to repair
    assert feedback[0] is None  # first attempt gets no feedback
    assert "Markdown bold" in (feedback[1] or "")  # the error is fed back to the agent


def test_exhausting_the_repair_budget_aborts(tmp_path, monkeypatch):
    with pytest.raises(SystemExit):
        _run_implement(monkeypatch, tmp_path, [False, False, False], max_repairs=2)


def test_no_validation_means_a_single_spawn_and_no_repair(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("validate_cc_output must not run when validate=False")

    monkeypatch.setattr(run, "validate_cc_output", boom)
    feedback = _run_implement(monkeypatch, tmp_path, [], max_repairs=2, validate=False)
    assert feedback == [None]  # spawned exactly once, never repaired
