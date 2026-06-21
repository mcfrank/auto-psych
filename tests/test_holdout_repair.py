"""A coding-stage validation failure in the holdout pipeline must feed the error
back to the agent and let it repair its output in place — not abort the whole
holdout task. Only an exhausted repair budget raises. Same self-correction as the
outer-loop runner; this is what would have saved the malformed-manifest / bad-PyMC
theory failures (_3/_12/_13).
"""

from __future__ import annotations

import pytest

from src.subjective_randomness import holdout_recovery


def _stub(monkeypatch, validate_results):
    """Stub spawn + validate; return the list of repair_feedback each spawn saw."""
    spawn_feedback = []

    def fake_spawn(
        agent_key, exp_dir, allowed_dirs=None, timeout_secs=900, backend=None,
        prompt_key=None, repair_feedback=None,
    ):
        spawn_feedback.append(repair_feedback)
        return True, ""

    monkeypatch.setattr(holdout_recovery, "spawn_cc_agent", fake_spawn)

    remaining = list(validate_results)

    def fake_validate(agent_key, exp_dir):
        ok = remaining.pop(0)
        return ok, "ok" if ok else "Invalid YAML in models_manifest.yaml"

    monkeypatch.setattr(holdout_recovery, "validate_cc_output", fake_validate)
    return spawn_feedback


def test_holdout_repairs_then_passes(tmp_path, monkeypatch):
    fb = _stub(monkeypatch, [False, True])
    holdout_recovery._spawn_with_repair(
        "1_theory", tmp_path, allowed_dirs=[tmp_path],
        agent_timeout_sec=10, backend="opencode", max_repairs=2,
    )
    assert len(fb) == 2  # re-spawned to repair
    assert fb[0] is None
    assert "YAML" in (fb[1] or "")  # error fed back to the agent


def test_holdout_raises_after_exhausting_repairs(tmp_path, monkeypatch):
    _stub(monkeypatch, [False, False, False])
    with pytest.raises(RuntimeError):
        holdout_recovery._spawn_with_repair(
            "1_theory", tmp_path, allowed_dirs=[tmp_path],
            agent_timeout_sec=10, backend="opencode", max_repairs=2,
        )


def test_holdout_post_spawn_runs_before_validation(tmp_path, monkeypatch):
    """The design stage scores the candidate pool into stimuli.json after each
    spawn and before validation — modelled by post_spawn."""
    _stub(monkeypatch, [True])
    calls = []
    holdout_recovery._spawn_with_repair(
        "2_design", tmp_path, allowed_dirs=[tmp_path], agent_timeout_sec=10,
        backend="opencode", prompt_key="2_design_candidates_only",
        post_spawn=lambda: calls.append("ensure_stimuli"), max_repairs=2,
    )
    assert calls == ["ensure_stimuli"]
