"""A coding-stage validation failure in the holdout pipeline must feed the error
back to the agent and let it repair its output in place — not abort the whole
holdout task. Only an exhausted repair budget raises. Same self-correction as the
outer-loop runner; this is what would have saved the malformed-manifest / bad-PyMC
theory failures (_3/_12/_13).
"""

from __future__ import annotations

from pathlib import Path

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
        "2_design", tmp_path, allowed_dirs=[tmp_path],
        agent_timeout_sec=10, backend="opencode", max_repairs=2,
    )
    assert len(fb) == 2  # re-spawned to repair
    assert fb[0] is None
    assert "YAML" in (fb[1] or "")  # error fed back to the agent


def test_holdout_raises_after_exhausting_repairs(tmp_path, monkeypatch):
    _stub(monkeypatch, [False, False, False])
    with pytest.raises(RuntimeError):
        holdout_recovery._spawn_with_repair(
            "2_design", tmp_path, allowed_dirs=[tmp_path],
            agent_timeout_sec=10, backend="opencode", max_repairs=2,
        )


def test_holdout_post_spawn_runs_before_validation(tmp_path, monkeypatch):
    """The design stage runs its post_spawn hook (the deterministic EIG fallback
    that finishes stimuli.json) after each spawn and before validation."""
    _stub(monkeypatch, [True])
    calls = []
    holdout_recovery._spawn_with_repair(
        "2_design", tmp_path, allowed_dirs=[tmp_path], agent_timeout_sec=10,
        backend="opencode", prompt_key="2_design",
        post_spawn=lambda: calls.append("ensure_stimuli"), max_repairs=2,
    )
    assert calls == ["ensure_stimuli"]


def test_holdout_design_uses_human_experiment_prompt(tmp_path, monkeypatch):
    """Recovery must design experiments with the SAME prompt as the live human
    experiment: the default ``2_design`` agent (proposes candidates AND scores
    them by EIG). A candidates-only prompt variant once made the recovery
    designs diverge from the human runs; that prompt file has been deleted, and
    this test guards against reintroducing a recovery-specific design prompt.
    """

    spawns: list[tuple[str, object]] = []

    def fake_spawn(
        agent_key, exp_dir, allowed_dirs=None, timeout_secs=900, backend=None,
        prompt_key=None, repair_feedback=None,
    ):
        spawns.append((agent_key, prompt_key))
        return True, ""

    def fake_inner_loop(exp_dir, **kwargs):
        loop = Path(exp_dir) / "model_loop"
        loop.mkdir(parents=True, exist_ok=True)
        (loop / "history.json").write_text("[]", encoding="utf-8")

    monkeypatch.setattr(holdout_recovery, "spawn_cc_agent", fake_spawn)
    monkeypatch.setattr(holdout_recovery, "validate_cc_output", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(holdout_recovery, "resolve_generating_params", lambda *a, **k: {})
    monkeypatch.setattr(
        holdout_recovery, "ensure_experiment_dirs",
        lambda d: Path(d).mkdir(parents=True, exist_ok=True),
    )
    monkeypatch.setattr(holdout_recovery, "init_registry", lambda *a, **k: None)
    monkeypatch.setattr(holdout_recovery, "seed_experiment_models_from_project", lambda *a, **k: True)
    monkeypatch.setattr(holdout_recovery, "write_context", lambda *a, **k: None)
    monkeypatch.setattr(holdout_recovery, "_ensure_design_stimuli", lambda *a, **k: None)
    monkeypatch.setattr(
        holdout_recovery, "load_stimuli",
        lambda p: [{"sequence_a": "HT", "sequence_b": "HH"}],
    )
    monkeypatch.setattr(
        holdout_recovery, "generate_responses",
        lambda *a, **k: [{"sequence_a": "HT", "sequence_b": "HH", "chose_left": 1}],
    )
    monkeypatch.setattr(holdout_recovery, "write_responses_csv", lambda *a, **k: None)
    monkeypatch.setattr(holdout_recovery, "run_inner_model_loop_programmatic", fake_inner_loop)
    monkeypatch.setattr(holdout_recovery, "update_registry_from_interpretation", lambda *a, **k: None)

    holdout_recovery.run_holdout_experiments(
        "prototype_similarity", {}, tmp_path / "run",
        seed_models_dir=tmp_path / "seeds",
        n_experiments=1, n_participants=2,
        inner_loop_iterations=1, candidate_count=1,
        fit_kwargs={}, backend="opencode",
    )

    design_prompts = [pk for (ak, pk) in spawns if ak == "2_design"]
    assert design_prompts == ["2_design"], spawns
