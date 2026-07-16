"""Across experiments, only the previous ``cognitive_models/`` carries forward —
the original hypotheses plus the single exported best ``inner_loop_model``. The
inner loop's intermediate zoo candidates (``iterN_candidateM``) live only in
``model_loop/models/`` and must NEVER be copied into a theory model set.

The model-set validator enforces this so that, paired with the repair loop, an agent
that over-copies the zoo is told to fix it rather than silently bloating the set.
"""

from __future__ import annotations

import yaml

from src.pipelines.outer_loop.orchestrator import _validate_model_set


def _write_manifest(exp_dir, models):
    td = exp_dir / "cognitive_models"
    td.mkdir(parents=True, exist_ok=True)
    (td / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": models}), encoding="utf-8"
    )


def test_model_set_rejects_inner_loop_zoo_candidates(tmp_path):
    _write_manifest(
        tmp_path,
        [
            {"name": "alternation_bias", "rationale": "alternation rate cue"},
            {"name": "iter0_candidate1", "rationale": "carried over from the zoo"},
        ],
    )
    ok, msg = _validate_model_set(tmp_path)
    assert not ok
    assert "iter0_candidate1" in msg
    assert "model_loop" in msg or "candidate" in msg.lower()


def test_zoo_rule_ignores_best_and_normal_names(tmp_path):
    """`inner_loop_model` (the exported best) and ordinary names must not trip the
    zoo rule — they fail later for a different reason (missing .py), proving the
    rule is specific to `iterN_candidateM` names."""
    _write_manifest(
        tmp_path, [{"name": "inner_loop_model", "rationale": "best from prev exp"}]
    )
    ok, msg = _validate_model_set(tmp_path)
    assert not ok  # still invalid (no .py file), but NOT due to the zoo rule
    assert "candidate" not in msg.lower()  # the zoo rule did not fire
    assert "model file" in msg or ".py" in msg  # failed for the real reason: missing file
