"""write_context writes a stage's CONTEXT.md with the experiment's key paths.

There is no design agent: the design stage is the programmatic exhaustive EIG
selection, so no stage context inlines model hypotheses — CONTEXT.md is the
same path listing for every coding stage.
"""

from __future__ import annotations

from src.pipelines.outer_loop.orchestrator import write_context


def test_context_lists_key_paths(tmp_path):
    path = write_context(tmp_path, "3_implement", "subjective_randomness", 2)
    text = path.read_text(encoding="utf-8")
    assert "agent 3_implement" in text
    assert "**Experiment number:** 2" in text
    assert str(tmp_path / "design") in text
    assert str(tmp_path / "cognitive_models") in text
    assert str(tmp_path / "data" / "responses.csv") in text


def test_context_does_not_require_models(tmp_path):
    # No cognitive_models/ on disk: the context is just paths, so nothing
    # should fail when the model set has not been seeded yet.
    path = write_context(tmp_path, "3_implement", "subjective_randomness", 1)
    assert path.exists()


def test_context_includes_previous_experiment_paths(tmp_path):
    prev = tmp_path / "experiment1"
    prev.mkdir()
    exp = tmp_path / "experiment2"
    exp.mkdir()
    path = write_context(exp, "3_implement", "subjective_randomness", 2, prev_exp_dir=prev)
    text = path.read_text(encoding="utf-8")
    assert str(prev / "model_registry.yaml") in text
