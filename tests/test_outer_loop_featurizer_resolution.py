"""Regression: the project featurizer must resolve from the assets dir.

After experiment outputs moved to ``data/outer_loop/<project>/experimentN``,
``exp_dir.parent`` points at the data tree, which holds no ``preprocess.py``.
The featurizer is a project *asset* and must be resolved via
``outer_project_dir(<project>)`` instead. These tests pin that resolution so the
asset/data split cannot silently drop featurization again.
"""

from __future__ import annotations

from pathlib import Path

from src.pipelines.outer_loop import orchestrator as orch


def test_inner_loop_resolves_featurizer_from_assets_dir(tmp_path, monkeypatch):
    project = "subjective_randomness"
    # Mimic the new layout: exp_dir lives under a data tree, so exp_dir.parent
    # is NOT where preprocess.py lives.
    exp_dir = tmp_path / "data" / "outer_loop" / project / "experiment1"
    (exp_dir / "cognitive_models").mkdir(parents=True)

    captured: dict[str, Path] = {}

    def fake_loader(project_dir: Path):
        captured["dir"] = project_dir
        return None

    monkeypatch.setattr(
        orch,
        "_pooled_response_rows",
        lambda e: [{"sequence_a": "HT", "sequence_b": "TH", "chose_left": "1"}],
    )
    monkeypatch.setattr(orch, "_load_project_featurizer", fake_loader)
    monkeypatch.setattr(orch, "_write_feature_csv", lambda rows, fz, out: out)
    monkeypatch.setattr(
        orch, "_export_inner_loop_model", lambda e, l, *, best_model: e
    )
    monkeypatch.setattr(
        "src.pipelines.inner_loop.pymc_orchestrator.run_pymc_inner_loop",
        lambda *a, **k: {"best_model": "stub_best"},
    )

    orch.run_inner_model_loop_programmatic(exp_dir, max_iterations=0, candidate_count=0)

    # Must look in the assets dir for the project, not the data-output parent.
    assert captured["dir"] == orch.outer_project_dir(project)
    assert captured["dir"] != exp_dir.parent
