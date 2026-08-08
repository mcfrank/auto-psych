"""Regression: the project featurizer must resolve from the assets dir, and
loading it must fail loudly.

After experiment outputs moved to ``data/outer_loop/<project>/experimentN``,
``exp_dir.parent`` points at the data tree, which holds no ``preprocess.py``.
The featurizer is a project *asset* and must be resolved via
``outer_project_dir(<project>)`` instead. These tests pin that resolution so the
asset/data split cannot silently drop featurization again.

The loader itself was hand-rolled at three call sites (collect, eig,
orchestrator) with three different failure policies; two returned ``None`` when
the module existed but exposed no ``featurize_stimulus``, which silently fed
unfeaturized rows downstream. One shared loader now raises on every failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.pipelines.outer_loop import orchestrator as orch
from src.pipelines.outer_loop.featurizer import load_featurizer


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


def _write_preprocess(directory: Path, body: str) -> Path:
    path = directory / "preprocess.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_featurizer_returns_the_modules_featurize_stimulus(tmp_path):
    path = _write_preprocess(
        tmp_path, "def featurize_stimulus(a, b):\n    return {'n_a': len(a)}\n"
    )
    assert load_featurizer(path)("HHT", "T") == {"n_a": 3}


def test_load_featurizer_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="featurize module not found"):
        load_featurizer(tmp_path / "preprocess.py")


def test_load_featurizer_raises_when_module_has_no_featurize_stimulus(tmp_path):
    path = _write_preprocess(tmp_path, "SOMETHING_ELSE = 1\n")
    with pytest.raises(AttributeError, match="featurize_stimulus"):
        load_featurizer(path)


def test_project_featurizer_is_none_only_when_the_project_has_no_preprocess(tmp_path):
    # "This project does not featurize" is legitimate and stays None; "this
    # project's preprocess.py is broken" must not look the same.
    assert orch._load_project_featurizer(tmp_path) is None
    _write_preprocess(tmp_path, "SOMETHING_ELSE = 1\n")
    with pytest.raises(AttributeError, match="featurize_stimulus"):
        orch._load_project_featurizer(tmp_path)
