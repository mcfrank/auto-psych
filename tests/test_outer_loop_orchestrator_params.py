"""Unit tests for holdout-aware seeding and inner-loop parameter threading.

`seed_experiment_models_from_project` gains an `exclude` option (hold one seed
model out as ground truth) that must fail loudly on unknown names or an empty
remainder. `run_inner_model_loop_programmatic` gains `cache_dir`, `project_id`,
and `agent_timeout_sec` so a harness with a non-standard experiment layout can
share the MCMC cache and resolve project assets explicitly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.pipelines.outer_loop import orchestrator as orch

PROJECT = "subjective_randomness"
SEED_MANIFEST = orch.project_seed_models_dir(PROJECT) / "models_manifest.yaml"


def _seeded_names(exp_dir: Path) -> set[str]:
    manifest = yaml.safe_load(
        (exp_dir / "cognitive_models" / "models_manifest.yaml").read_text(
            encoding="utf-8"
        )
    )
    return {m["name"] for m in manifest["models"]}


# ── seed_experiment_models_from_project exclude ─────────────────────


def test_seed_exclude_filters_files_and_manifest(tmp_path):
    exp_dir = tmp_path / "experiment1"
    assert orch.seed_experiment_models_from_project(
        exp_dir, PROJECT, exclude=("prototype_similarity",)
    )
    assert not (exp_dir / "cognitive_models" / "prototype_similarity.py").exists()
    assert (exp_dir / "cognitive_models" / "bayesian_diagnosticity.py").exists()
    assert _seeded_names(exp_dir) == {
        "bayesian_diagnosticity",
        "encoding_compressibility",
    }


def test_seed_exclude_unknown_name_raises(tmp_path):
    with pytest.raises(ValueError, match="not_a_seed_model"):
        orch.seed_experiment_models_from_project(
            tmp_path / "experiment1", PROJECT, exclude=("not_a_seed_model",)
        )


def test_seed_exclude_all_models_raises(tmp_path):
    everything = tuple(
        m["name"]
        for m in yaml.safe_load(SEED_MANIFEST.read_text(encoding="utf-8"))["models"]
    )
    with pytest.raises(ValueError, match="empt"):
        orch.seed_experiment_models_from_project(
            tmp_path / "experiment1", PROJECT, exclude=everything
        )


def test_seed_without_exclude_copies_manifest_verbatim(tmp_path):
    exp_dir = tmp_path / "experiment1"
    assert orch.seed_experiment_models_from_project(exp_dir, PROJECT)
    dest_manifest = exp_dir / "cognitive_models" / "models_manifest.yaml"
    assert dest_manifest.read_bytes() == SEED_MANIFEST.read_bytes()


# ── run_inner_model_loop_programmatic threading ─────────────────────


def _run_programmatic_loop(tmp_path, monkeypatch, exp_dir, **kwargs):
    captured: dict = {}

    def fake_inner_loop(responses_path, results_dir, **inner_kwargs):
        captured["inner_kwargs"] = inner_kwargs

    monkeypatch.setattr(
        orch, "_pooled_response_rows", lambda e: [{"chose_left": "1"}]
    )
    monkeypatch.setattr(
        orch, "_load_project_featurizer",
        lambda project_dir: captured.setdefault("featurizer_dir", project_dir) and None,
    )
    monkeypatch.setattr(orch, "_write_feature_csv", lambda rows, fz, out: out)
    monkeypatch.setattr(orch, "_export_inner_loop_model", lambda e, l: e)
    monkeypatch.setattr(
        "src.pipelines.inner_loop.pymc_orchestrator.run_pymc_inner_loop",
        fake_inner_loop,
    )
    orch.run_inner_model_loop_programmatic(
        exp_dir, max_iterations=0, candidate_count=0, **kwargs
    )
    return captured


def test_inner_loop_programmatic_threads_cache_dir_and_timeout(tmp_path, monkeypatch):
    exp_dir = tmp_path / "data" / "outer_loop" / PROJECT / "experiment1"
    (exp_dir / "cognitive_models").mkdir(parents=True)
    cache_dir = tmp_path / "cache"

    captured = _run_programmatic_loop(
        tmp_path, monkeypatch, exp_dir, cache_dir=cache_dir, agent_timeout_sec=123
    )

    assert captured["inner_kwargs"]["cache_dir"] == cache_dir
    assert captured["inner_kwargs"]["agent_timeout_sec"] == 123


def test_inner_loop_programmatic_explicit_project_id_overrides_parent_name(
    tmp_path, monkeypatch
):
    # Holdout layout: experiments live under <gt_model>/, so the parent dir is
    # NOT the project id and the featurizer must resolve via the explicit one.
    exp_dir = tmp_path / "holdout_runs" / "prototype_similarity" / "experiment1"
    (exp_dir / "cognitive_models").mkdir(parents=True)

    captured = _run_programmatic_loop(
        tmp_path, monkeypatch, exp_dir, project_id=PROJECT
    )

    assert captured["featurizer_dir"] == orch.outer_project_dir(PROJECT)
