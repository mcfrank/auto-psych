"""Hero-run knobs thread from the CLIs down to run_pymc_inner_loop.

The exploration levers (hints, novelty gate, pruning, candidate parallelism)
are only useful if a launch config can set them without editing source; this
pins the plumbing: outer-loop programmatic wrapper -> inner loop, the inner
CLI's hints-file loader, and tyro parsing of the new flags.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import tyro
import yaml

from src.pipelines.inner_loop.run import Args as InnerArgs, load_hints_file
from src.pipelines.outer_loop import orchestrator as orch


def test_programmatic_wrapper_threads_hero_knobs(tmp_path, monkeypatch):
    exp_dir = tmp_path / "data" / "outer_loop" / "subjective_randomness" / "experiment1"
    (exp_dir / "cognitive_models").mkdir(parents=True)
    captured = {}

    def fake_inner_loop(responses_path, results_dir, **inner_kwargs):
        captured.update(inner_kwargs)
        return {"best_model": "stub_best"}

    monkeypatch.setattr(
        orch, "_pooled_response_rows", lambda e: [{"chose_left": "1"}]
    )
    monkeypatch.setattr(orch, "_load_project_featurizer", lambda project_dir: None)
    monkeypatch.setattr(orch, "_write_feature_csv", lambda rows, fz, out: out)
    monkeypatch.setattr(
        orch, "_export_inner_loop_model", lambda e, l, *, best_model: e
    )
    monkeypatch.setattr(
        "src.pipelines.inner_loop.pymc_orchestrator.run_pymc_inner_loop",
        fake_inner_loop,
    )

    orch.run_inner_model_loop_programmatic(
        exp_dir,
        max_iterations=1,
        candidate_count=7,
        candidate_hints=["lens one", "lens two"],
        novelty_rmse_threshold=0.03,
        prune_dse_multiplier=3.0,
        prune_weight_floor=0.02,
        candidate_parallelism=4,
    )

    assert captured["candidate_hints"] == ["lens one", "lens two"]
    assert captured["novelty_rmse_threshold"] == 0.03
    assert captured["prune_dse_multiplier"] == 3.0
    assert captured["prune_weight_floor"] == 0.02
    assert captured["candidate_parallelism"] == 4


def test_inner_cli_parses_hero_knobs():
    args = tyro.cli(
        InnerArgs,
        args=[
            "--responses", "r.csv",
            "--seed-models", "seeds",
            "--results", "out",
            "--novelty-rmse-threshold", "0.05",
            "--prune-dse-multiplier", "2.5",
            "--prune-weight-floor", "0.005",
            "--candidate-parallelism", "8",
            "--hints-file", "hints.yaml",
        ],
    )
    assert args.novelty_rmse_threshold == 0.05
    assert args.prune_dse_multiplier == 2.5
    assert args.prune_weight_floor == 0.005
    assert args.candidate_parallelism == 8
    assert args.hints_file == Path("hints.yaml")


def test_hints_file_loads_a_yaml_list(tmp_path):
    path = tmp_path / "hints.yaml"
    path.write_text(yaml.safe_dump(["lens a", "lens b"]), encoding="utf-8")
    assert load_hints_file(path) == ["lens a", "lens b"]


def test_hints_file_rejects_non_list(tmp_path):
    path = tmp_path / "hints.yaml"
    path.write_text("just a string", encoding="utf-8")
    with pytest.raises(ValueError, match="list"):
        load_hints_file(path)


def test_hints_file_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_hints_file(tmp_path / "absent.yaml")
