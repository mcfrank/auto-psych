"""The inner loop writes only raw columns to model_loop/responses.csv.

Every model computes its own features from raw stimulus rows. The pipeline
never runs a featurizer.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from src.pipelines.outer_loop import model_loop_runner as mlr
from src.pipelines.outer_loop.columns import RAW_RESPONSE_COLUMNS


# --- helpers ---

RAW_HEADER = list(RAW_RESPONSE_COLUMNS)


def _raw_row(**overrides):
    base = {
        "sequence_a": "HHT",
        "sequence_b": "THT",
        "participant_id": "0",
        "trial_index": "0",
        "chose_left": "1",
    }
    base.update(overrides)
    return base


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _setup_exp_dir(tmp_path, rows, project_id="subjective_randomness"):
    exp_dir = tmp_path / "data" / "outer_loop" / project_id / "experiment1"
    (exp_dir / "cognitive_models").mkdir(parents=True)
    (exp_dir / "cognitive_models" / "models_manifest.yaml").write_text(
        yaml.safe_dump(
            {"models": [{"name": "falk_konold_dp", "rationale": "seed"}]}
        ),
        encoding="utf-8",
    )
    _write_csv(exp_dir / "data" / "responses.csv", rows)
    return exp_dir


def _patch_inner_loop(monkeypatch):
    captured = {}

    def fake_inner_loop(responses_path, results_dir, **kw):
        captured["responses_path"] = responses_path
        captured["inner_kwargs"] = kw
        return {"best_model": "stub_best"}

    monkeypatch.setattr(
        "src.pipelines.inner_loop.pymc_orchestrator.run_pymc_inner_loop",
        fake_inner_loop,
    )
    monkeypatch.setattr(
        mlr,
        "_export_inner_loop_models",
        lambda e, l, *, best_model, protected_names: e,
    )
    return captured


# --- tests ---


def test_inner_loop_writes_only_raw_columns(tmp_path, monkeypatch):
    """model_loop/responses.csv has exactly the five raw columns."""
    exp_dir = _setup_exp_dir(tmp_path, [_raw_row()])
    monkeypatch.setattr(
        mlr, "_pooled_response_rows", lambda e: [_raw_row()]
    )
    _patch_inner_loop(monkeypatch)

    mlr.run_inner_model_loop_programmatic(
        exp_dir,
        max_iterations=0,
        candidate_count=0,
        project_id="subjective_randomness",
    )

    responses_csv = exp_dir / "model_loop" / "responses.csv"
    with responses_csv.open(encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    assert header == RAW_HEADER


def test_candidate_context_has_no_precomputed_feature_references(
    tmp_path,
):
    """When the CSV has only raw columns, CONTEXT.md must say there are no
    precomputed feature columns and that compute_features/prepare_observed is
    required."""
    from src.pipelines.inner_loop.candidate_agent import _write_candidate_context

    responses_csv = tmp_path / "responses.csv"
    _write_csv(responses_csv, [_raw_row()])
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "models_manifest.yaml").write_text(
        yaml.safe_dump({"models": [{"name": "seed", "rationale": "r"}]}),
        encoding="utf-8",
    )
    (models_dir / "seed.py").write_text("# stub\n", encoding="utf-8")

    candidate_dir = tmp_path / "candidate0"
    _write_candidate_context(
        candidate_dir,
        responses_csv,
        models_dir,
        iteration=0,
        candidate_idx=0,
        candidate_count=1,
        current_posterior=None,
    )

    context = (candidate_dir / "CONTEXT.md").read_text(encoding="utf-8")
    assert "precomputed" not in context.lower()
    assert "compute_features" in context or "prepare_observed" in context
    for col in ("rep_motifs", "occ_n20", "multiscale_imbalance"):
        assert col not in context
    assert "chose_left" in context
