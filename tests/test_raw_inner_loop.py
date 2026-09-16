"""In a raw_features run the inner loop must never see engineered columns.

The bug this guards against: arm C's ``run_inner_model_loop_programmatic``
always loaded the project featurizer and wrote 59-column
``model_loop/responses.csv`` even when ``data/responses.csv`` was raw (5
columns). Candidates read the 59-column file, so the run was not raw at all.

After P2 ``run_inner_model_loop_programmatic(raw_features=True)`` skips the
featurizer, writes only the five raw columns, and raises if the pooled rows
carry anything extra.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import yaml

from src.pipelines.outer_loop import orchestrator as orch
from src.pipelines.outer_loop.featurizer import RAW_RESPONSE_COLUMNS


# ─── helpers ───

RAW_HEADER = list(RAW_RESPONSE_COLUMNS)
FEATURIZED_EXTRA = ["n_a", "n_b", "h_a", "h_b", "rep_motifs_a", "rep_motifs_b"]


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


def _featurized_row(**overrides):
    row = _raw_row()
    for col in FEATURIZED_EXTRA:
        row[col] = "0.5"
    row.update(overrides)
    return row


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
    """Monkeypatch everything except _write_feature_csv so we can inspect the CSV."""
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
        orch,
        "_export_inner_loop_models",
        lambda e, l, *, best_model, protected_names: e,
    )
    return captured


# ─── tests ───


def test_raw_features_true_writes_only_raw_columns(tmp_path, monkeypatch):
    """raw_features=True: model_loop/responses.csv has exactly the five raw
    columns, even though the project has a featurizer."""
    exp_dir = _setup_exp_dir(tmp_path, [_raw_row()])
    monkeypatch.setattr(
        orch, "_pooled_response_rows", lambda e: [_raw_row()]
    )
    _patch_inner_loop(monkeypatch)

    orch.run_inner_model_loop_programmatic(
        exp_dir,
        max_iterations=0,
        candidate_count=0,
        project_id="subjective_randomness",
        raw_features=True,
    )

    responses_csv = exp_dir / "model_loop" / "responses.csv"
    with responses_csv.open(encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    assert header == RAW_HEADER


def test_raw_features_false_preserves_featurized_columns(tmp_path, monkeypatch):
    """raw_features=False (default): the featurizer runs, so the CSV has the
    extra columns the project supplies."""
    exp_dir = _setup_exp_dir(tmp_path, [_featurized_row()])
    monkeypatch.setattr(
        orch, "_pooled_response_rows", lambda e: [_raw_row()]
    )
    _patch_inner_loop(monkeypatch)

    orch.run_inner_model_loop_programmatic(
        exp_dir,
        max_iterations=0,
        candidate_count=0,
        project_id="subjective_randomness",
    )

    responses_csv = exp_dir / "model_loop" / "responses.csv"
    with responses_csv.open(encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    # The project featurizer adds columns beyond the raw set.
    assert len(header) > len(RAW_HEADER)
    for col in RAW_HEADER:
        assert col in header


def test_raw_features_true_raises_on_engineered_columns_in_pool(
    tmp_path, monkeypatch
):
    """If the pooled rows already carry engineered columns, a raw run must
    raise — that means the upstream data is not truly raw."""
    exp_dir = _setup_exp_dir(tmp_path, [_featurized_row()])
    monkeypatch.setattr(
        orch, "_pooled_response_rows", lambda e: [_featurized_row()]
    )
    _patch_inner_loop(monkeypatch)

    with pytest.raises(ValueError, match="raw_features"):
        orch.run_inner_model_loop_programmatic(
            exp_dir,
            max_iterations=0,
            candidate_count=0,
            project_id="subjective_randomness",
            raw_features=True,
        )


def test_candidate_context_in_raw_mode_has_no_precomputed_feature_references(
    tmp_path,
):
    """When the CSV has only raw columns, CONTEXT.md must say there are no
    precomputed feature columns and that compute_features/prepare_observed is
    required."""
    from src.pipelines.inner_loop.pymc_orchestrator import _write_candidate_context

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
    # Must not claim precomputed feature columns exist.
    assert "precomputed" not in context.lower()
    # Must say compute_features or prepare_observed is required.
    assert "compute_features" in context or "prepare_observed" in context
    # Must not mention specific featurized column names.
    for col in ("rep_motifs", "occ_n20", "multiscale_imbalance"):
        assert col not in context
    # Must say chose_left is the only numeric column.
    assert "chose_left" in context
