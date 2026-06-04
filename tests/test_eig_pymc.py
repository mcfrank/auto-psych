"""Tests for the PyMC EIG/design annotator.

`annotate` scores candidate stimuli by expected information gain over the PyMC
model set, using prior-predictive p_left (no MCMC fit). It featurizes each raw
stimulus via the project's `featurize_stimulus` before handing it to the models.
Uses prior-predictive sampling (fast-ish, no NUTS) — marked slow to be safe.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.pipelines.outer_loop import eig as eig_mod

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pymc_models"
FEATURIZE = (
    Path(__file__).resolve().parent.parent
    / "src/pipelines/outer_loop/projects/subjective_randomness/preprocess.py"
)


def _seed(tmp_path):
    models_dir = tmp_path / "cognitive_models"
    models_dir.mkdir(parents=True)
    for name in ("bayesian_fair_coin", "representativeness"):
        shutil.copyfile(FIXTURE_DIR / f"{name}.py", models_dir / f"{name}.py")
    shutil.copyfile(FIXTURE_DIR / "models_manifest.yaml", models_dir / "models_manifest.yaml")
    return models_dir


@pytest.mark.slow
def test_annotate_adds_nonnegative_eig_and_sorts(tmp_path):
    models_dir = _seed(tmp_path)
    candidates = [
        {"sequence_a": "HHHHH", "sequence_b": "HHHHH"},   # identical → low EIG
        {"sequence_a": "HHHHHHHH", "sequence_b": "HTHTHTHT"},  # discriminating
    ]
    out = eig_mod.annotate(candidates, models_dir, featurize_path=FEATURIZE, n_samples=100)

    assert len(out) == 2
    for item in out:
        assert "eig" in item
        assert 0.0 <= item["eig"] <= 1.0  # 2 models → ≤ log2(2) = 1 bit
        assert "sequence_a" in item and "sequence_b" in item
    # Sorted descending by EIG.
    assert out[0]["eig"] >= out[1]["eig"]


@pytest.mark.slow
def test_annotate_featurizes_so_models_can_read_columns(tmp_path):
    """Without featurization the models' pm.Data columns are absent → this
    proves the annotator derives n_a/h_a/... from raw sequences."""
    models_dir = _seed(tmp_path)
    candidates = [{"sequence_a": "HHHT", "sequence_b": "HTHT"}]
    out = eig_mod.annotate(candidates, models_dir, featurize_path=FEATURIZE, n_samples=100)
    assert out[0]["eig"] >= 0.0


def test_missing_manifest_raises(tmp_path):
    (tmp_path / "cognitive_models").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        eig_mod.annotate([{"sequence_a": "H", "sequence_b": "T"}], tmp_path / "cognitive_models")
