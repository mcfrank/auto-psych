"""Tests for PyMC-based synthetic data generation in collect.

`_generate_from_pymc_models` samples synthetic responses from the theorist's
PyMC models' prior-predictive p_left (no MCMC fit — the prior is the generative
distribution for synthetic participants). Each raw stimulus is featurized first
so the models can read their `pm.Data` columns.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from src.pipelines.outer_loop.collect import _generate_from_pymc_models

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
    shutil.copyfile(
        FIXTURE_DIR / "models_manifest.yaml", models_dir / "models_manifest.yaml"
    )
    return models_dir


@pytest.mark.slow
def test_generate_from_pymc_models_shapes_and_columns(tmp_path):
    models_dir = _seed(tmp_path)
    stimuli = [
        {"sequence_a": "HHHT", "sequence_b": "HTHT"},
        {"sequence_a": "HHHHHHHH", "sequence_b": "HTHTHTHT"},
    ]
    rows = _generate_from_pymc_models(
        stimuli,
        ["bayesian_fair_coin", "representativeness"],
        n_participants=3,
        models_dir=models_dir,
        featurize_path=FEATURIZE,
        n_samples=100,
        seed=0,
    )

    assert len(rows) == 3 * len(stimuli)
    for r in rows:
        assert r["chose_left"] in (0, 1)
        assert r["chose_right"] == 1 - r["chose_left"]
        assert r["sequence_a"] and r["sequence_b"]
        assert r["model"] in {"bayesian_fair_coin", "representativeness"}
        assert {"participant_id", "trial_index"} <= set(r)


@pytest.mark.slow
def test_generate_is_deterministic_under_fixed_seed(tmp_path):
    models_dir = _seed(tmp_path)
    stimuli = [{"sequence_a": "HHHT", "sequence_b": "HTHT"}]
    kw = dict(models_dir=models_dir, featurize_path=FEATURIZE, n_samples=100, seed=7)
    rows_a = _generate_from_pymc_models(stimuli, ["bayesian_fair_coin"], 2, **kw)
    rows_b = _generate_from_pymc_models(stimuli, ["bayesian_fair_coin"], 2, **kw)
    assert [r["chose_left"] for r in rows_a] == [r["chose_left"] for r in rows_b]
