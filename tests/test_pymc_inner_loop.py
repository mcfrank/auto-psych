"""Integration test for the PyMC-native inner model loop.

With ``max_iterations=0`` no coding agent is spawned: the loop simply seeds its
model set from ``seed_models_dir``, fits each model to the responses via MCMC,
scores them by ELPD-LOO, and exports the best. Slow (two NUTS fits) — runs in
~20 s. Mark ``slow``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.pipelines.inner_loop.pymc_orchestrator import run_pymc_inner_loop

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pymc_models"


@pytest.mark.slow
def test_inner_loop_seeds_fits_and_selects_best(tmp_path):
    results_dir = tmp_path / "model_loop"

    result = run_pymc_inner_loop(
        responses_path=FIXTURE_DIR / "responses.csv",
        results_dir=results_dir,
        seed_models_dir=FIXTURE_DIR,
        max_iterations=0,
        fit_kwargs={"draws": 300, "tune": 300, "chains": 2},
    )

    posterior = json.loads((results_dir / "model_posterior.json").read_text())
    assert set(posterior["posteriors"]) == {"bayesian_fair_coin", "representativeness"}
    assert abs(sum(posterior["posteriors"].values()) - 1.0) < 1e-4
    # Data were simulated from bayesian_fair_coin → it must win.
    assert posterior["posteriors"]["bayesian_fair_coin"] > 0.7

    # The best model is exported as a standalone PyMC model file + manifest.
    best_model = results_dir / "best_model.py"
    assert best_model.exists()
    assert "pm.Model" in best_model.read_text()
    assert (results_dir / "report.md").read_text().strip()

    models_dir = results_dir / "models"
    assert (models_dir / "bayesian_fair_coin.py").exists()
    assert (models_dir / "representativeness.py").exists()
    assert (models_dir / "models_manifest.yaml").exists()

    assert result["best_model"] == "bayesian_fair_coin"
    assert result["posteriors"]["bayesian_fair_coin"] > 0.7
