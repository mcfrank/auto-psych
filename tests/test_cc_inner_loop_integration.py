"""Integration test: the outer loop's inner-model-loop step, PyMC contract.

`run_inner_model_loop_programmatic` pools responses, fits the experiment's PyMC
models by MCMC, scores them by ELPD-LOO, and exports the best model back into
`cognitive_models/`. Slow (MCMC) — run with ``-m slow``.
"""
import json
import shutil
from pathlib import Path

import pytest
import yaml

from src.pipelines.outer_loop.orchestrator import run_inner_model_loop_programmatic

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pymc_models"


@pytest.mark.slow
def test_inner_model_loop_exports_best_pymc_model(tmp_path):
    exp_dir = tmp_path / "project" / "experiment1"

    # Seed the experiment's model set with the two fixture PyMC models.
    models_dir = exp_dir / "cognitive_models"
    models_dir.mkdir(parents=True)
    for name in ("bayesian_fair_coin", "representativeness"):
        shutil.copyfile(FIXTURE_DIR / f"{name}.py", models_dir / f"{name}.py")
    shutil.copyfile(FIXTURE_DIR / "models_manifest.yaml", models_dir / "models_manifest.yaml")

    # Pooled responses already carry the feature columns the models read.
    data_dir = exp_dir / "data"
    data_dir.mkdir(parents=True)
    shutil.copyfile(FIXTURE_DIR / "responses.csv", data_dir / "responses.csv")

    loop_dir = run_inner_model_loop_programmatic(
        exp_dir,
        max_iterations=0,
        candidate_count=0,
        fit_kwargs={"draws": 300, "tune": 300, "chains": 2},
    )

    assert loop_dir == exp_dir / "model_loop"

    posterior = json.loads((loop_dir / "model_posterior.json").read_text())
    assert "posteriors" in posterior
    assert set(posterior["posteriors"]) == {"bayesian_fair_coin", "representativeness"}
    # Data were simulated from bayesian_fair_coin → it wins.
    assert posterior["posteriors"]["bayesian_fair_coin"] > 0.7
    assert (loop_dir / "report.md").read_text().strip()

    # Best model exported verbatim as a PyMC model + listed in the manifest.
    exported = models_dir / "inner_loop_model.py"
    assert exported.exists()
    assert "pm.Model" in exported.read_text()
    manifest = yaml.safe_load((models_dir / "models_manifest.yaml").read_text())
    names = [m["name"] if isinstance(m, dict) else m for m in manifest["models"]]
    assert "inner_loop_model" in names
