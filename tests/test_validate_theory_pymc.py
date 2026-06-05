"""Fast tests for the PyMC theory validator (no MCMC).

`_validate_theory` must accept an experiment whose `cognitive_models/` holds
agent-written **PyMC models** (module-level `model: pm.Model`) and reject ones
that are missing, unloadable, or lack a proper observed-response container.
Validation only builds the model graph — it never samples.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from src.pipelines.outer_loop.orchestrator import _validate_theory

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "pymc_models"


def _seed(exp_dir: Path, names) -> Path:
    models_dir = exp_dir / "cognitive_models"
    models_dir.mkdir(parents=True)
    for name in names:
        shutil.copyfile(FIXTURE_DIR / f"{name}.py", models_dir / f"{name}.py")
    manifest = "models:\n" + "".join(f"  - name: {n}\n" for n in names)
    (models_dir / "models_manifest.yaml").write_text(manifest, encoding="utf-8")
    return models_dir


def test_valid_pymc_models_pass(tmp_path):
    _seed(tmp_path, ["bayesian_fair_coin", "representativeness"])
    ok, msg = _validate_theory(tmp_path)
    assert ok, msg


def test_missing_manifest_fails(tmp_path):
    (tmp_path / "cognitive_models").mkdir(parents=True)
    ok, msg = _validate_theory(tmp_path)
    assert not ok
    assert "manifest" in msg.lower()


def test_manifest_model_without_file_fails(tmp_path):
    models_dir = tmp_path / "cognitive_models"
    models_dir.mkdir(parents=True)
    (models_dir / "models_manifest.yaml").write_text(
        "models:\n  - name: ghost\n", encoding="utf-8"
    )
    ok, msg = _validate_theory(tmp_path)
    assert not ok
    assert "ghost" in msg


def test_model_without_observed_rv_fails(tmp_path):
    models_dir = tmp_path / "cognitive_models"
    models_dir.mkdir(parents=True)
    # A module-level pm.Model with NO observed RV — must be rejected.
    (models_dir / "no_obs.py").write_text(
        "import numpy as np\n"
        "import pymc as pm\n"
        "with pm.Model() as model:\n"
        "    x = pm.Data('x', np.zeros(1, dtype='int64'))\n"
        "    pm.Normal('z', mu=0, sigma=1)\n",
        encoding="utf-8",
    )
    (models_dir / "models_manifest.yaml").write_text(
        "models:\n  - name: no_obs\n", encoding="utf-8"
    )
    ok, msg = _validate_theory(tmp_path)
    assert not ok
    assert "no_obs" in msg
