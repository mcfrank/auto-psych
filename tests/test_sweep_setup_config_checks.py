"""The sweep setup jobs' consistency check accepts the configs they ship with.

Each setup job (``holdout_setup.sbatch``, ``impossible_holdout_setup.sbatch``)
ends with an inline Python check that the submit script's ``GT_MODELS``, the
setup's own ``SEED_MODELS_REL`` default and the committed config agree. The
impossible setup's default still named the project's ``seed_models/`` after
its config moved to ``pymc_model_families``, so the 2026-09-28 impossible
sweep stopped at setup. These tests run each setup's real check, taken from
the sbatch file, with the defaults its submit script and setup job use,
against the real committed config and model directories.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.paths import REPO_ROOT

SLURM_DIR = REPO_ROOT / "scripts" / "subjective_randomness" / "slurm"


def _default(script: Path, name: str) -> str:
    """The ``${NAME:-default}`` a script falls back on for ``name``."""
    [value] = set(re.findall(r"\$\{" + name + r":-([^}]*)\}", script.read_text(encoding="utf-8")))
    return value


def _consistency_check(setup: Path) -> str:
    """The setup job's inline Python check: the heredoc that reads SEED_MODELS_REL."""
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY\n", setup.read_text(encoding="utf-8"), re.DOTALL)
    [check] = [block for block in blocks if "SEED_MODELS_REL" in block]
    return check


def _run_check(setup: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _consistency_check(setup)],
        env={"PATH": os.environ["PATH"], **env},
        capture_output=True, text=True, timeout=60,
    )


def _literature_env(seed_models_rel: str) -> dict[str, str]:
    setup = SLURM_DIR / "holdout_setup.sbatch"
    submit = SLURM_DIR / "submit_holdout_test_retest.sh"
    return {
        "SRC_CONFIG": str(REPO_ROOT / _default(submit, "CONFIG")),
        "GT_MODELS": _default(submit, "GT_MODELS"),
        "SEED_MODELS_REL": seed_models_rel,
        "GT_MODELS_SRC": str(REPO_ROOT / _default(setup, "SEED_MODELS_REL")),
    }


def _impossible_env(seed_models_rel: str) -> dict[str, str]:
    setup = SLURM_DIR / "impossible_holdout_setup.sbatch"
    submit = SLURM_DIR / "submit_impossible_holdout_test_retest.sh"
    impossible_rel = _default(setup, "IMPOSSIBLE_MODELS_REL")
    return {
        "SRC_CONFIG": str(REPO_ROOT / _default(submit, "CONFIG")),
        "GT_MODELS": _default(submit, "GT_MODELS"),
        "SEED_MODELS_REL": seed_models_rel,
        "IMPOSSIBLE_MODELS_REL": impossible_rel,
        "IMPOSSIBLE_MODELS_SRC": str(REPO_ROOT / impossible_rel),
    }


@pytest.mark.parametrize(
    "setup_name, env_for",
    [("holdout_setup.sbatch", _literature_env), ("impossible_holdout_setup.sbatch", _impossible_env)],
)
def test_the_setup_check_accepts_the_committed_config(setup_name, env_for):
    setup = SLURM_DIR / setup_name
    result = _run_check(setup, env_for(_default(setup, "SEED_MODELS_REL")))
    assert result.returncode == 0, result.stderr
    assert "[setup] validated GT_MODELS=" in result.stdout


@pytest.mark.parametrize(
    "setup_name, env_for",
    [("holdout_setup.sbatch", _literature_env), ("impossible_holdout_setup.sbatch", _impossible_env)],
)
def test_the_setup_check_rejects_a_seed_path_other_than_the_configs(setup_name, env_for):
    setup = SLURM_DIR / setup_name
    result = _run_check(setup, env_for("src/pipelines/outer_loop/projects/subjective_randomness/elsewhere"))
    assert result.returncode != 0
    assert "SEED_MODELS_REL env" in result.stderr
