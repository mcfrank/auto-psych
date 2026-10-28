"""The sweep setup job's consistency check accepts the configs it ships with.

The setup job (``holdout_setup.sbatch``) ends with an inline Python check that
the submit script's ``GT_MODELS``, the setup's ``SEED_MODELS_REL`` and
``GT_MODELS_REL`` and the committed config agree. The impossible sweep's own
setup (merged into this one on 2026-10-18) once defaulted to the project's
``seed_models/`` after its config moved to ``pymc_model_families``, so the
2026-09-28 impossible sweep stopped at setup. These tests run the setup's real
check, taken from the sbatch file, with the defaults each submit script uses,
against the real committed configs and model directories.
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
    [value] = set(
        re.findall(r"\$\{" + name + r":-([^}]*)\}", script.read_text(encoding="utf-8"))
    )
    return value


def _consistency_check(setup: Path) -> str:
    """The setup job's inline Python check: the heredoc that reads SEED_MODELS_REL."""
    blocks = re.findall(
        r"<<'PY'\n(.*?)\nPY\n", setup.read_text(encoding="utf-8"), re.DOTALL
    )
    [check] = [block for block in blocks if "SEED_MODELS_REL" in block]
    return check


def _run_check(setup: Path, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", _consistency_check(setup)],
        env={"PATH": os.environ["PATH"], **env},
        capture_output=True,
        text=True,
        timeout=60,
    )


SETUP = SLURM_DIR / "holdout_setup.sbatch"
SEED_REL = _default(SETUP, "SEED_MODELS_REL")


def _env(submit_name: str, seed_models_rel: str) -> dict[str, str]:
    """The check's environment under ``submit_name``'s defaults."""
    submit = SLURM_DIR / submit_name
    text = submit.read_text(encoding="utf-8")
    gt_rel = _default(submit, "GT_MODELS_REL") if "GT_MODELS_REL:-" in text else seed_models_rel
    return {
        "SRC_CONFIG": str(REPO_ROOT / _default(submit, "CONFIG")),
        "GT_MODELS": _default(submit, "GT_MODELS"),
        "SEED_MODELS_REL": seed_models_rel,
        "GT_MODELS_REL": gt_rel,
        "GT_MODELS_SRC": str(REPO_ROOT / gt_rel),
    }


SUBMITS = ["submit_holdout_test_retest.sh", "submit_impossible_holdout_test_retest.sh"]


@pytest.mark.parametrize("submit_name", SUBMITS)
def test_the_setup_check_accepts_the_committed_config(submit_name):
    result = _run_check(SETUP, _env(submit_name, SEED_REL))
    assert result.returncode == 0, result.stderr
    assert "[setup] validated GT_MODELS=" in result.stdout


@pytest.mark.parametrize("submit_name", SUBMITS)
def test_the_setup_check_rejects_a_seed_path_other_than_the_configs(submit_name):
    result = _run_check(
        SETUP,
        _env(submit_name, "src/pipelines/outer_loop/projects/subjective_randomness/elsewhere"),
    )
    assert result.returncode != 0
    assert "SEED_MODELS_REL env" in result.stderr


def test_the_setup_check_rejects_an_impossible_snapshot_from_the_seed_registry():
    """Without GT_MODELS_REL the snapshot would be the seed registry, which
    holds none of the impossible ground truths."""
    env = {**_env("submit_impossible_holdout_test_retest.sh", SEED_REL), "GT_MODELS_REL": SEED_REL}
    result = _run_check(SETUP, env)
    assert result.returncode != 0
    assert "GT_MODELS_REL" in result.stderr
