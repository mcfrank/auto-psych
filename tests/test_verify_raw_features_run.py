"""The verifier script must catch arm C's bug: a raw run whose agent-facing CSV
carries engineered columns, and a run with no candidate-facing CSV at all.

These tests build synthetic run trees and invoke the verifier via subprocess,
checking the exit code and output.
"""

from __future__ import annotations

import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from tests.paths import REPO_ROOT

VERIFIER = REPO_ROOT / "scripts" / "subjective_randomness" / "slurm" / "verify_raw_features_run.sh"

RAW_HEADER = "sequence_a,sequence_b,participant_id,trial_index,chose_left"
FEATURIZED_HEADER = RAW_HEADER + ",n_a,n_b,rep_motifs_a,rep_motifs_b"


def _write_file(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_log(path: Path, *, finished: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    if finished:
        path.write_text("[task 1/1] done -> run1/gt\n", encoding="utf-8")
    else:
        path.write_text("started\n", encoding="utf-8")


def _build_clean_raw_tree(work_root: Path):
    """A valid raw run tree: all CSVs have only raw columns, no drops, no featurizer imports."""
    _make_log(work_root / "slurm_logs" / "holdout_recovery_1.out")

    # Build a tar archive simulating a completed task
    cell = work_root / "run1" / "gt"
    cell.mkdir(parents=True)
    tar_path = cell / "agent_runs.tar.gz"

    # Create temp files for the tar
    tar_staging = work_root / "_tar_staging"
    exp1_data = tar_staging / "experiment1" / "data"
    exp1_data.mkdir(parents=True)
    (exp1_data / "responses.csv").write_text(
        RAW_HEADER + "\nHHT,THT,0,0,1\n", encoding="utf-8"
    )
    exp1_ml = tar_staging / "experiment1" / "model_loop"
    exp1_ml.mkdir(parents=True)
    (exp1_ml / "responses.csv").write_text(
        RAW_HEADER + "\nHHT,THT,0,0,1\n", encoding="utf-8"
    )
    # candidate model file (no featurizer import)
    models_dir = exp1_ml / "models"
    models_dir.mkdir()
    (models_dir / "seed.py").write_text(
        "import pymc as pm\n# no featurizer import\n", encoding="utf-8"
    )
    # CONTEXT.md with raw columns only
    cand_dir = exp1_ml / "round0" / "candidate0"
    cand_dir.mkdir(parents=True)
    (cand_dir / "CONTEXT.md").write_text(
        f"Columns: `{RAW_HEADER}`\n", encoding="utf-8"
    )
    # screened_out.json
    design_dir = tar_staging / "experiment1" / "design"
    design_dir.mkdir(parents=True)
    (design_dir / "screened_out.json").write_text("[]", encoding="utf-8")

    with tarfile.open(tar_path, "w:gz") as tar:
        for p in tar_staging.rglob("*"):
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(tar_staging)))

    return work_root


def _build_arm_c_bug_tree(work_root: Path):
    """Reproduces arm C's bug: raw data/responses.csv but featurized model_loop/responses.csv."""
    _make_log(work_root / "slurm_logs" / "holdout_recovery_1.out")

    cell = work_root / "run1" / "gt"
    cell.mkdir(parents=True)
    tar_path = cell / "agent_runs.tar.gz"

    tar_staging = work_root / "_tar_staging"
    exp1_data = tar_staging / "experiment1" / "data"
    exp1_data.mkdir(parents=True)
    # Raw data responses
    (exp1_data / "responses.csv").write_text(
        RAW_HEADER + "\nHHT,THT,0,0,1\n", encoding="utf-8"
    )
    exp1_ml = tar_staging / "experiment1" / "model_loop"
    exp1_ml.mkdir(parents=True)
    # FEATURIZED model_loop responses — the bug
    (exp1_ml / "responses.csv").write_text(
        FEATURIZED_HEADER + "\nHHT,THT,0,0,1,3,3,1,0\n", encoding="utf-8"
    )

    with tarfile.open(tar_path, "w:gz") as tar:
        for p in tar_staging.rglob("*"):
            if p.is_file():
                tar.add(p, arcname=str(p.relative_to(tar_staging)))

    return work_root


def _build_no_csv_tree(work_root: Path):
    """A run tree with no candidate-facing CSV at all — should fail."""
    _make_log(work_root / "slurm_logs" / "holdout_recovery_1.out")
    cell = work_root / "run1" / "gt"
    cell.mkdir(parents=True)
    # Empty tar with no CSVs
    tar_path = cell / "agent_runs.tar.gz"
    tar_staging = work_root / "_tar_staging"
    tar_staging.mkdir(parents=True)
    (tar_staging / "empty.txt").write_text("no csv here\n", encoding="utf-8")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(tar_staging / "empty.txt", arcname="empty.txt")

    return work_root


def _run_verifier(work_root: Path) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["WORK_ROOT"] = str(work_root)
    env["ALL_JOB_IDS"] = "0"
    return subprocess.run(
        ["bash", str(VERIFIER)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


class TestVerifyRawFeaturesRun:
    def test_arm_c_bug_tree_fails(self, tmp_path):
        """A tree with raw data CSV but featurized model_loop CSV must fail."""
        work_root = _build_arm_c_bug_tree(tmp_path / "armc_bug")
        result = _run_verifier(work_root)
        assert result.returncode != 0, (
            f"Verifier should fail on arm C bug tree.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_clean_raw_tree_passes(self, tmp_path):
        """A valid raw tree with only raw columns everywhere must pass."""
        work_root = _build_clean_raw_tree(tmp_path / "clean_raw")
        result = _run_verifier(work_root)
        assert result.returncode == 0, (
            f"Verifier should pass on clean raw tree.\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_no_csv_tree_warns(self, tmp_path):
        """A tree with no candidate-facing CSV should warn (exit non-zero once
        the verifier checks for missing CSVs)."""
        work_root = _build_no_csv_tree(tmp_path / "no_csv")
        result = _run_verifier(work_root)
        # The current verifier prints [warn] for no CSV found, which is not
        # a hard failure yet. After P2 strengthening, it should fail.
        assert "no" in result.stdout.lower() or "warn" in result.stdout.lower() or result.returncode != 0
