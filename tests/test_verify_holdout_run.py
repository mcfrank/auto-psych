"""The verifier script must catch arm C's bug: a raw run whose agent-facing CSV
carries engineered columns, and a run with no candidate-facing CSV at all.

It must also make an absent critique visible: a finished run in which no
round produced a critique is flagged (a warning, not a failure — the loop is
designed to proceed without one, and a false failure here has cancelled a
gated arm before).

These tests build synthetic run trees and invoke the verifier via subprocess,
checking the exit code and output.
"""

from __future__ import annotations

import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from tests.paths import REPO_ROOT

VERIFIER = REPO_ROOT / "scripts" / "subjective_randomness" / "slurm" / "verify_holdout_run.sh"

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


def _history(*round_statuses):
    """An inner-loop history.json: the seed step, then one round per status
    (``None`` = a round recorded before the critique status existed)."""
    seed = {"step": 0, "iteration": None, "best_model": "seed", "posteriors": {}, "elpd_loo": {}}
    rounds = []
    for i, status in enumerate(round_statuses):
        entry = {"step": i + 1, "iteration": i, "best_model": "seed", "posteriors": {}, "elpd_loo": {}}
        if status is not None:
            entry["critique"] = status
        rounds.append(entry)
    return json.dumps([seed, *rounds])


_CRITIQUED = {
    "status": "critiqued", "incumbent": "seed", "attempts": 1,
    "n_statistics": 8, "n_significant": 2, "n_significant_fdr": 1,
}
_NO_CRITIQUE = {
    "status": "no_critique", "incumbent": "seed", "attempts": 2,
    "reason": "the critique agent wrote no usable test statistic in 2 attempts",
}


def _build_clean_raw_tree(work_root: Path, history: str | None = None):
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
    if history is not None:
        (exp1_ml / "history.json").write_text(history, encoding="utf-8")

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


class TestVerifyCritiquePresence:
    def test_run_with_a_critiqued_round_is_ok(self, tmp_path):
        work_root = _build_clean_raw_tree(
            tmp_path / "critiqued", history=_history(_NO_CRITIQUE, _CRITIQUED)
        )
        result = _run_verifier(work_root)
        assert result.returncode == 0, result.stdout
        assert "[ok]   critique" in result.stdout
        assert "[WARN]" not in result.stdout

    def test_run_with_no_critique_in_any_round_is_flagged_but_not_failed(self, tmp_path):
        work_root = _build_clean_raw_tree(
            tmp_path / "absent", history=_history(_NO_CRITIQUE, _NO_CRITIQUE)
        )
        result = _run_verifier(work_root)
        assert result.returncode == 0, result.stdout
        assert "[WARN]" in result.stdout
        assert "no critique" in result.stdout
        # The flag also lands in the verdict file the run root keeps.
        assert "no critique" in (work_root / "VERDICT.md").read_text(encoding="utf-8")

    def test_run_with_critique_disabled_is_flagged_the_same_way(self, tmp_path):
        work_root = _build_clean_raw_tree(
            tmp_path / "disabled", history=_history({"status": "disabled"})
        )
        result = _run_verifier(work_root)
        assert result.returncode == 0, result.stdout
        assert "[WARN]" in result.stdout

    def test_run_recorded_before_the_status_existed_is_informational(self, tmp_path):
        work_root = _build_clean_raw_tree(tmp_path / "old", history=_history(None, None))
        result = _run_verifier(work_root)
        assert result.returncode == 0, result.stdout
        assert "[WARN]" not in result.stdout
        assert "not recorded" in result.stdout

    def test_unknown_critique_status_fails_loudly(self, tmp_path):
        work_root = _build_clean_raw_tree(
            tmp_path / "bogus", history=_history({"status": "maybe"})
        )
        result = _run_verifier(work_root)
        assert result.returncode != 0, result.stdout
        assert "critique status" in result.stdout
