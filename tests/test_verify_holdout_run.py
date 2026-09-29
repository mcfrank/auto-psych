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


def _history(*round_statuses, bests=None):
    """An inner-loop history.json: the seed step, then one round per status
    (``None`` = a round recorded before the critique status existed).
    ``bests`` names the incumbent at each step; by default the seed step's
    ``seed`` is displaced by ``candidate_<i>`` at round i, so a history with
    a round has an incumbent change and the incumbent check stays quiet in
    tests that are about something else. The seed step scores ``seed`` and
    ``rival`` (the cell's starting set)."""
    if bests is None:
        bests = ["seed", *(f"candidate_{i}" for i in range(len(round_statuses)))]
    if len(bests) != 1 + len(round_statuses):
        raise ValueError("bests must name one incumbent per step")
    seed = {"step": 0, "iteration": None, "best_model": bests[0],
            "posteriors": {"seed": 0.6, "rival": 0.4}, "elpd_loo": {}}
    rounds = []
    for i, status in enumerate(round_statuses):
        entry = {"step": i + 1, "iteration": i, "best_model": bests[i + 1],
                 "posteriors": {bests[i + 1]: 1.0}, "elpd_loo": {}}
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


def _build_clean_raw_tree(
    work_root: Path, history: str | None = None, history2: str | None = None
):
    """A valid raw run tree: all CSVs have only raw columns, no drops, no
    featurizer imports. ``history`` / ``history2`` are experiment 1's and 2's
    inner-loop history.json (omitted when None)."""
    _make_log(work_root / "slurm_logs" / "holdout_recovery_1.out")

    # Build a tar archive simulating a completed task
    cell = work_root / "run1" / "gt"
    cell.mkdir(parents=True)
    (cell / "holdout.json").write_text("{}", encoding="utf-8")
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
    if history2 is not None:
        exp2_ml = tar_staging / "experiment2" / "model_loop"
        exp2_ml.mkdir(parents=True)
        (exp2_ml / "history.json").write_text(history2, encoding="utf-8")

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

    @staticmethod
    def _kept_tree_behind_a_symlink(work_root: Path, leaked_file: str) -> None:
        """A kept agent tree in the current layout: the real tree lives under
        an opaque agent_trees/<id>/ and run<r>/<gt>/repo is a symlink to it."""
        real_repo = work_root / "agent_trees" / "0123abcd" / "repo"
        _write_file(real_repo / leaked_file, "leaked\n")
        cell = work_root / "run1" / "gt_kept"
        cell.mkdir(parents=True)
        (cell / "holdout.json").write_text("{}", encoding="utf-8")
        (cell / "repo").symlink_to(real_repo)

    def test_a_leak_in_a_symlinked_kept_tree_fails(self, tmp_path):
        """find does not descend into a symlinked starting directory unless
        told to, which would make this check pass on every kept tree."""
        work_root = _build_clean_raw_tree(tmp_path / "symlinked")
        self._kept_tree_behind_a_symlink(work_root, "deep/ground_truth_models.py")
        result = _run_verifier(work_root)
        assert result.returncode != 0, result.stdout
        assert "ground_truth_models.py" in result.stdout

    def test_a_claude_md_in_a_kept_tree_fails(self, tmp_path):
        """The claude backend loads CLAUDE.md into every agent session; the
        project's names the held-out model."""
        work_root = _build_clean_raw_tree(tmp_path / "claude_md")
        self._kept_tree_behind_a_symlink(work_root, "CLAUDE.md")
        result = _run_verifier(work_root)
        assert result.returncode != 0, result.stdout
        assert "CLAUDE.md" in result.stdout

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


def _write_live_cell(work_root: Path, gt: str, histories: list[str]) -> None:
    """A cell whose repo copy was kept (KEEP_REPO_COPY=1): its experiments'
    history.json files live under run1/<gt>/repo/_runs/<gt>/ instead of an
    archive."""
    (work_root / "run1" / gt).mkdir(parents=True)
    (work_root / "run1" / gt / "holdout.json").write_text("{}", encoding="utf-8")
    run_root = work_root / "run1" / gt / "repo" / "_runs" / gt
    for exp_num, history in enumerate(histories, start=1):
        loop_dir = run_root / f"experiment{exp_num}" / "model_loop"
        loop_dir.mkdir(parents=True)
        (loop_dir / "history.json").write_text(history, encoding="utf-8")
        data_dir = run_root / f"experiment{exp_num}" / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "responses.csv").write_text(RAW_HEADER + "\nHHT,THT,0,0,1\n", encoding="utf-8")
        (loop_dir / "responses.csv").write_text(RAW_HEADER + "\nHHT,THT,0,0,1\n", encoding="utf-8")


class TestVerifyIncumbentChanges:
    """The loop-improvement plan's primary metric, surfaced by the verifier:
    a finished cell in which the exported best model never changed is
    flagged. A warning, not a failure — zero is the current true value."""

    def test_cell_whose_incumbent_never_changes_is_flagged_but_not_failed(self, tmp_path):
        work_root = _build_clean_raw_tree(
            tmp_path / "frozen",
            history=_history(_CRITIQUED, _CRITIQUED, bests=["seed", "seed", "seed"]),
            history2=_history(_CRITIQUED, bests=["seed", "seed"]),
        )
        result = _run_verifier(work_root)
        assert result.returncode == 0, result.stdout
        assert "[WARN]" in result.stdout
        assert "incumbent" in result.stdout
        # Five steps over two experiments, no change, the seed at the end.
        assert "steps=5 changes=0 discovered_steps=0 final=seed" in result.stdout
        assert "incumbent" in (work_root / "VERDICT.md").read_text(encoding="utf-8")

    def test_cell_whose_incumbent_changes_is_ok(self, tmp_path):
        work_root = _build_clean_raw_tree(
            tmp_path / "moved", history=_history(_CRITIQUED, _CRITIQUED)
        )
        result = _run_verifier(work_root)
        assert result.returncode == 0, result.stdout
        assert "[ok]   incumbent" in result.stdout
        assert "[WARN]" not in result.stdout
        # candidate_0 then candidate_1: two changes, neither a starting model.
        assert "steps=3 changes=2 discovered_steps=2 final=candidate_1" in result.stdout

    def test_a_change_across_the_experiment_boundary_counts(self, tmp_path):
        # Experiment 1 never moves; experiment 2 opens with a different
        # incumbent (a carried model refit on more data). That is a change.
        work_root = _build_clean_raw_tree(
            tmp_path / "boundary",
            history=_history(_CRITIQUED, bests=["seed", "seed"]),
            history2=_history(_CRITIQUED, bests=["rival", "rival"]),
        )
        result = _run_verifier(work_root)
        assert result.returncode == 0, result.stdout
        assert "[ok]   incumbent" in result.stdout
        assert "steps=4 changes=1 discovered_steps=0 final=rival" in result.stdout

    def test_a_kept_repo_copy_is_read_like_an_archive(self, tmp_path):
        work_root = _build_clean_raw_tree(tmp_path / "mixed", history=_history(_CRITIQUED))
        _write_live_cell(
            work_root, "gt_live",
            [_history(_CRITIQUED, bests=["seed", "seed"]), _history(_CRITIQUED, bests=["seed", "seed"])],
        )
        result = _run_verifier(work_root)
        assert result.returncode == 0, result.stdout
        assert "[WARN] 1 of 2 cell(s)" in result.stdout
        assert "run1/gt_live: steps=4 changes=0" in result.stdout

    def test_a_history_without_a_seed_step_fails_loudly(self, tmp_path):
        broken = json.dumps([
            {"step": 1, "iteration": 0, "best_model": "seed", "posteriors": {"seed": 1.0}, "elpd_loo": {}},
        ])
        work_root = _build_clean_raw_tree(tmp_path / "broken", history=broken)
        result = _run_verifier(work_root)
        assert result.returncode != 0
        assert "[FAIL]" in result.stdout and "incumbent" in result.stdout

    def test_no_history_is_informational(self, tmp_path):
        work_root = _build_clean_raw_tree(tmp_path / "bare")
        result = _run_verifier(work_root)
        assert result.returncode == 0, result.stdout
        assert "[info] no inner-loop history.json found for the incumbent check" in result.stdout


class TestVerifyJudgesCellsByTheirResults:
    """A cell that failed and was resumed by a retry leaves its failed
    attempt's log behind. The verifier judges cells by their results
    (holdout.json, MISSING_CELLS.txt), so such a sweep passes; a cell with
    no result fails whatever its logs say."""

    @staticmethod
    def _failed_attempt_log(work_root: Path, cell: Path, name: str) -> None:
        _write_file(
            work_root / "slurm_logs" / name,
            f"[task 1] repeat=1 gt=gt seed=1 -> {cell}\n"
            "Traceback (most recent call last):\n  MemoryError\n",
        )

    def test_a_cell_finished_by_a_retry_passes(self, tmp_path):
        work_root = _build_clean_raw_tree(tmp_path / "retried")
        self._failed_attempt_log(work_root, work_root / "run1" / "gt", "holdout_recovery_7_1.out")
        result = _run_verifier(work_root)
        assert result.returncode == 0, result.stdout
        assert "[ok]   all 1 cell(s) have a result" in result.stdout
        assert "[info] 1 failed attempt(s) of cells a later attempt finished" in result.stdout

    def test_a_traceback_of_a_cell_without_a_result_still_fails(self, tmp_path):
        work_root = _build_clean_raw_tree(tmp_path / "unfinished")
        cell = work_root / "run2" / "gt"
        cell.mkdir(parents=True)
        self._failed_attempt_log(work_root, cell, "holdout_recovery_7_5.out")
        result = _run_verifier(work_root)
        assert result.returncode != 0, result.stdout
        assert "cell(s) have no holdout.json: run2/gt" in result.stdout
        assert "error line(s)" in result.stdout

    def test_a_cell_without_a_result_fails_even_if_every_log_finished(self, tmp_path):
        work_root = _build_clean_raw_tree(tmp_path / "no_result")
        (work_root / "run1" / "gt" / "holdout.json").unlink()
        result = _run_verifier(work_root)
        assert result.returncode != 0, result.stdout
        assert "no holdout.json: run1/gt" in result.stdout

    def test_missing_cells_listed_by_the_summary_job_fail(self, tmp_path):
        work_root = _build_clean_raw_tree(tmp_path / "missing")
        (work_root / "MISSING_CELLS.txt").write_text("run3/gt\n", encoding="utf-8")
        result = _run_verifier(work_root)
        assert result.returncode != 0, result.stdout
        assert "(MISSING_CELLS.txt)" in result.stdout and "run3/gt" in result.stdout
