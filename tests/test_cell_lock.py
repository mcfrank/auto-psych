"""One job per cell: a resubmitted task exits if its cell is running elsewhere.

The retry job and the sweep watcher can both resubmit a failed cell. Two jobs
in one cell would rebuild its agent tree under each other and corrupt its
experiments. ``cell_lock.sh`` takes a per-cell lock, refuses it while the
holder is still in the queue, and takes over a stale one.
"""

from __future__ import annotations

import os
import subprocess

from pyprojroot import here

LOCK = here() / "scripts" / "subjective_randomness" / "slurm" / "cell_lock.sh"


def _run(tmp_path, lock, job, live_jobs):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    squeue = bin_dir / "squeue"
    # A fake squeue: the job id for "live" jobs, and — like the real one — an
    # error on stdout for a job that has left the queue.
    squeue.write_text(
        '#!/bin/bash\njob="$3"\n'
        + "".join(f'[[ "$job" == "{j}" ]] && echo "{j}" && exit 0\n' for j in live_jobs)
        + 'echo "slurm_load_jobs error: Invalid job id specified"\nexit 1\n'
    )
    squeue.chmod(0o755)
    env = {**os.environ, "SQUEUE": str(squeue)}
    return subprocess.run(
        ["bash", str(LOCK), str(lock), job], env=env, capture_output=True, text=True
    )


def test_a_free_cell_is_taken(tmp_path):
    lock = tmp_path / ".cell_lock"
    assert _run(tmp_path, lock, "101", live_jobs=[]).returncode == 0
    assert lock.read_text().strip() == "101"


def test_a_cell_held_by_a_live_job_is_refused(tmp_path):
    lock = tmp_path / ".cell_lock"
    lock.write_text("101\n")
    result = _run(tmp_path, lock, "202", live_jobs=["101"])
    assert result.returncode == 3
    assert lock.read_text().strip() == "101"


def test_a_stale_lock_is_taken_over(tmp_path):
    lock = tmp_path / ".cell_lock"
    lock.write_text("101\n")
    assert _run(tmp_path, lock, "202", live_jobs=[]).returncode == 0
    assert lock.read_text().strip() == "202"
