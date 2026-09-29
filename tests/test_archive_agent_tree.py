"""A failed archive must not delete the only copy (first audit R6).

Both holdout arrays archived a finished cell's agent ``_runs`` with
``tar czf … && echo …`` and then ran ``rm -rf`` on the agent tree whatever
tar did, so a tar that failed (quota, inodes) deleted the only copy. Both now
call ``archive_agent_tree.sh``, which removes the tree only once the archive
is written and lists back. The script runs here for real, with a stand-in
``tar`` on ``PATH`` where a failure is needed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

SLURM_DIR = (
    Path(__file__).resolve().parents[1] / "scripts" / "subjective_randomness" / "slurm"
)
SCRIPT = SLURM_DIR / "archive_agent_tree.sh"


def _cell(tmp_path: Path):
    run_dir = tmp_path / "run0" / "gt"
    run_dir.mkdir(parents=True)
    agent_dir = tmp_path / "agent_trees" / "a1b2c3"
    runs = agent_dir / "repo" / "_runs" / "experiment1"
    runs.mkdir(parents=True)
    (runs / "notes.md").write_text("what the agent did\n", encoding="utf-8")
    (run_dir / "repo").symlink_to(agent_dir / "repo")
    return run_dir, agent_dir


def _run(run_dir: Path, agent_dir: Path, path_prefix: Path | None = None):
    env = dict(os.environ)
    if path_prefix is not None:
        env["PATH"] = f"{path_prefix}:{env['PATH']}"
    return subprocess.run(
        ["bash", str(SCRIPT), "7", str(run_dir), str(agent_dir)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _fake_tar(tmp_path: Path, body: str) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "tar"
    fake.write_text(
        f"#!/bin/bash\nREAL_TAR={shutil.which('tar')}\n" + body, encoding="utf-8"
    )
    fake.chmod(0o755)
    return bin_dir


def test_a_verified_archive_replaces_the_tree(tmp_path):
    run_dir, agent_dir = _cell(tmp_path)
    result = _run(run_dir, agent_dir)
    assert result.returncode == 0, result.stderr
    with tarfile.open(run_dir / "agent_runs.tar.gz") as tar:
        assert "_runs/experiment1/notes.md" in tar.getnames()
    assert not agent_dir.exists()
    assert not (run_dir / "repo").is_symlink()
    assert not (run_dir / "agent_runs.tar.gz.partial").exists()


@pytest.mark.parametrize(
    "tar_body",
    [
        # tar fails outright (a full quota): exit 2, a partial file left behind.
        'for a; do case "$prev" in czf) printf truncated > "$a";; esac; prev="$a"; done\n'
        'echo "tar: write error: Disk quota exceeded" >&2; exit 2\n',
        # tar "succeeds" but the file does not read back as an archive.
        'if [[ "$1" == czf ]]; then printf "not a gzip" > "$2"; exit 0; fi\n'
        'exec "$REAL_TAR" "$@"\n',
    ],
    ids=["tar_fails", "archive_unreadable"],
)
def test_a_failed_archive_keeps_the_tree_and_says_so(tmp_path, tar_body):
    run_dir, agent_dir = _cell(tmp_path)
    result = _run(run_dir, agent_dir, _fake_tar(tmp_path, tar_body))
    assert result.returncode == 0
    assert (agent_dir / "repo" / "_runs" / "experiment1" / "notes.md").exists()
    assert (run_dir / "repo").is_symlink()
    assert not (run_dir / "agent_runs.tar.gz").exists()
    assert not (run_dir / "agent_runs.tar.gz.partial").exists()
    assert "ERROR" in result.stderr and "KEEPING the agent tree" in result.stderr


def test_an_existing_archive_is_never_overwritten(tmp_path):
    run_dir, agent_dir = _cell(tmp_path)
    (run_dir / "agent_runs.tar.gz").write_bytes(b"an earlier run's archive")
    result = _run(run_dir, agent_dir)
    assert result.returncode == 0
    assert (run_dir / "agent_runs.tar.gz").read_bytes() == b"an earlier run's archive"
    assert agent_dir.exists()


@pytest.mark.parametrize(
    "array",
    ["holdout_recovery_array.sbatch", "impossible_holdout_recovery_array.sbatch"],
)
def test_both_arrays_archive_through_the_verified_helper(array):
    text = (SLURM_DIR / array).read_text(encoding="utf-8")
    assert (
        'bash "$SLURM_DIR/archive_agent_tree.sh" "$TASK" "$RUN_DIR" "$AGENT_DIR"'
        in text
    )
    assert "tar czf" not in text and 'rm -rf "$AGENT_DIR"' not in text
