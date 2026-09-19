"""Tests for the shared cell-archive extraction helper.

The helper resolves a cell's run-tree root, extracting ``agent_runs.tar.gz``
into a cached temp directory when the on-disk tree is absent, and reusing that
extraction across callers.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import pytest

from src.subjective_randomness.cell_archive import (
    CellArchiveManager,
    resolve_run_root,
)


def _build_run_tree(root: Path, gt_name: str, n_experiments: int = 1) -> Path:
    """Build a minimal run tree and return the run root."""
    run_root = root / "_runs" / gt_name
    for exp in range(1, n_experiments + 1):
        data_dir = run_root / f"experiment{exp}" / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "responses.csv").write_text(
            "sequence_a,sequence_b,participant_id,trial_index,chose_left\n"
            "HHH,TTT,0,0,1\n",
            encoding="utf-8",
        )
        loop_dir = run_root / f"experiment{exp}" / "model_loop"
        loop_dir.mkdir(parents=True)
        (loop_dir / "responses.csv").write_text(
            "sequence_a,sequence_b,participant_id,trial_index,chose_left\n"
            "HHH,TTT,0,0,1\n",
            encoding="utf-8",
        )
        (loop_dir / "history.json").write_text("[]", encoding="utf-8")
        models_dir = loop_dir / "models"
        models_dir.mkdir()
    (run_root / "eval_stimuli.json").write_text("[]", encoding="utf-8")
    return run_root


def _archive_run_tree(cell_dir: Path, staging: Path) -> None:
    """Archive the staging/_runs/ tree into cell_dir/agent_runs.tar.gz."""
    tar_path = cell_dir / "agent_runs.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        tf.add(str(staging / "_runs"), arcname="_runs")


def test_resolve_run_root_existing_on_disk(tmp_path):
    """When the run_root path exists, return it directly."""
    staging = tmp_path / "staging"
    staging.mkdir()
    run_root = _build_run_tree(staging, "gt_a")

    mgr = CellArchiveManager()
    resolved = resolve_run_root(tmp_path, str(run_root), "gt_a", mgr)
    assert resolved == run_root
    assert (resolved / "experiment1" / "data" / "responses.csv").exists()


def test_resolve_run_root_extracts_archive(tmp_path):
    """When run_root doesn't exist but agent_runs.tar.gz does, extract it."""
    staging = tmp_path / "staging"
    staging.mkdir()
    _build_run_tree(staging, "gt_a")

    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    _archive_run_tree(cell_dir, staging)

    fake_run_root = cell_dir / "repo" / "_runs" / "gt_a"
    assert not fake_run_root.exists()

    mgr = CellArchiveManager()
    resolved = resolve_run_root(cell_dir, str(fake_run_root), "gt_a", mgr)
    assert resolved.exists()
    assert (resolved / "experiment1" / "data" / "responses.csv").exists()
    assert (resolved / "eval_stimuli.json").exists()
    mgr.cleanup()


def test_resolve_run_root_reuses_extraction(tmp_path):
    """Repeated calls for the same cell reuse the same extraction."""
    staging = tmp_path / "staging"
    staging.mkdir()
    _build_run_tree(staging, "gt_a")

    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    _archive_run_tree(cell_dir, staging)

    fake_run_root = str(cell_dir / "repo" / "_runs" / "gt_a")

    mgr = CellArchiveManager()
    resolved1 = resolve_run_root(cell_dir, fake_run_root, "gt_a", mgr)
    resolved2 = resolve_run_root(cell_dir, fake_run_root, "gt_a", mgr)
    assert resolved1 == resolved2
    mgr.cleanup()


def test_resolve_run_root_no_archive_raises(tmp_path):
    """When neither run_root nor archive exists, raise FileNotFoundError."""
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()

    fake_run_root = str(cell_dir / "repo" / "_runs" / "gt_a")

    mgr = CellArchiveManager()
    with pytest.raises(FileNotFoundError, match="agent_runs.tar.gz"):
        resolve_run_root(cell_dir, fake_run_root, "gt_a", mgr)


def test_resolve_run_root_corrupt_archive(tmp_path):
    """A corrupt archive raises rather than silently skipping."""
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    (cell_dir / "agent_runs.tar.gz").write_bytes(b"not a tarball")

    fake_run_root = str(cell_dir / "repo" / "_runs" / "gt_a")

    mgr = CellArchiveManager()
    with pytest.raises(Exception):
        resolve_run_root(cell_dir, fake_run_root, "gt_a", mgr)


def test_resolve_run_root_missing_gt_in_archive(tmp_path):
    """Archive exists but doesn't contain the expected ground truth."""
    staging = tmp_path / "staging"
    staging.mkdir()
    _build_run_tree(staging, "gt_a")

    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    _archive_run_tree(cell_dir, staging)

    fake_run_root = str(cell_dir / "repo" / "_runs" / "gt_b")

    mgr = CellArchiveManager()
    with pytest.raises(FileNotFoundError, match="gt_b"):
        resolve_run_root(cell_dir, fake_run_root, "gt_b", mgr)
    mgr.cleanup()


def test_manager_cleanup_removes_temp_dirs(tmp_path):
    """CellArchiveManager.cleanup removes all extracted temp dirs."""
    staging = tmp_path / "staging"
    staging.mkdir()
    _build_run_tree(staging, "gt_a")

    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    _archive_run_tree(cell_dir, staging)

    fake_run_root = str(cell_dir / "repo" / "_runs" / "gt_a")

    mgr = CellArchiveManager()
    resolved = resolve_run_root(cell_dir, fake_run_root, "gt_a", mgr)
    assert resolved.exists()
    mgr.cleanup()
    assert not resolved.exists()


def test_manager_context_manager(tmp_path):
    """CellArchiveManager works as a context manager."""
    staging = tmp_path / "staging"
    staging.mkdir()
    _build_run_tree(staging, "gt_a")

    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    _archive_run_tree(cell_dir, staging)

    fake_run_root = str(cell_dir / "repo" / "_runs" / "gt_a")

    with CellArchiveManager() as mgr:
        resolved = resolve_run_root(cell_dir, fake_run_root, "gt_a", mgr)
        assert resolved.exists()
    assert not resolved.exists()
