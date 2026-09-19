"""Shared archive extraction for holdout-recovery cells.

A cell's run tree may be on disk (during a live run or before archiving) or
packed into ``agent_runs.tar.gz`` (the standard post-run state).
``resolve_run_root`` transparently handles both: it returns the on-disk path
when it exists, or extracts the archive into a temp directory and returns the
path inside it. ``CellArchiveManager`` owns the temp directories and cleans
them up when the caller is done.
"""

from __future__ import annotations

import tarfile
import tempfile
from pathlib import Path
from typing import Dict, Optional


class CellArchiveManager:
    """Manages temp directories for extracted cell archives.

    Reuses extractions across calls for the same cell directory.
    Use as a context manager or call ``.cleanup()`` explicitly.
    """

    def __init__(self) -> None:
        self._extractions: Dict[Path, tempfile.TemporaryDirectory] = {}

    def get_or_extract(self, cell_dir: Path) -> Path:
        """Return the extraction root for ``cell_dir``, extracting if needed.

        Raises ``FileNotFoundError`` if no archive exists,
        and propagates tarfile errors for corrupt archives.
        """
        cell_dir = Path(cell_dir).resolve()
        if cell_dir in self._extractions:
            return Path(self._extractions[cell_dir].name)

        tar_path = cell_dir / "agent_runs.tar.gz"
        if not tar_path.exists():
            raise FileNotFoundError(
                f"No agent_runs.tar.gz at {cell_dir} and the on-disk "
                f"run tree does not exist"
            )

        td = tempfile.TemporaryDirectory(dir=cell_dir, prefix="archive_extract_")
        with tarfile.open(tar_path) as tf:
            tf.extractall(td.name, filter="data")
        self._extractions[cell_dir] = td
        return Path(td.name)

    def cleanup(self) -> None:
        """Remove all extracted temp directories."""
        for td in self._extractions.values():
            td.cleanup()
        self._extractions.clear()

    def __enter__(self) -> "CellArchiveManager":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.cleanup()


def resolve_run_root(
    cell_dir: Path,
    recorded_run_root: str,
    gt_name: str,
    manager: CellArchiveManager,
) -> Path:
    """Return a usable run-root path, extracting the archive if needed.

    If ``recorded_run_root`` (the path from ``holdout.json``) exists on disk,
    return it directly. Otherwise extract ``cell_dir/agent_runs.tar.gz`` via
    ``manager`` and locate ``_runs/<gt_name>`` inside the extraction.

    Raises ``FileNotFoundError`` if neither the on-disk path nor the archive
    exists, or if the archive does not contain the expected ground truth.
    """
    on_disk = Path(recorded_run_root)
    if on_disk.exists():
        return on_disk

    extract_root = manager.get_or_extract(cell_dir)
    resolved = extract_root / "_runs" / gt_name
    if not resolved.exists():
        raise FileNotFoundError(
            f"Extracted archive at {cell_dir} but {gt_name} not found "
            f"inside _runs/ (available: "
            f"{sorted(p.name for p in (extract_root / '_runs').iterdir()) if (extract_root / '_runs').exists() else []})"
        )
    return resolved
