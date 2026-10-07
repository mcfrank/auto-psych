"""Shared access to the external reference-game CSVs for the ingest tests.

A CC-BY source's CSV is committed and always available. A source without a
licence lives only in the gitignored cache (``data/rsa/external/``): its tests
skip unless the cached CSV exists (``uv run python -m src.rsa.ingest.run
--sources <name>`` builds it) or ``RSA_INGEST_FETCH=1`` lets the test fetch the
pinned files and build it in a temporary directory. Rebuilding a committed CSV
(to compare it byte for byte) needs the raw files the same way.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from src.rsa.ingest.common import Source, frame_of
from src.rsa.ingest.fetch import EXTERNAL_DIR, cached_path, fetch

FETCH_ENV = "RSA_INGEST_FETCH"


def may_fetch() -> bool:
    return os.environ.get(FETCH_ENV) == "1"


def raw_paths(source: Source) -> dict:
    """The verified raw files of ``source``, or skip when they are not cached and fetching is off."""
    missing = [f.name for f in source.files if not cached_path(EXTERNAL_DIR, source.name, f).exists()]
    if missing and not may_fetch():
        pytest.skip(
            f"{source.name}: raw files {missing} are not in {EXTERNAL_DIR}; run "
            f"`uv run python -m src.rsa.ingest.run --sources {source.name}` or set {FETCH_ENV}=1"
        )
    return {f.name: fetch(f, EXTERNAL_DIR, source.name) for f in source.files}


def build_csv(source: Source, out_dir: Path) -> Path:
    """Derive ``source`` from its raw files into ``out_dir`` (never into the repo)."""
    path = Path(out_dir) / source.csv_name
    frame_of(source.build(raw_paths(source))).to_csv(path, index=False, lineterminator="\n")
    return path


def derived_csv(source: Source, tmp_dir: Path) -> Path:
    """The source's derived CSV: committed, cached, or (with fetching on) built now."""
    path = source.csv_path()
    if path.exists():
        return path
    if source.commit_csv:
        raise FileNotFoundError(f"{path} is committed but missing")
    if not may_fetch():
        pytest.skip(
            f"{source.name} has no licence, so its CSV is not committed and {path} is not built; run "
            f"`uv run python -m src.rsa.ingest.run --sources {source.name}` or set {FETCH_ENV}=1"
        )
    return build_csv(source, tmp_dir)


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
