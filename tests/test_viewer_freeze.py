"""Integration tests for freezing the run explorer into a static snapshot.

The freeze step must reproduce, as plain files on disk, exactly what the live
Flask API serves — so the same vanilla-JS frontend renders from a static host
(Firebase Hosting) with no Python server. These outside-in tests freeze a
curated subset of the demo tree and assert the static layout matches the
scanners byte-for-byte.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.viewer.freeze import freeze_snapshot
from src.viewer.scan import scan_run, scan_run_experiment
from tests.viewer_fixtures import build_demo_tree


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    return build_demo_tree(tmp_path / "data")


@pytest.fixture
def snapshot(tmp_path: Path, data_root: Path) -> Path:
    """Freeze a curated subset (one multi-experiment run) into a dist directory."""
    out = tmp_path / "dist"
    freeze_snapshot(data_root=data_root, out_dir=out, run_paths=["outer_loop/demo"])
    return out


def _run_dir(snapshot: Path) -> Path:
    return snapshot / "data" / "run" / "outer_loop" / "demo"


def test_index_is_filtered_to_curated_runs(snapshot: Path):
    index = json.loads((snapshot / "data" / "index.json").read_text())
    # Only the curated run is published — not every run in the tree.
    assert {r["path"] for r in index["runs"]} == {"outer_loop/demo"}


def test_run_json_matches_live_scan(snapshot: Path, data_root: Path):
    frozen = json.loads((_run_dir(snapshot) / "run.json").read_text())
    assert frozen == scan_run(data_root, "outer_loop/demo").model_dump()


def test_experiment_json_per_unit_matches_live_scan(snapshot: Path, data_root: Path):
    # Each experiment unit becomes its own static file (the ?unit= query string
    # cannot drive routing on a static host).
    frozen = json.loads((_run_dir(snapshot) / "experiment" / "experiment1.json").read_text())
    assert frozen == scan_run_experiment(data_root, "outer_loop/demo", "experiment1").model_dump()
    assert (_run_dir(snapshot) / "experiment" / "smoke.json").is_file()


def test_referenced_files_are_copied(snapshot: Path):
    fig = _run_dir(snapshot) / "files" / "analysis" / "loop_trajectory.png"
    assert fig.read_bytes()[:4] == b"\x89PNG"
    # The deployed experiment page is lazy-loaded by the frontend, so it must be
    # copied for units whose has_index_html is true (experiment1, not smoke).
    assert (_run_dir(snapshot) / "files" / "experiment1" / "experiment" / "index.html").is_file()
    assert not (_run_dir(snapshot) / "files" / "smoke" / "experiment" / "index.html").exists()


def test_frontend_is_self_contained_static(snapshot: Path):
    assert (snapshot / "app.js").is_file()
    assert (snapshot / "styles.css").is_file()
    html = (snapshot / "index.html").read_text()
    assert 'window.VIEWER_STATIC_BASE = "data"' in html
    # Absolute /static/ asset refs are rewritten to relative paths for static hosting.
    assert "/static/" not in html


def test_unknown_run_path_fails_loudly(data_root: Path, tmp_path: Path):
    with pytest.raises(ValueError, match="outer_loop/nope"):
        freeze_snapshot(
            data_root=data_root, out_dir=tmp_path / "x", run_paths=["outer_loop/nope"]
        )


def test_empty_run_list_fails_loudly(data_root: Path, tmp_path: Path):
    with pytest.raises(ValueError):
        freeze_snapshot(data_root=data_root, out_dir=tmp_path / "x", run_paths=[])


def test_missing_data_root_fails_loudly(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        freeze_snapshot(
            data_root=tmp_path / "does_not_exist",
            out_dir=tmp_path / "x",
            run_paths=["outer_loop/demo"],
        )
