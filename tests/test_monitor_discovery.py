"""Unit tests for discovering monitorable sessions from deployment manifests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.monitor.discovery import find_monitored_sessions
from tests.monitor_fixtures import write_manifest


def test_finds_firebase_sessions_newest_first(tmp_path: Path):
    root = tmp_path / "data"
    write_manifest(
        root,
        rel="run_a/experiment1",
        collection_session_id="session_a",
        experiment_id="proj_experiment1",
        created_at="2026-06-19T18:00:00Z",
    )
    write_manifest(
        root,
        rel="run_b/experiment1",
        collection_session_id="session_b",
        experiment_id="proj_experiment1",
        created_at="2026-06-19T20:00:00Z",
    )
    sessions = find_monitored_sessions(root)
    assert [s.collection_session_id for s in sessions] == ["session_b", "session_a"]
    assert sessions[0].target_participants == 30
    assert sessions[0].prolific_study_id is None or isinstance(sessions[0].prolific_study_id, str)


def test_excludes_dry_run_deployments(tmp_path: Path):
    root = tmp_path / "data"
    write_manifest(
        root,
        rel="run_a/experiment1",
        collection_session_id="session_real",
        experiment_id="proj_experiment1",
    )
    write_manifest(
        root,
        rel="run_b/experiment1",
        collection_session_id="session_dry",
        experiment_id="proj_experiment1",
        deploy_target="dry-run",
    )
    ids = {s.collection_session_id for s in find_monitored_sessions(root)}
    assert ids == {"session_real"}


def test_deduplicates_repeated_session_ids(tmp_path: Path):
    # The same manifest is mirrored under deployment/ and experiment/. Discovery
    # only walks deployment/, so a session id appears once even across mirrors.
    root = tmp_path / "data"
    path = write_manifest(
        root,
        rel="run_a/experiment1",
        collection_session_id="session_a",
        experiment_id="proj_experiment1",
    )
    # Mirror it under experiment/ the way write_manifest() does in production.
    mirror = path.parent.parent / "experiment" / "deployment_manifest.json"
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(path.read_text(), encoding="utf-8")

    sessions = find_monitored_sessions(root)
    assert [s.collection_session_id for s in sessions] == ["session_a"]


def test_missing_session_id_is_skipped(tmp_path: Path):
    root = tmp_path / "data"
    path = write_manifest(
        root,
        rel="run_a/experiment1",
        collection_session_id="placeholder",
        experiment_id="proj_experiment1",
    )
    import json

    data = json.loads(path.read_text())
    data.pop("collection_session_id")
    path.write_text(json.dumps(data), encoding="utf-8")
    assert find_monitored_sessions(root) == []


def test_malformed_manifest_fails_loudly(tmp_path: Path):
    root = tmp_path / "data"
    bad = root / "run_a" / "experiment1" / "deployment" / "deployment_manifest.json"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(ValueError):
        find_monitored_sessions(root)


def test_empty_data_root_returns_nothing(tmp_path: Path):
    root = tmp_path / "data"
    root.mkdir()
    assert find_monitored_sessions(root) == []
