"""Integration tests for the live-study monitor web server.

These outside-in tests drive the public HTTP surface. The monitor discovers
studies from deployment manifests on disk, then reports live participant data
(from a fake Firestore) and recruitment status (from a fake Prolific). The
single most important behavior is catching degenerate data — e.g. every
participant choosing the same side — which silently ruined an earlier pilot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.monitor.server import MonitorSources, create_app
from tests.monitor_fixtures import (
    FakeFirestore,
    FakeProlific,
    response_doc,
    write_manifest,
)

HEALTHY_SESSION = "session_healthy"
DEGENERATE_SESSION = "session_degenerate"
HEALTHY_STUDY = "prolific_study_healthy"


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    write_manifest(
        root,
        rel="outer_loop/subjective_randomness/experiment1",
        collection_session_id=HEALTHY_SESSION,
        experiment_id="subjective_randomness_experiment1",
        prolific_study_id=HEALTHY_STUDY,
        target_participants=30,
        created_at="2026-06-19T18:00:00Z",
    )
    write_manifest(
        root,
        rel="outer_loop/subjective_randomness/experiment2",
        collection_session_id=DEGENERATE_SESSION,
        experiment_id="subjective_randomness_experiment2",
        prolific_study_id=None,
        prolific_mode="none",
        target_participants=10,
        created_at="2026-06-19T19:00:00Z",
    )
    # A dry-run deployment never reaches Firestore — it must not be monitored.
    write_manifest(
        root,
        rel="outer_loop/subjective_randomness/dry",
        collection_session_id="session_dryrun",
        experiment_id="subjective_randomness_experiment3",
        deploy_target="dry-run",
        created_at="2026-06-19T20:00:00Z",
    )
    return root


@pytest.fixture
def sources() -> MonitorSources:
    firestore = FakeFirestore(
        {
            HEALTHY_SESSION: [
                response_doc("PID_A", [True, False, True, False, True], created_at="2026-06-19T18:05:00Z"),
                response_doc("PID_B", [False, True, False, False, True], created_at="2026-06-19T18:09:00Z"),
            ],
            # Every participant chose left on every trial: the canary.
            DEGENERATE_SESSION: [
                response_doc("PID_C", [True] * 6, created_at="2026-06-19T19:05:00Z"),
                response_doc("PID_D", [True] * 6, created_at="2026-06-19T19:06:00Z"),
                response_doc("PID_E", [True] * 6, created_at="2026-06-19T19:07:00Z"),
            ],
        }
    )
    prolific = FakeProlific(
        statuses={HEALTHY_STUDY: {"status": "ACTIVE", "total_available_places": 30, "places_taken": 2}},
        counts={HEALTHY_STUDY: {"AWAITING_REVIEW": 2, "APPROVED": 0, "RETURNED": 1, "TIMED_OUT": 0}},
    )
    return MonitorSources(firestore=firestore, prolific=prolific)


@pytest.fixture
def client(data_root: Path, sources: MonitorSources):
    app = create_app(data_root=data_root, sources=sources)
    app.config.update(TESTING=True)
    return app.test_client()


def test_index_page_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"<div id=\"app\">" in resp.data


def test_sessions_lists_only_real_deployments_newest_first(client):
    body = client.get("/api/sessions").get_json()
    ids = [s["collection_session_id"] for s in body["sessions"]]
    # dry-run is excluded; the later-created session sorts first.
    assert ids == [DEGENERATE_SESSION, HEALTHY_SESSION]


def test_session_summary_reports_live_progress(client):
    body = client.get("/api/sessions").get_json()
    healthy = next(s for s in body["sessions"] if s["collection_session_id"] == HEALTHY_SESSION)
    assert healthy["experiment_id"] == "subjective_randomness_experiment1"
    assert healthy["n_responses"] == 2
    assert healthy["n_with_data"] == 2
    assert healthy["target_participants"] == 30
    assert healthy["overall_p_left"] == pytest.approx(0.5)
    assert healthy["n_degenerate_participants"] == 0
    assert healthy["has_warning"] is False
    assert healthy["last_submission_at"] == "2026-06-19T18:09:00Z"


def test_summary_flags_degenerate_session(client):
    body = client.get("/api/sessions").get_json()
    bad = next(s for s in body["sessions"] if s["collection_session_id"] == DEGENERATE_SESSION)
    assert bad["overall_p_left"] == pytest.approx(1.0)
    assert bad["n_degenerate_participants"] == 3
    assert bad["has_warning"] is True


def test_session_detail_has_per_participant_breakdown(client):
    detail = client.get(f"/api/session/{HEALTHY_SESSION}").get_json()
    pids = {p["participant_id"] for p in detail["participants"]}
    assert pids == {"PID_A", "PID_B"}
    pid_a = next(p for p in detail["participants"] if p["participant_id"] == "PID_A")
    assert pid_a["n_valid_trials"] == 5
    assert pid_a["n_left"] == 3
    assert pid_a["p_left"] == pytest.approx(0.6)
    assert pid_a["degenerate"] is False


def test_session_detail_includes_prolific_status(client):
    detail = client.get(f"/api/session/{HEALTHY_SESSION}").get_json()
    prolific = detail["prolific"]
    assert prolific["status"] == "ACTIVE"
    assert prolific["places_taken"] == 2
    assert prolific["counts"]["AWAITING_REVIEW"] == 2
    assert prolific["counts"]["RETURNED"] == 1
    assert prolific["error"] is None


def test_session_detail_choice_balance_warns_on_degenerate(client):
    detail = client.get(f"/api/session/{DEGENERATE_SESSION}").get_json()
    balance = detail["choice_balance"]
    assert balance["is_degenerate"] is True
    assert balance["p_left"] == pytest.approx(1.0)
    assert balance["warning"]  # a human-readable explanation is present
    # No Prolific study was attached: status is absent but not an error.
    assert detail["prolific"]["status"] is None
    assert detail["prolific"]["error"] is None


def test_unknown_session_returns_404(client):
    resp = client.get("/api/session/session_does_not_exist")
    assert resp.status_code == 404


def test_prolific_error_surfaces_without_breaking_detail(data_root: Path):
    firestore = FakeFirestore(
        {HEALTHY_SESSION: [response_doc("PID_A", [True, False], created_at="2026-06-19T18:05:00Z")]}
    )
    prolific = FakeProlific(errors={HEALTHY_STUDY: "GET /studies/ 503: upstream down"})
    app = create_app(data_root=data_root, sources=MonitorSources(firestore=firestore, prolific=prolific))
    app.config.update(TESTING=True)
    detail = app.test_client().get(f"/api/session/{HEALTHY_SESSION}").get_json()
    # The Firestore-derived data is still present; the Prolific failure is shown.
    assert detail["n_responses"] == 1
    assert "503" in detail["prolific"]["error"]
