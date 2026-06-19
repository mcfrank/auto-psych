"""Fixtures for the live-study monitor: a manifest tree and fake data sources.

The monitor discovers studies from ``deployment_manifest.json`` files in the
data tree, then reads live participant data from Firestore and recruitment
status from Prolific. These fakes stand in for those two external services so
the monitor can be driven end-to-end without a network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# The full set of keys a real deployment manifest carries (see
# src/pipelines/outer_loop/deployment/manifest.py). The monitor only reads a
# handful, but fixtures write the whole shape so discovery is tested faithfully.
_MANIFEST_DEFAULTS: dict[str, Any] = {
    "project_id": "subjective_randomness",
    "run_id": 1,
    "deployment_id": "deploy_demo",
    "study_id": "study_subjective-randomness",
    "deploy_target": "firebase",
    "prolific_mode": "live",
    "agent_backend": "gemini",
    "collection_owner": "researcher",
    "firebase_project": "auto-psych-2c5da",
    "firebase_region": "us-central1",
    "results_api_url": "https://auto-psych-2c5da.web.app",
    "hosting_path": "e1-pilot",
    "prolific_completion_code": "AUTO_PSYCH_COMPLETE",
    "git_commit": "abc1234",
    "git_dirty": False,
}


def write_manifest(
    data_root: Path,
    *,
    rel: str,
    collection_session_id: str,
    experiment_id: str,
    prolific_study_id: str | None = None,
    deploy_target: str = "firebase",
    prolific_mode: str = "live",
    target_participants: int | None = 30,
    created_at: str = "2026-06-19T18:00:00Z",
    experiment_url: str | None = "https://auto-psych-2c5da.web.app/e1-pilot/",
    **overrides: Any,
) -> Path:
    """Write a deployment manifest under ``data_root/rel/deployment/``."""
    manifest = dict(_MANIFEST_DEFAULTS)
    manifest.update(
        {
            "collection_session_id": collection_session_id,
            "experiment_id": experiment_id,
            "prolific_study_id": prolific_study_id,
            "deploy_target": deploy_target,
            "prolific_mode": prolific_mode,
            "total_available_places": target_participants,
            "created_at": created_at,
            "experiment_url": experiment_url,
        }
    )
    manifest.update(overrides)
    path = data_root / rel / "deployment" / "deployment_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return path


def response_doc(
    participant_id: str,
    chose_left_per_trial: list[bool],
    *,
    created_at: str,
    prolific_pid: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build a (doc_id, data) Firestore response for the subjective-randomness task.

    Each trial shows two coin-flip sequences; ``chose_left`` records which the
    participant judged more random. The sequences here are dummies — aggregation
    only cares that they are present (valid trial) and what was chosen.
    """
    trials = [
        {"sequence_a": [0, 1, 1, 0], "sequence_b": [1, 1, 1, 1], "chose_left": chose_left}
        for chose_left in chose_left_per_trial
    ]
    data = {
        "prolific_pid": prolific_pid or participant_id,
        "trials": trials,
        "created_at": created_at,
    }
    return participant_id, data


class FakeFirestore:
    """A Firestore source backed by an in-memory ``{session_id: [response, ...]}``."""

    def __init__(self, responses_by_session: dict[str, list[tuple[str, dict[str, Any]]]]):
        self._responses = responses_by_session

    def list_responses(self, collection_session_id: str) -> list[tuple[str, dict[str, Any]]]:
        return list(self._responses.get(collection_session_id, []))


class FakeProlific:
    """A Prolific source backed by in-memory status/count maps keyed by study id."""

    def __init__(
        self,
        *,
        statuses: dict[str, dict[str, Any]] | None = None,
        counts: dict[str, dict[str, int]] | None = None,
        errors: dict[str, str] | None = None,
    ):
        self._statuses = statuses or {}
        self._counts = counts or {}
        self._errors = errors or {}

    def study_status(self, study_id: str) -> tuple[dict[str, Any] | None, str | None]:
        if study_id in self._errors:
            return None, self._errors[study_id]
        return self._statuses.get(study_id), None

    def submission_counts(self, study_id: str) -> tuple[dict[str, int] | None, str | None]:
        if study_id in self._errors:
            return None, self._errors[study_id]
        return self._counts.get(study_id), None
