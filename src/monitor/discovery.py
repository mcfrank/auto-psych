"""Discover which collection sessions to monitor from deployment manifests.

Every deployment writes a ``deployment_manifest.json`` into the data tree (see
``src/pipelines/outer_loop/deployment/manifest.py``). Each manifest names a
Firestore ``collection_session_id`` and, for live studies, a Prolific study id.
Reading these is how the monitor knows what to watch without being told.

We read the manifest JSON leniently — pulling only the fields the monitor needs
— so it keeps working as the manifest schema grows. Malformed JSON still fails
loudly: a corrupt manifest is a real problem, not something to skip silently.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

# Only these deploy targets actually write participant data to Firestore.
# A "dry-run" stages files but never collects, so it is not monitorable.
MONITORABLE_DEPLOY_TARGETS = ("firebase",)


@dataclass(frozen=True)
class MonitoredSession:
    """The slice of a deployment manifest the monitor needs to watch a study."""

    collection_session_id: str
    experiment_id: str
    project_id: str
    deploy_target: str
    prolific_mode: str
    prolific_study_id: str | None
    target_participants: int | None
    created_at: str | None
    experiment_url: str | None
    firebase_project: str | None
    manifest_path: str


def _load_manifest(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed deployment manifest at {path}: {exc}") from exc


def find_monitored_sessions(
    data_root: Path,
    *,
    include_targets: tuple[str, ...] = MONITORABLE_DEPLOY_TARGETS,
) -> list[MonitoredSession]:
    """Return monitorable sessions found under ``data_root``, newest first.

    Sessions are deduplicated by ``collection_session_id`` (each deploy mints a
    unique one) and sorted by ``created_at`` descending so the study you most
    likely just launched is at the top.
    """
    data_root = Path(data_root)
    by_session: dict[str, MonitoredSession] = {}

    for path in sorted(data_root.glob("**/deployment/deployment_manifest.json")):
        data = _load_manifest(path)
        session_id = data.get("collection_session_id")
        if not session_id:
            continue
        if data.get("deploy_target") not in include_targets:
            continue
        by_session[session_id] = MonitoredSession(
            collection_session_id=session_id,
            experiment_id=data.get("experiment_id", session_id),
            project_id=data.get("project_id", ""),
            deploy_target=data.get("deploy_target", ""),
            prolific_mode=data.get("prolific_mode", "none"),
            prolific_study_id=data.get("prolific_study_id"),
            target_participants=data.get("total_available_places"),
            created_at=data.get("created_at"),
            experiment_url=data.get("experiment_url"),
            firebase_project=data.get("firebase_project"),
            manifest_path=str(path),
        )

    return sorted(
        by_session.values(),
        key=lambda s: s.created_at or "",
        reverse=True,
    )
