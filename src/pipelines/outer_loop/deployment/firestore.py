"""Firestore metadata and result-shaping helpers for deployments."""

from __future__ import annotations

import csv
import io
from typing import Any

from .manifest import DeploymentManifest

ADC_HELP = (
    "Firestore metadata write needs Application Default Credentials. "
    "Run: gcloud auth application-default login && "
    "gcloud auth application-default set-quota-project auto-psych-2c5da"
)


def firestore_paths(manifest: DeploymentManifest) -> dict[str, str]:
    return {
        "study": f"studies/{manifest.study_id}",
        "deployment": f"deployments/{manifest.deployment_id}",
        "collection_session": f"collection_sessions/{manifest.collection_session_id}",
        "responses": f"collection_sessions/{manifest.collection_session_id}/responses",
    }


def metadata_documents(manifest: DeploymentManifest) -> dict[str, dict[str, Any]]:
    paths = firestore_paths(manifest)
    base = manifest.to_dict()
    return {
        paths["study"]: {
            "study_id": manifest.study_id,
            "project_id": manifest.project_id,
            "updated_at": manifest.created_at,
        },
        paths["deployment"]: {
            **base,
            "firestore_paths": paths,
        },
        paths["collection_session"]: {
            "collection_session_id": manifest.collection_session_id,
            "deployment_id": manifest.deployment_id,
            "study_id": manifest.study_id,
            "project_id": manifest.project_id,
            "experiment_id": manifest.experiment_id,
            "run_id": manifest.run_id,
            "agent_backend": manifest.agent_backend,
            "collection_owner": manifest.collection_owner,
            "firebase_project": manifest.firebase_project,
            "prolific_mode": manifest.prolific_mode,
            "prolific_study_id": manifest.prolific_study_id,
            "target_participants": manifest.total_available_places,
            "created_at": manifest.created_at,
            "status": "planned" if manifest.deploy_target == "dry-run" else "deployed",
        },
    }


def write_metadata(manifest: DeploymentManifest, client: Any | None = None) -> dict[str, str]:
    """Write study/deployment/session metadata to Firestore.

    A client can be injected by tests. In production this uses the Firebase
    project from the manifest.
    """
    if client is None:
        from google.cloud import firestore
        from google.auth.exceptions import DefaultCredentialsError

        try:
            client = firestore.Client(project=manifest.firebase_project)
        except DefaultCredentialsError as exc:
            raise RuntimeError(ADC_HELP) from exc

    docs = metadata_documents(manifest)
    try:
        for path, payload in docs.items():
            client.document(path).set(payload, merge=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to write Firestore deployment metadata. {ADC_HELP}") from exc
    return firestore_paths(manifest)


def validate_submit_payload(payload: dict[str, Any]) -> tuple[bool, str]:
    trials = payload.get("trials")
    if not isinstance(trials, list):
        return False, "trials must be a list"
    if payload.get("collection_session_id"):
        return True, ""
    if payload.get("project_id") and payload.get("run_id"):
        return True, ""
    return False, "collection_session_id or project_id/run_id is required"


def responses_to_csv(response_docs: list[tuple[str, dict[str, Any]]]) -> str:
    """Flatten Firestore response docs into the pipeline CSV shape."""
    rows: list[dict[str, Any]] = []
    for participant_index, (doc_id, data) in enumerate(response_docs):
        trials = data.get("trials") or []
        for trial_index, trial in enumerate(trials):
            if trial.get("sequence_a") is None or trial.get("sequence_b") is None:
                continue
            chose_left = trial.get("chose_left")
            if chose_left is None:
                continue
            rows.append(
                {
                    "participant_id": participant_index,
                    "participant_id_str": doc_id,
                    "trial_index": trial_index,
                    "sequence_a": str(trial["sequence_a"]),
                    "sequence_b": str(trial["sequence_b"]),
                    "chose_left": 1 if bool(chose_left) else 0,
                    "chose_right": 0 if bool(chose_left) else 1,
                    "model": "",
                }
            )

    fieldnames = [
        "participant_id",
        "participant_id_str",
        "trial_index",
        "sequence_a",
        "sequence_b",
        "chose_left",
        "chose_right",
        "model",
    ]
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()
