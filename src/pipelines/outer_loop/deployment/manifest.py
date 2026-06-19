"""Deployment manifest and provenance helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
from typing import Any
import uuid

MANIFEST_FILENAME = "deployment_manifest.json"
CLIENT_CONFIG_FILENAME = "auto_psych_config.json"
VALID_DEPLOY_TARGETS = {"none", "dry-run", "firebase"}
VALID_PROLIFIC_MODES = {"none", "test", "live"}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip())
    cleaned = re.sub(r"-+", "-", cleaned).strip("-")
    return cleaned or "auto-psych"


def experiment_number_from_dir(exp_dir: Path) -> int:
    match = re.search(r"experiment(\d+)$", exp_dir.name)
    return int(match.group(1)) if match else 1


def git_metadata(repo_root: Path) -> dict[str, Any]:
    def _git(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return None
        return result.stdout.strip()

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
    }


@dataclass
class DeploymentManifest:
    project_id: str
    experiment_id: str
    run_id: int
    deployment_id: str
    collection_session_id: str
    study_id: str
    deploy_target: str
    prolific_mode: str
    agent_backend: str
    collection_owner: str
    firebase_project: str | None
    firebase_region: str
    experiment_url: str | None = None
    results_api_url: str | None = None
    hosting_path: str | None = None
    prolific_study_id: str | None = None
    prolific_completion_code: str | None = None
    prolific_redirect_url: str | None = None
    total_available_places: int | None = None
    created_at: str = field(default_factory=utc_now)
    git_commit: str | None = None
    git_dirty: bool | None = None
    source_experiment_dir: str | None = None
    staged_public_dir: str | None = None
    firebase_config_path: str | None = None
    firestore_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_client_config(self) -> dict[str, Any]:
        """Return the public config shipped with the deployed experiment."""
        return {
            "project_id": self.project_id,
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "study_id": self.study_id,
            "deployment_id": self.deployment_id,
            "collection_session_id": self.collection_session_id,
            "deploy_target": self.deploy_target,
            "agent_backend": self.agent_backend,
            "collection_owner": self.collection_owner,
            "firebase_project": self.firebase_project,
            "firebase_region": self.firebase_region,
            "experiment_url": self.experiment_url,
            "results_api_url": self.results_api_url,
            "prolific_mode": self.prolific_mode,
            "prolific_study_id": self.prolific_study_id,
            "prolific_completion_code": self.prolific_completion_code,
            "prolific_redirect_url": self.prolific_redirect_url,
            "total_available_places": self.total_available_places,
            "created_at": self.created_at,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeploymentManifest":
        return cls(**data)


def build_manifest(
    *,
    exp_dir: Path,
    project_id: str,
    run_id: int | None,
    deploy_target: str,
    prolific_mode: str,
    agent_backend: str,
    collection_owner: str,
    firebase_project: str | None,
    firebase_region: str,
    n_participants: int,
    repo_root: Path,
    run_label: str | None = None,
) -> DeploymentManifest:
    if deploy_target not in VALID_DEPLOY_TARGETS:
        raise ValueError(f"Unknown deploy target: {deploy_target}")
    if prolific_mode not in VALID_PROLIFIC_MODES:
        raise ValueError(f"Unknown Prolific mode: {prolific_mode}")

    resolved_run_id = run_id if run_id is not None else experiment_number_from_dir(exp_dir)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    git = git_metadata(repo_root)
    short_sha = (git.get("git_commit") or "nogit")[:7]
    experiment_id = f"{project_id}_experiment{resolved_run_id}"
    # The run label (an explicit --run-label, or a unique auto token) makes the
    # deployment/session ids and hosting path unique PER RUN, so parallel runs
    # never collide on Firestore session ids or deploy to the same path. The
    # e{run} part separates experiments within a single run.
    label = slug(run_label) if run_label else uuid.uuid4().hex[:8]
    id_base = slug(f"{project_id}-e{resolved_run_id}-{label}-{timestamp}-{short_sha}")
    site_root = (
        f"https://{firebase_project}.web.app"
        if deploy_target == "firebase" and firebase_project
        else None
    )
    # /submit and /results Cloud Functions stay at the site root (results_api_url).
    hosting_path = f"e{resolved_run_id}-{label}"
    experiment_url = f"{site_root}/{hosting_path}/" if site_root else None

    manifest = DeploymentManifest(
        project_id=project_id,
        experiment_id=experiment_id,
        run_id=resolved_run_id,
        deployment_id=f"deploy_{id_base}",
        collection_session_id=f"session_{id_base}",
        study_id=f"study_{slug(project_id)}",
        deploy_target=deploy_target,
        prolific_mode=prolific_mode,
        agent_backend=agent_backend,
        collection_owner=collection_owner,
        firebase_project=firebase_project,
        firebase_region=firebase_region,
        experiment_url=experiment_url,
        results_api_url=site_root,
        hosting_path=hosting_path,
        total_available_places=n_participants,
        git_commit=git.get("git_commit"),
        git_dirty=git.get("git_dirty"),
        source_experiment_dir=str(exp_dir / "experiment"),
    )
    return manifest


def manifest_dir(exp_dir: Path) -> Path:
    return exp_dir / "deployment"


def manifest_path(exp_dir: Path) -> Path:
    return manifest_dir(exp_dir) / MANIFEST_FILENAME


def experiment_manifest_path(exp_dir: Path) -> Path:
    return exp_dir / "experiment" / MANIFEST_FILENAME


def write_manifest(exp_dir: Path, manifest: DeploymentManifest) -> Path:
    out = manifest_path(exp_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    mirror = experiment_manifest_path(exp_dir)
    mirror.parent.mkdir(parents=True, exist_ok=True)
    mirror.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_manifest(path: Path) -> DeploymentManifest:
    return DeploymentManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))


def write_client_config(exp_dir: Path, manifest: DeploymentManifest, existing: dict[str, Any] | None = None) -> Path:
    config_path = exp_dir / "experiment" / "config.json"
    merged = dict(existing or {})
    merged.update(manifest.to_client_config())
    merged.setdefault("run_mode", "deployed" if manifest.deploy_target == "firebase" else "dry_run")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return config_path
