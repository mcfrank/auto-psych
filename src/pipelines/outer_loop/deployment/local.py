"""Top-level deployment orchestration for dry-run and Firebase targets."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .firebase import (
    firebase_project_from_rc,
    load_experiment_config,
    run_firebase_deploy,
    stage_experiment,
    write_firebase_config,
)
from .firestore import firestore_paths, metadata_documents, write_metadata
from .manifest import build_manifest, write_client_config, write_manifest
from .prolific import build_prolific_plan, create_draft_study, publish_study


def run_deployment(
    *,
    exp_dir: Path,
    project_id: str,
    run_id: int,
    deploy_target: str,
    prolific_mode: str,
    agent_backend: str,
    collection_owner: str,
    firebase_project: str | None,
    firebase_region: str,
    n_participants: int,
    repo_root: Path,
) -> Path:
    if deploy_target == "none":
        raise ValueError("run_deployment should not be called with deploy_target='none'")

    resolved_project = firebase_project or firebase_project_from_rc(repo_root)
    if deploy_target == "firebase" and not resolved_project:
        raise RuntimeError("Firebase deploy requires --firebase-project or a real .firebaserc")

    existing_config = load_experiment_config(exp_dir)
    manifest = build_manifest(
        exp_dir=exp_dir,
        project_id=project_id,
        run_id=run_id,
        deploy_target=deploy_target,
        prolific_mode=prolific_mode,
        agent_backend=agent_backend,
        collection_owner=collection_owner,
        firebase_project=resolved_project,
        firebase_region=firebase_region,
        n_participants=n_participants,
        repo_root=repo_root,
    )
    manifest.firestore_paths = firestore_paths(manifest)

    if prolific_mode != "none":
        if deploy_target == "firebase":
            plan = create_draft_study(project_id, manifest, n_participants, prolific_mode)
        else:
            payload_manifest = manifest
            if not payload_manifest.experiment_url:
                payload_manifest = replace(
                    manifest,
                    experiment_url=f"https://example.invalid/auto-psych/{manifest.deployment_id}",
                )
                manifest.metadata["dry_run_experiment_url"] = payload_manifest.experiment_url
            plan = build_prolific_plan(
                project_id=project_id,
                manifest=payload_manifest,
                n_participants=n_participants,
                mode=prolific_mode,
            )
        manifest.prolific_study_id = plan.study_id
        manifest.prolific_completion_code = plan.completion_code
        manifest.prolific_redirect_url = plan.redirect_url
        manifest.metadata["prolific_payload"] = plan.payload
        if plan.test_participant_id:
            manifest.metadata["prolific_test_participant_id"] = plan.test_participant_id

    deployment_dir = exp_dir / "deployment"
    public_dir = repo_root / "public" if deploy_target == "firebase" else deployment_dir / "public"
    firebase_config_path = repo_root / "firebase.generated.json" if deploy_target == "firebase" else deployment_dir / "firebase.generated.json"
    manifest.staged_public_dir = str(public_dir)
    manifest.firebase_config_path = str(firebase_config_path)
    manifest.metadata["firestore_documents"] = metadata_documents(manifest)

    write_firebase_config(firebase_config_path, manifest)
    write_client_config(exp_dir, manifest, existing=existing_config)
    stage_experiment(exp_dir, manifest, public_dir)
    write_manifest(exp_dir, manifest)

    if deploy_target == "firebase":
        run_firebase_deploy(repo_root, manifest, firebase_config_path)
        try:
            write_metadata(manifest)
        except Exception as exc:
            manifest.metadata["firestore_metadata_write_error"] = str(exc)
            print(
                "  [deploy] Warning: Firebase deploy succeeded, but Firestore metadata "
                f"write failed: {exc}",
                flush=True,
            )
            write_manifest(exp_dir, manifest)
        if prolific_mode != "none" and manifest.prolific_study_id:
            published = publish_study(plan)
            manifest.metadata["prolific_published"] = published.published
            write_client_config(exp_dir, manifest, existing=existing_config)
            stage_experiment(exp_dir, manifest, public_dir)

    return write_manifest(exp_dir, manifest)
