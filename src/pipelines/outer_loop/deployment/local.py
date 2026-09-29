"""Top-level deployment orchestration for dry-run and Firebase targets.

Order for a Firebase deploy: the results token and (live mode) Prolific's
eligibility IDs are checked; the page is staged and deployed and its
collection session registered; only then is the Prolific draft created, its
id recorded at once, and, in live mode only, the study published. The draft
used to be created before the Firebase deploy, so a failed deploy left a
recorded live study that ``refuse_second_live_study`` then treated as
published, blocking a plain relaunch. Now a failed deploy records no study,
and a recorded study id always belongs to a page that was live.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .firebase import (
    firebase_project_from_rc,
    load_experiment_config,
    register_collection_session,
    results_token,
    run_firebase_deploy,
    stage_experiment,
    write_firebase_config,
    write_functions_env,
)
from .manifest import (
    archive_superseded_manifest,
    build_manifest,
    recorded_live_study,
    refuse_second_live_study,
    write_client_config,
    write_manifest,
)
from .prolific import (
    build_prolific_plan,
    create_draft_study,
    publish_study,
    verify_live_eligibility,
)


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
    run_label: str | None = None,
    publish_another_prolific_study: bool = False,
) -> Path:
    if deploy_target == "none":
        raise ValueError(
            "run_deployment should not be called with deploy_target='none'"
        )
    if publish_another_prolific_study:
        if recorded_live_study(exp_dir) is not None:
            archived = archive_superseded_manifest(exp_dir)
            print(
                f"  [deploy] Kept the earlier live study's manifest at {archived}",
                flush=True,
            )
    else:
        refuse_second_live_study(exp_dir, refused="deploy this experiment again")

    resolved_project = firebase_project or firebase_project_from_rc(repo_root)
    if deploy_target == "firebase" and not resolved_project:
        raise RuntimeError(
            "Firebase deploy requires --firebase-project or a real .firebaserc"
        )
    if deploy_target == "firebase":
        # Fail before any staging or Prolific work if the admin token for the
        # protected endpoints (/results, /register_session) is missing.
        results_token()
        if prolific_mode == "live":
            # A read-only check, so a changed eligibility mapping stops the
            # run before the page is deployed rather than after.
            verify_live_eligibility()

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
        run_label=run_label,
    )

    plan = None
    if prolific_mode != "none":
        # The study payload is built here, locally, for the record and for the
        # page's completion redirect. The study itself is created only after
        # the page is live (below); a dry run never creates one.
        payload_manifest = manifest
        if deploy_target != "firebase" and not payload_manifest.experiment_url:
            payload_manifest = replace(
                manifest,
                experiment_url=f"https://example.invalid/auto-psych/{manifest.deployment_id}",
            )
            manifest.metadata["dry_run_experiment_url"] = (
                payload_manifest.experiment_url
            )
        plan = build_prolific_plan(
            project_id=project_id,
            manifest=payload_manifest,
            n_participants=n_participants,
            mode=prolific_mode,
        )
        _record_plan(manifest, plan)

    deployment_dir = exp_dir / "deployment"
    # Stage under a per-experiment subdir (e.g. public/e2) so deploying a later
    # experiment leaves earlier experiments' live pages untouched. The whole
    # public/ tree is what Firebase Hosting serves.
    hosting_subdir = manifest.hosting_path or ""
    public_root = (
        repo_root / "public"
        if deploy_target == "firebase"
        else deployment_dir / "public"
    )
    public_dir = public_root / hosting_subdir
    firebase_config_path = (
        repo_root / "firebase.generated.json"
        if deploy_target == "firebase"
        else deployment_dir / "firebase.generated.json"
    )
    manifest.staged_public_dir = str(public_dir)
    manifest.firebase_config_path = str(firebase_config_path)

    write_firebase_config(firebase_config_path, manifest)
    write_client_config(exp_dir, manifest, existing=existing_config)
    stage_experiment(exp_dir, manifest, public_dir)
    write_manifest(exp_dir, manifest)

    if deploy_target == "firebase":
        # Provision the functions' shared secret before deploying them, then
        # register this deployment's collection session — /submit only accepts
        # registered sessions, so registration must succeed BEFORE any
        # participant can arrive (and long before a study is published).
        write_functions_env(repo_root)
        run_firebase_deploy(repo_root, manifest, firebase_config_path)
        register_collection_session(manifest)
        if plan is not None:
            # The page is live: create the draft now and record its id at
            # once (collection and the relaunch guard read it from disk).
            plan = create_draft_study(
                project_id, manifest, n_participants, prolific_mode
            )
            _record_plan(manifest, plan)
            write_client_config(exp_dir, manifest, existing=existing_config)
            write_manifest(exp_dir, manifest)
        # No other Firestore metadata write here: participant data flows
        # through the /submit and /results Cloud Functions (which write/read
        # the responses subcollection directly, with their own admin
        # credentials), so the pipeline needs no server-side Firestore access.
        # The deployment record lives in deployment_manifest.json on disk.
        #
        # Publish ONLY for live mode. Test mode leaves the study as a DRAFT you
        # preview yourself; none mode creates no study to publish.
        if prolific_mode == "live":
            published = publish_study(plan)
            manifest.metadata["prolific_published"] = published.published
            write_client_config(exp_dir, manifest, existing=existing_config)
            stage_experiment(exp_dir, manifest, public_dir)

    return write_manifest(exp_dir, manifest)


def _record_plan(manifest, plan) -> None:
    """Copy a Prolific plan's study id, completion code, redirect and payload
    into the manifest (the study id is ``None`` until the draft exists)."""
    manifest.prolific_study_id = plan.study_id
    manifest.prolific_completion_code = plan.completion_code
    manifest.prolific_redirect_url = plan.redirect_url
    manifest.metadata["prolific_payload"] = plan.payload
    if plan.test_participant_id:
        manifest.metadata["prolific_test_participant_id"] = plan.test_participant_id
