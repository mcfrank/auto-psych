from pathlib import Path

from src.pipelines.outer_loop.deployment.manifest import (
    build_manifest,
    write_client_config,
    write_manifest,
)


def test_manifest_contains_required_provenance(tmp_path):
    exp_dir = tmp_path / "experiment3"
    (exp_dir / "experiment").mkdir(parents=True)

    manifest = build_manifest(
        exp_dir=exp_dir,
        project_id="subjective_randomness",
        run_id=3,
        deploy_target="dry-run",
        prolific_mode="none",
        agent_backend="opencode",
        collection_owner="linas",
        firebase_project="auto-psych-test",
        firebase_region="us-central1",
        n_participants=12,
        repo_root=Path(__file__).resolve().parent.parent,
    )

    assert manifest.project_id == "subjective_randomness"
    assert manifest.experiment_id == "subjective_randomness_experiment3"
    assert manifest.deployment_id.startswith("deploy_subjective_randomness-e3-")
    assert manifest.collection_session_id.startswith("session_subjective_randomness-e3-")
    assert manifest.agent_backend == "opencode"
    assert manifest.collection_owner == "linas"
    assert "git_commit" in manifest.to_client_config()


def _firebase_manifest(tmp_path, run_id, run_label=None):
    return build_manifest(
        exp_dir=tmp_path / f"experiment{run_id}",
        project_id="subjective_randomness",
        run_id=run_id,
        deploy_target="firebase",
        prolific_mode="none",
        agent_backend="claude",
        collection_owner="ben",
        firebase_project="auto-psych-2c5da",
        firebase_region="us-central1",
        n_participants=5,
        repo_root=tmp_path,
        run_label=run_label,
    )


def test_firebase_manifest_isolates_experiments_by_subpath(tmp_path):
    # Within a run, each experiment is hosted under /e{run}-{label}/ so deploying a
    # later experiment never overwrites an earlier one. The /results Cloud Function
    # stays at the site root.
    m1 = _firebase_manifest(tmp_path, 1, run_label="pilot")
    m2 = _firebase_manifest(tmp_path, 2, run_label="pilot")
    assert m1.experiment_url == "https://auto-psych-2c5da.web.app/e1-pilot/"
    assert m2.experiment_url == "https://auto-psych-2c5da.web.app/e2-pilot/"
    assert m1.experiment_url != m2.experiment_url
    assert m1.results_api_url == "https://auto-psych-2c5da.web.app"
    assert m1.hosting_path == "e1-pilot"


def test_parallel_runs_get_distinct_hosting_paths_and_ids(tmp_path):
    # Two runs of the SAME experiment with no explicit label must not collide on
    # the hosting path OR the Firestore session/deployment ids, so they can run
    # and deploy in parallel.
    a = _firebase_manifest(tmp_path, 1)
    b = _firebase_manifest(tmp_path, 1)
    assert a.hosting_path != b.hosting_path
    assert a.hosting_path.startswith("e1-")
    assert b.hosting_path.startswith("e1-")
    assert a.collection_session_id != b.collection_session_id
    assert a.deployment_id != b.deployment_id


def test_manifest_and_client_config_are_written(tmp_path):
    exp_dir = tmp_path / "experiment1"
    (exp_dir / "experiment").mkdir(parents=True)
    (exp_dir / "experiment" / "config.json").write_text('{"experiment_url": null}\n')
    manifest = build_manifest(
        exp_dir=exp_dir,
        project_id="subjective_randomness",
        run_id=1,
        deploy_target="dry-run",
        prolific_mode="none",
        agent_backend="claude",
        collection_owner="linas",
        firebase_project=None,
        firebase_region="us-central1",
        n_participants=5,
        repo_root=tmp_path,
    )

    manifest_path = write_manifest(exp_dir, manifest)
    config_path = write_client_config(exp_dir, manifest, existing={"experiment_url": None})

    assert manifest_path.exists()
    assert (exp_dir / "experiment" / "deployment_manifest.json").exists()
    text = config_path.read_text()
    assert "collection_session_id" in text
    assert "deployment_id" in text
