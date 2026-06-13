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
