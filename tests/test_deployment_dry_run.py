import json

from src.pipelines.outer_loop.deployment.local import run_deployment


def test_dry_run_deployment_writes_manifest_config_and_staging(tmp_path):
    exp_dir = tmp_path / "experiment2"
    (exp_dir / "experiment").mkdir(parents=True)
    (exp_dir / "design").mkdir()
    (exp_dir / "experiment" / "index.html").write_text(
        "<html><body><script>window.__experimentData=[];</script></body></html>",
        encoding="utf-8",
    )
    (exp_dir / "experiment" / "config.json").write_text('{"experiment_url": null}\n')
    (exp_dir / "design" / "stimuli.json").write_text('[{"sequence_a":"HH","sequence_b":"HT"}]\n')

    manifest_path = run_deployment(
        exp_dir=exp_dir,
        project_id="subjective_randomness",
        run_id=2,
        deploy_target="dry-run",
        prolific_mode="none",
        agent_backend="claude",
        collection_owner="linas",
        firebase_project=None,
        firebase_region="us-central1",
        n_participants=5,
        repo_root=tmp_path,
    )

    assert manifest_path.exists()
    assert (exp_dir / "deployment" / "public" / "index.html").exists()
    config = json.loads((exp_dir / "experiment" / "config.json").read_text(encoding="utf-8"))
    assert config["deployment_id"].startswith("deploy_subjective_randomness-e2-")
    assert config["collection_owner"] == "linas"
    assert config["deploy_target"] == "dry-run"
