import json

from src.pipelines.outer_loop.deployment.firebase import ensure_submit_bridge, stage_experiment
from src.pipelines.outer_loop.deployment.manifest import build_manifest


def _manifest(tmp_path):
    exp_dir = tmp_path / "experiment1"
    (exp_dir / "experiment").mkdir(parents=True)
    (exp_dir / "design").mkdir()
    (exp_dir / "experiment" / "index.html").write_text(
        "<!doctype html><html><body><script>window.__experimentData = [];</script></body></html>",
        encoding="utf-8",
    )
    (exp_dir / "design" / "stimuli.json").write_text('[{"sequence_a":"HH","sequence_b":"HT"}]\n')
    manifest = build_manifest(
        exp_dir=exp_dir,
        project_id="subjective_randomness",
        run_id=1,
        deploy_target="dry-run",
        prolific_mode="none",
        agent_backend="claude",
        collection_owner="linas",
        firebase_project="auto-psych-test",
        firebase_region="us-central1",
        n_participants=2,
        repo_root=tmp_path,
    )
    return exp_dir, manifest


def test_ensure_submit_bridge_is_idempotent():
    html = "<html><body></body></html>"
    once = ensure_submit_bridge(html)
    twice = ensure_submit_bridge(once)
    assert "/submit" in once
    assert once == twice


def test_stage_experiment_copies_files_and_leaves_source_untouched(tmp_path):
    exp_dir, manifest = _manifest(tmp_path)
    source_before = (exp_dir / "experiment" / "index.html").read_text(encoding="utf-8")

    public_dir = stage_experiment(exp_dir, manifest, tmp_path / "public")

    assert (public_dir / "index.html").exists()
    assert (public_dir / "stimuli.json").exists()
    assert "/submit" in (public_dir / "index.html").read_text(encoding="utf-8")
    cfg = json.loads((public_dir / "auto_psych_config.json").read_text(encoding="utf-8"))
    assert cfg["collection_session_id"] == manifest.collection_session_id
    assert (exp_dir / "experiment" / "index.html").read_text(encoding="utf-8") == source_before
