import json

from src.pipelines.outer_loop.deployment.firebase import (
    CONSENT_GATE_MARKER,
    ensure_consent_gate,
    ensure_submit_bridge,
    load_consent_html,
    stage_experiment,
    write_firebase_config,
)
from src.pipelines.outer_loop.deployment.manifest import build_manifest
from src.runtime.config import REPO_ROOT


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


def test_submit_bridge_fetches_config_relatively_for_subpath_hosting():
    # The experiment is served from /e{run}/, so the config must be fetched
    # relative to the page, not from the site root.
    bridge = ensure_submit_bridge("<html><body></body></html>")
    assert '"auto_psych_config.json"' in bridge
    assert '"/auto_psych_config.json"' not in bridge
    # /submit and /results are global Cloud Functions at the site root.
    assert '"/submit"' in bridge


def test_stage_experiment_relativizes_agent_absolute_config_fetch(tmp_path):
    exp_dir, manifest = _manifest(tmp_path)
    # An agent that wrote its own submit with an absolute config fetch would
    # break under subpath hosting; staging must normalize it to a relative path.
    (exp_dir / "experiment" / "index.html").write_text(
        '<html><body><script>'
        'fetch("/auto_psych_config.json").then(r=>r.json());'
        'fetch("/submit",{method:"POST"});'
        'window.__experimentData=[];</script></body></html>',
        encoding="utf-8",
    )
    public_dir = stage_experiment(exp_dir, manifest, tmp_path / "public")
    staged = (public_dir / "index.html").read_text(encoding="utf-8")
    assert '"/auto_psych_config.json"' not in staged
    assert '"auto_psych_config.json"' in staged
    assert '"/submit"' in staged  # global function path is untouched


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


def test_ensure_consent_gate_injects_verbatim_irb_text():
    html = "<html><body><script>/* experiment */</script></body></html>"
    gated = ensure_consent_gate(html, load_consent_html())
    assert CONSENT_GATE_MARKER in gated
    # Verbatim phrases from the IRB consent text must appear.
    assert "at least 18 years old" in gated
    assert "Your anonymity is assured" in gated
    assert "langcoglab@stanford.edu" in gated


def test_ensure_consent_gate_is_idempotent():
    once = ensure_consent_gate("<html><body></body></html>", load_consent_html())
    twice = ensure_consent_gate(once, load_consent_html())
    assert once == twice
    assert once.count(CONSENT_GATE_MARKER) == twice.count(CONSENT_GATE_MARKER)


def test_stage_experiment_injects_consent_into_deployed_html(tmp_path):
    exp_dir, manifest = _manifest(tmp_path)
    public_dir = stage_experiment(exp_dir, manifest, tmp_path / "public")
    staged = (public_dir / "index.html").read_text(encoding="utf-8")
    assert CONSENT_GATE_MARKER in staged
    assert "Your anonymity is assured" in staged


def test_staging_into_subdir_preserves_sibling_experiments(tmp_path):
    # Deploying experiment 2 must not wipe experiment 1's staged page.
    exp1, manifest1 = _manifest(tmp_path / "a")
    exp2, manifest2 = _manifest(tmp_path / "b")
    public_root = tmp_path / "public"

    stage_experiment(exp1, manifest1, public_root / "e1")
    stage_experiment(exp2, manifest2, public_root / "e2")

    assert (public_root / "e1" / "index.html").exists()
    assert (public_root / "e2" / "index.html").exists()


def test_firebase_config_includes_locked_firestore_rules(tmp_path):
    _, manifest = _manifest(tmp_path)
    config_path = write_firebase_config(tmp_path / "firebase.generated.json", manifest)
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    assert cfg["firestore"]["rules"] == "firestore.rules"


def test_repo_firestore_rules_deny_all_client_access():
    rules = (REPO_ROOT / "firestore.rules").read_text(encoding="utf-8")
    assert "allow read, write: if false;" in rules
