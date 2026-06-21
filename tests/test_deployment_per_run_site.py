"""Parallel runs must deploy to their OWN Firebase Hosting site, so concurrent
``firebase deploy`` calls don't clobber a shared live site (the failure mode that
took down run2/run3: a full-site hosting deploy replaces everything, so the last
run wins and the others 404).

``AUTO_PSYCH_HOSTING_SITE`` selects the per-run site; without it, the default
project site is used (backward compat for single pilots).
"""

from __future__ import annotations

import json

from src.pipelines.outer_loop.deployment.firebase import write_firebase_config
from src.pipelines.outer_loop.deployment.manifest import build_manifest


def _build(tmp_path, monkeypatch, site_env):
    exp = tmp_path / "experiment1"
    exp.mkdir()
    if site_env is None:
        monkeypatch.delenv("AUTO_PSYCH_HOSTING_SITE", raising=False)
    else:
        monkeypatch.setenv("AUTO_PSYCH_HOSTING_SITE", site_env)
    return build_manifest(
        exp_dir=exp,
        project_id="subjective_randomness",
        run_id=1,
        deploy_target="firebase",
        prolific_mode="live",
        agent_backend="opencode",
        collection_owner="me",
        firebase_project="auto-psych-2c5da",
        firebase_region="us-central1",
        n_participants=40,
        repo_root=tmp_path,
        run_label="run2",
    )


def test_per_run_site_drives_experiment_url(tmp_path, monkeypatch):
    m = _build(tmp_path, monkeypatch, "auto-psych-2c5da-run2")
    assert m.hosting_site == "auto-psych-2c5da-run2"
    assert m.experiment_url == "https://auto-psych-2c5da-run2.web.app/e1-run2/"
    assert m.results_api_url == "https://auto-psych-2c5da-run2.web.app"


def test_default_site_when_env_unset(tmp_path, monkeypatch):
    m = _build(tmp_path, monkeypatch, None)
    assert m.hosting_site == "auto-psych-2c5da"  # falls back to the project's default site
    assert m.experiment_url == "https://auto-psych-2c5da.web.app/e1-run2/"


def test_firebase_config_targets_the_hosting_site(tmp_path, monkeypatch):
    m = _build(tmp_path, monkeypatch, "auto-psych-2c5da-run2")
    cfg_path = tmp_path / "firebase.generated.json"
    write_firebase_config(cfg_path, m)
    cfg = json.loads(cfg_path.read_text())
    assert cfg["hosting"]["site"] == "auto-psych-2c5da-run2"
