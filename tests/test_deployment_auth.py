"""Auth plumbing for the deployed Cloud Functions.

/results (all participant data) and /register_session (what makes /submit
accept a session) are guarded by a shared secret: the deployer's environment
provides AUTO_PSYCH_RESULTS_TOKEN, deploy staging provisions it into
functions/.env, and every pipeline read of /results sends it as a header. All
misconfigurations fail loudly — an open /results or an unregistered session
must never look like "no participants yet".
"""

from __future__ import annotations

import json

import pytest

from src.pipelines.outer_loop.collect import _results_request
from src.pipelines.outer_loop.deployment.firebase import (
    DeploymentError,
    register_collection_session,
    results_token,
    write_functions_env,
)
from src.pipelines.outer_loop.deployment.manifest import build_manifest

TOKEN_ENV = "AUTO_PSYCH_RESULTS_TOKEN"


def _manifest(tmp_path):
    exp = tmp_path / "experiment1"
    exp.mkdir(exist_ok=True)
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
        run_label="hero1",
    )


def test_results_token_missing_raises(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with pytest.raises(DeploymentError, match=TOKEN_ENV):
        results_token()


def test_write_functions_env_provisions_the_token(tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "sekrit")
    (tmp_path / "functions").mkdir()
    env_path = write_functions_env(tmp_path)
    assert env_path.read_text(encoding="utf-8") == "RESULTS_TOKEN=sekrit\n"


def test_register_collection_session_posts_token_and_payload(tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "sekrit")
    manifest = _manifest(tmp_path)
    captured = {}

    class _Resp:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["token"] = req.get_header("X-results-token")
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    register_collection_session(manifest)

    assert captured["url"].endswith("/register_session")
    assert captured["url"].startswith("https://")
    assert captured["token"] == "sekrit"
    assert (
        captured["payload"]["collection_session_id"]
        == manifest.collection_session_id
    )


def test_register_collection_session_failure_raises(tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "sekrit")
    manifest = _manifest(tmp_path)

    def failing_urlopen(req, timeout=None):
        raise OSError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    with pytest.raises(DeploymentError, match="register"):
        register_collection_session(manifest)


def test_results_request_sends_token_for_deployed_urls(monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "sekrit")
    req = _results_request("https://example.web.app/results?x=1")
    assert req.get_header("X-results-token") == "sekrit"


def test_results_request_missing_token_raises_for_deployed_urls(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    with pytest.raises(RuntimeError, match=TOKEN_ENV):
        _results_request("https://example.web.app/results?x=1")


def test_results_request_local_server_needs_no_token(monkeypatch):
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    req = _results_request("http://127.0.0.1:8123/results?x=1")
    assert req.get_header("X-results-token") is None
