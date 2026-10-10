"""Auth plumbing for the deployed Cloud Functions.

/results (all participant data, Prolific ids included) is guarded by a
shared secret: the deployer's environment provides AUTO_PSYCH_RESULTS_TOKEN,
deploy staging provisions it into functions/.env, and every pipeline read of
/results sends it as a header. After a deploy the pipeline checks the live
functions' behaviour (403 without the token, 200 with it): a functions deploy
from Sherlock exited 0 for months without replacing anything, leaving the
June /results open to anyone. All misconfigurations fail loudly.
"""

from __future__ import annotations

import json

import pytest

from src.pipelines.outer_loop.collect import _results_request
from src.pipelines.outer_loop.deployment.firebase import (
    DeploymentError,
    check_functions_deploy_output,
    verify_functions_live,
    write_firebase_config,
    results_token,
    write_functions_env,
)
from src.pipelines.outer_loop.deployment.manifest import build_manifest
from tests.paths import REPO_ROOT

# git provenance is read from the real checkout: build_manifest refuses to
# record a null commit, and a bare tmp_path is not a git repo.

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
        repo_root=REPO_ROOT,
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


class _Resp:
    def __init__(self, status):
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _fake_results(monkeypatch, *, without_token, with_token):
    """Answer /results by whether the request carries the token."""
    import urllib.error

    seen = []

    def fake_urlopen(req, timeout=None):
        token = req.get_header("X-results-token")
        seen.append((req.full_url, token))
        status = with_token if token else without_token
        if status == 200:
            return _Resp(200)
        raise urllib.error.HTTPError(req.full_url, status, "x", {}, None)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return seen


def test_verify_functions_live_accepts_a_guarded_results(tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, "sekrit")
    manifest = _manifest(tmp_path)
    seen = _fake_results(monkeypatch, without_token=403, with_token=200)

    verify_functions_live(manifest, attempts=1, delay=0)

    assert [token for _, token in seen] == [None, "sekrit"]
    assert all(url.startswith("https://") and "/results" in url for url, _ in seen)
    assert manifest.collection_session_id in seen[1][0]


def test_verify_functions_live_refuses_an_open_results(tmp_path, monkeypatch):
    # The June functions answer a tokenless read with 400 (they never check it).
    monkeypatch.setenv(TOKEN_ENV, "sekrit")
    manifest = _manifest(tmp_path)
    _fake_results(monkeypatch, without_token=400, with_token=200)
    with pytest.raises(DeploymentError, match="without the token"):
        verify_functions_live(manifest, attempts=1, delay=0)


def test_verify_functions_live_refuses_a_token_the_functions_do_not_hold(
    tmp_path, monkeypatch
):
    monkeypatch.setenv(TOKEN_ENV, "sekrit")
    manifest = _manifest(tmp_path)
    _fake_results(monkeypatch, without_token=403, with_token=403)
    with pytest.raises(DeploymentError, match="with the token"):
        verify_functions_live(manifest, attempts=1, delay=0)


def test_a_functions_deploy_whose_discovery_died_is_a_failure():
    # firebase-tools exits 0 when its function-discovery child dies (on
    # Sherlock: the child loses LD_LIBRARY_PATH and cannot load libstdc++).
    output = (
        "node: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.21' not found (required by node)\n"
        "Failed to call quitquitquit. This often means the server failed to start\n"
    )
    with pytest.raises(DeploymentError, match="LD_LIBRARY_PATH"):
        check_functions_deploy_output(output)
    check_functions_deploy_output("✔  Deploy complete!\n")  # does not raise


def test_there_is_no_register_session_function_or_route(tmp_path):
    source = (REPO_ROOT / "functions" / "index.js").read_text(encoding="utf-8")
    assert "register_session" not in source
    config_path = write_firebase_config(tmp_path / "firebase.json", _manifest(tmp_path))
    rewrites = json.loads(config_path.read_text())["hosting"]["rewrites"]
    assert sorted(r["source"] for r in rewrites) == ["/assign", "/results", "/submit"]


def test_the_live_env_gives_node_children_the_module_libraries():
    env_sh = (REPO_ROOT / "scripts" / "outer_loop_live" / "_env.sh").read_text(
        encoding="utf-8"
    )
    assert "node-wrapper" in env_sh
    assert 'export PATH="$WORK_ROOT/bin:' in env_sh  # first on PATH


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
