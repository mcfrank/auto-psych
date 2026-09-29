"""Claude agents are billed the way the run says: subscription or API, never both.

The sandbox used to require CLAUDE_CODE_OAUTH_TOKEN (the user's subscription)
while its environment allowlist also passed every ``ANTHROPIC_*`` variable, and
the Claude CLI prefers ANTHROPIC_API_KEY over the OAuth token: a stray key in
.secrets would silently move a subscription run onto per-token API billing.
Now a run whose agents use the ``claude`` backend states its mode (config key
``agent.claude_auth``, CLI flag ``--claude-auth``, or ``CLAUDE_AUTH`` in the job
scripts), fails before any agent starts without it or without the mode's
credential, and each agent gets that credential and no other.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest
import tyro

from src.runtime import token_usage
from src.runtime.agent_sandbox import require_claude_auth, sandbox_command
from tests.paths import REPO_ROOT
from tests.test_agent_sandbox import _fake_bwrap

TOKEN, KEY = "oauth-token", "api-key"
HARNESS_ENV = {
    "PATH": "/usr/bin",
    "CLAUDE_CODE_OAUTH_TOKEN": TOKEN,
    "ANTHROPIC_API_KEY": KEY,
    "ANTHROPIC_BASE_URL": "https://example.invalid",
}


# ── the run states its mode, and holds that mode's credential ───────────────


def test_a_claude_run_without_a_mode_fails():
    with pytest.raises(RuntimeError, match="claude_auth"):
        require_claude_auth("claude", None, env=dict(HARNESS_ENV))


def test_an_unknown_mode_fails():
    with pytest.raises(ValueError, match="billing mode"):
        require_claude_auth("claude", "free", env=dict(HARNESS_ENV))


@pytest.mark.parametrize("mode, credential", [
    ("subscription", "CLAUDE_CODE_OAUTH_TOKEN"),
    ("api", "ANTHROPIC_API_KEY"),
])
def test_a_mode_without_its_credential_fails(mode, credential):
    env = {k: v for k, v in HARNESS_ENV.items() if k != credential}
    with pytest.raises(RuntimeError, match=credential):
        require_claude_auth("claude", mode, env=env)


@pytest.mark.parametrize("mode", ["subscription", "api"])
def test_a_stated_mode_is_exported_for_the_sandbox(mode):
    env = dict(HARNESS_ENV)
    assert require_claude_auth("claude", mode, env=env) == mode
    assert env["CLAUDE_AUTH"] == mode


def test_the_job_scripts_environment_variable_states_the_mode():
    env = {**HARNESS_ENV, "CLAUDE_AUTH": "api"}
    assert require_claude_auth("claude", None, env=env) == "api"
    # The flag or config key wins over the environment.
    assert require_claude_auth("claude", "subscription", env=env) == "subscription"


def test_other_backends_need_no_mode():
    assert require_claude_auth("opencode", None, env={}) is None


# ── the agent gets its mode's credential and no other ──────────────────────


def _agent_env(tmp_path, monkeypatch, harness_env):
    _fake_bwrap(monkeypatch, "claude")
    _, env = sandbox_command(
        ["claude"], backend="claude", cwd=tmp_path, writable_dirs=[],
        agent_dir=tmp_path, env={**harness_env, "HOME": str(tmp_path)},
    )
    return env


def test_a_subscription_agent_gets_the_token_and_no_anthropic_variable(tmp_path, monkeypatch):
    env = _agent_env(tmp_path, monkeypatch, {**HARNESS_ENV, "CLAUDE_AUTH": "subscription"})
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == TOKEN
    assert not [k for k in env if k.startswith("ANTHROPIC_")]


def test_an_api_agent_gets_the_key_and_not_the_token(tmp_path, monkeypatch):
    env = _agent_env(tmp_path, monkeypatch, {**HARNESS_ENV, "CLAUDE_AUTH": "api"})
    assert env["ANTHROPIC_API_KEY"] == KEY
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in env
    assert "CLAUDE_AUTH" not in env  # the harness's setting, not the agent's


def test_a_sandboxed_claude_agent_without_a_mode_does_not_start(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="billing mode"):
        _agent_env(tmp_path, monkeypatch, HARNESS_ENV)


def test_a_sandboxed_api_agent_without_a_key_does_not_start(tmp_path, monkeypatch):
    env = {"PATH": "/usr/bin", "CLAUDE_CODE_OAUTH_TOKEN": TOKEN, "CLAUDE_AUTH": "api"}
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        _agent_env(tmp_path, monkeypatch, env)


def test_the_mode_is_recorded_with_the_agents_token_usage(tmp_path, monkeypatch):
    import src.runtime.coding_agent as coding_agent

    def fake_sandbox_command(cmd, *, env, **kwargs):
        return [sys.executable, "-c", "print('{\"type\": \"result\", \"subtype\": \"success\", \"result\": \"ok\"}')"], dict(env)

    monkeypatch.setattr(coding_agent, "sandbox_command", fake_sandbox_command)
    token_usage.reset_usage_log()
    coding_agent.run_coding_agent(
        "p", cwd=tmp_path, log_path=tmp_path / "agent" / "agent.jsonl", backend="claude",
        allowed_dirs=[], writable_dirs=[tmp_path / "agent"], sandbox=True, on_summary=None,
        env={**HARNESS_ENV, "CLAUDE_AUTH": "subscription"},
    )
    summary = token_usage.summarize(token_usage.records_since(0))
    token_usage.reset_usage_log()
    assert summary["claude_auth"] == ["subscription"]


# ── the entry points take the setting and check it first ────────────────────

HOLDOUT_SCRIPTS = [
    REPO_ROOT / "scripts" / "subjective_randomness" / "holdout_recovery.py",
    REPO_ROOT / "scripts" / "subjective_randomness" / "impossible_holdout_recovery.py",
]


@pytest.mark.parametrize("script", HOLDOUT_SCRIPTS, ids=lambda p: p.name)
def test_holdout_entry_points_take_claude_auth(script, monkeypatch):
    spec = importlib.util.spec_from_file_location(f"_entry_{script.stem}", script)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    args_type = module.Args
    args = tyro.cli(
        args_type,
        args=["--config", "c.yaml", "--out", "o.json", "--backend", "claude",
              "--claude-auth", "api"],
    )
    assert args.claude_auth == "api"


def test_a_holdout_run_with_claude_and_no_mode_stops_before_anything_runs(tmp_path, monkeypatch):
    from src.subjective_randomness.config import load_config
    from src.subjective_randomness.holdout_recovery import run_holdout_recovery_from_config

    monkeypatch.delenv("CLAUDE_AUTH", raising=False)
    config_path = (
        REPO_ROOT / "scripts" / "subjective_randomness" / "configs"
        / "holdout_recovery_faithful.yaml"
    )
    results_root = tmp_path / "results"
    with pytest.raises(RuntimeError, match="claude_auth"):
        run_holdout_recovery_from_config(
            load_config(config_path), config_path, results_root, backend_override="claude",
        )
    assert not results_root.exists()


def test_a_holdout_config_states_the_mode(tmp_path, monkeypatch):
    """agent.claude_auth reaches the check (which then wants its credential)."""
    from src.subjective_randomness.config import load_config
    from src.subjective_randomness.holdout_recovery import run_holdout_recovery_from_config

    monkeypatch.delenv("CLAUDE_AUTH", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    config_path = (
        REPO_ROOT / "scripts" / "subjective_randomness" / "configs"
        / "holdout_recovery_faithful.yaml"
    )
    config = load_config(config_path)
    config["agent"] = {**config["agent"], "backend": "claude", "claude_auth": "api"}
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        run_holdout_recovery_from_config(config, config_path, tmp_path / "results")


def test_the_live_entry_point_stops_a_claude_run_without_a_mode(monkeypatch):
    from src.pipelines.outer_loop import run

    monkeypatch.delenv("CLAUDE_AUTH", raising=False)
    monkeypatch.setenv("CODING_AGENT", "opencode")  # main exports its backend here
    args = tyro.cli(
        run.Args,
        args=["--project", "subjective_randomness", "--experiment", "1",
              "--agent", "2_design", "--coding-agent", "claude"],
    )
    with pytest.raises(RuntimeError, match="claude_auth"):
        run.main(args)


def test_the_job_scripts_pass_claude_auth_through():
    slurm = REPO_ROOT / "scripts" / "subjective_randomness" / "slurm"
    for script in (
        slurm / "holdout_recovery_array.sbatch",
        slurm / "impossible_holdout_recovery_array.sbatch",
    ):
        assert '${CLAUDE_AUTH:+--claude-auth "$CLAUDE_AUTH"}' in script.read_text()
    live = (REPO_ROOT / "scripts" / "outer_loop_live" / "run_live.sbatch").read_text()
    assert live.count('${CLAUDE_AUTH:+--claude-auth "$CLAUDE_AUTH"}') == 2


def test_secrets_example_names_both_credentials():
    text = (REPO_ROOT / ".secrets.example").read_text()
    assert "\nANTHROPIC_API_KEY=" in text
    assert "\nCLAUDE_CODE_OAUTH_TOKEN=" in text


def test_the_harness_environment_scripts_export_every_secrets_key():
    """ANTHROPIC_API_KEY reaches the harness like the other keys: _env.sh
    exports every KEY=value line of .secrets."""
    for env_sh in (
        REPO_ROOT / "scripts" / "subjective_randomness" / "slurm" / "_env.sh",
        REPO_ROOT / "scripts" / "outer_loop_live" / "_env.sh",
    ):
        assert 'export "$k=$(echo "$v" | xargs)"' in env_sh.read_text()
