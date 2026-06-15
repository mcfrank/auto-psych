"""Unit tests for the backend-agnostic coding-agent launcher."""

import pytest

from src.runtime.coding_agent import build_command, select_backend


def test_select_backend_defaults_to_claude(monkeypatch):
    monkeypatch.delenv("CODING_AGENT", raising=False)
    assert select_backend(None) == "claude"


def test_select_backend_env_fallback(monkeypatch):
    monkeypatch.setenv("CODING_AGENT", "opencode")
    assert select_backend(None) == "opencode"


def test_select_backend_explicit_overrides_env(monkeypatch):
    monkeypatch.setenv("CODING_AGENT", "opencode")
    assert select_backend("claude") == "claude"


def test_select_backend_rejects_unknown(monkeypatch):
    monkeypatch.delenv("CODING_AGENT", raising=False)
    with pytest.raises(ValueError):
        select_backend("gemini")


def test_build_command_claude_uses_add_dir_and_stream_json(tmp_path):
    cmd = build_command(
        "claude",
        prompt="do the thing",
        allowed_dirs=[tmp_path / "a", tmp_path / "b"],
        model=None,
    )
    assert cmd[0] == "claude"
    assert "--output-format" in cmd and "stream-json" in cmd
    assert "--dangerously-skip-permissions" in cmd
    assert cmd.count("--add-dir") == 2
    assert str(tmp_path / "a") in cmd
    assert "claude-sonnet-4-6" in cmd
    assert cmd[-1] == "do the thing"


def test_build_command_claude_model_override(tmp_path):
    cmd = build_command("claude", prompt="p", allowed_dirs=[], model="claude-opus-4-7")
    assert "claude-opus-4-7" in cmd
    assert "claude-sonnet-4-6" not in cmd


def test_build_command_opencode_uses_provider_model_and_no_add_dir(tmp_path):
    cmd = build_command(
        "opencode",
        prompt="do the thing",
        allowed_dirs=[tmp_path / "a"],
        model=None,
    )
    assert cmd[0] == "opencode"
    assert cmd[1] == "run"
    assert "-m" in cmd
    # opencode model ids are provider-prefixed (provider/model); assert the
    # format rather than a specific provider so swapping the default backend
    # model does not break this test.
    assert "/" in cmd[cmd.index("-m") + 1]
    assert "--add-dir" not in cmd
    assert cmd[-1] == "do the thing"


def test_build_command_rejects_unknown_backend():
    with pytest.raises(ValueError):
        build_command("gemini", prompt="p", allowed_dirs=[], model=None)
