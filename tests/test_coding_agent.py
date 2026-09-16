"""Unit tests for the backend-agnostic coding-agent launcher."""

import pytest

from src.runtime.coding_agent import build_command, select_backend


def test_select_backend_defaults_to_opencode(monkeypatch):
    monkeypatch.delenv("CODING_AGENT", raising=False)
    assert select_backend(None) == "opencode"


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
    # JSON event output is load-bearing: token-usage tracking parses the
    # step_finish events, and the result text comes from the text events.
    assert "--format" in cmd and "json" in cmd
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


def test_build_command_claude_extra_args_precede_the_prompt():
    cmd = build_command(
        "claude", prompt="p", allowed_dirs=[], model=None,
        extra_args=["--max-turns", "50", "--disallowedTools", "Bash(scancel:*)"],
    )
    assert cmd[-1] == "p" and cmd[-2] == "-p"
    idx = cmd.index("--max-turns")
    assert cmd[idx + 1] == "50" and idx < cmd.index("-p")
    assert "Bash(scancel:*)" in cmd


def test_build_command_opencode_extra_args_precede_the_prompt():
    cmd = build_command(
        "opencode", prompt="p", allowed_dirs=[], model=None, extra_args=["--dir", "/x"],
    )
    assert cmd[-1] == "p" and "--dir" in cmd and cmd.index("--dir") < len(cmd) - 1


def test_build_command_codex_exec_json_reads_the_prompt_from_stdin():
    from src.runtime.coding_agent import prompt_via_stdin

    cmd = build_command("codex", prompt="review this", allowed_dirs=[], model=None)
    assert cmd[:2] == ["codex", "exec"]
    assert "--json" in cmd and "--skip-git-repo-check" in cmd
    assert cmd[cmd.index("--sandbox") + 1] == "danger-full-access"
    assert cmd[cmd.index("--model") + 1] == "gpt-5.6-sol"
    assert cmd[-1] == "-" and prompt_via_stdin("codex", "review this")


def test_long_claude_prompt_goes_to_stdin_and_opencode_refuses_it():
    from src.runtime.coding_agent import STDIN_PROMPT_THRESHOLD, prompt_via_stdin

    short, long = "p", "x" * (STDIN_PROMPT_THRESHOLD + 1)
    assert build_command("claude", prompt=short, allowed_dirs=[], model=None)[-2:] == ["-p", short]
    assert build_command("claude", prompt=long, allowed_dirs=[], model=None)[-1] == "-p"
    assert prompt_via_stdin("claude", long) and not prompt_via_stdin("claude", short)
    with pytest.raises(ValueError, match="argv limit"):
        build_command("opencode", prompt=long, allowed_dirs=[], model=None)


def test_codex_stream_reads_last_message_and_sums_usage():
    from src.runtime.coding_agent import _CodexStream

    stream = _CodexStream()
    stream.feed({"type": "item.completed", "item": {"type": "agent_message", "text": "first"}})
    stream.feed({"type": "item.completed", "item": {"type": "command_execution", "command": "ls"}})
    stream.feed({"type": "turn.completed", "usage": {"input_tokens": 1000, "cached_input_tokens": 600, "output_tokens": 50, "reasoning_output_tokens": 20}})
    stream.feed({"type": "item.completed", "item": {"type": "agent_message", "text": "final answer"}})
    stream.feed({"type": "turn.completed", "usage": {"input_tokens": 500, "cached_input_tokens": 0, "output_tokens": 10}})
    assert stream.result_text() == "final answer"
    usage = stream.usage_fields()
    assert usage["input_tokens"] == 900 and usage["cache_read_tokens"] == 600
    assert usage["output_tokens"] == 60 and usage["reasoning_tokens"] == 20
