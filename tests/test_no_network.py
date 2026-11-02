"""Agents without internet access: the shell filter, the denied web tools, the log check."""

import json
import os
import platform
import shutil
import subprocess
import sys

import pytest

from src.runtime import no_network

linux_x86 = pytest.mark.skipif(sys.platform != "linux" or platform.machine() != "x86_64",
                               reason="the seccomp filter is x86-64 Linux only")

PROBE = """
import socket
for fam in (socket.AF_INET, socket.AF_INET6):
    try:
        socket.socket(fam, socket.SOCK_STREAM).close()
        print("open", int(fam))
    except OSError as e:
        print("refused", int(fam), e.errno)
socket.socket(socket.AF_UNIX, socket.SOCK_STREAM).close()
print("unix ok")
"""


@linux_x86
def test_the_wrapper_shell_refuses_internet_sockets_to_everything_it_runs(tmp_path):
    wrapper = no_network.write_shell_wrapper(tmp_path / "shell", sys.executable, shutil.which("bash"))
    assert wrapper.name == "bash"  # opencode accepts $SHELL by name
    out = subprocess.run([str(wrapper), "-c", f"{sys.executable} -c '{PROBE}'"], capture_output=True, text=True,
                         check=True, cwd=tmp_path).stdout.split("\n")
    assert "refused 2 13" in out and "refused 10 13" in out and "unix ok" in out
    # Outside the wrapper the same probe opens both.
    plain = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True, check=True).stdout
    assert "open 2" in plain


def test_the_web_tools_are_denied_over_existing_rules():
    merged = json.loads(no_network.opencode_permission(json.dumps({"external_directory": "allow", "webfetch": "allow"})))
    assert merged == {"external_directory": "allow", "webfetch": "deny", "websearch": "deny", "codesearch": "deny"}


def test_a_completed_web_tool_call_in_the_log_is_found(tmp_path):
    log = tmp_path / "agent.jsonl"
    events = [
        {"type": "tool_use", "part": {"tool": "bash", "state": {"status": "completed", "input": {"command": "ls"}}}},
        {"type": "tool_use", "part": {"tool": "webfetch", "state": {"status": "error", "input": {"url": "x"}}}},
        {"type": "tool_use", "part": {"tool": "webfetch", "state": {"status": "completed", "input": {"url": "https://osf.io"}}}},
    ]
    log.write_text("\n".join(json.dumps(e) for e in events) + "\nnot json\n")
    used = no_network.web_tool_uses(log)
    assert len(used) == 1 and "osf.io" in used[0]
    assert no_network.web_tool_uses(tmp_path / "missing.jsonl") == []


def test_the_rsa_spawner_passes_its_network_setting_to_the_launcher(tmp_path, monkeypatch):
    from src.rsa.loop import orchestrator
    from src.runtime import coding_agent

    seen = {}

    def fake_run_coding_agent(prompt, **kwargs):
        seen.update(kwargs)
        return True, ""

    monkeypatch.setattr(coding_agent, "run_coding_agent", fake_run_coding_agent)
    spawn = orchestrator.coding_agent_spawner(models_dir=tmp_path, responses_path=tmp_path / "r.csv", timeout_sec=1,
                                              backend="opencode", model=None, agent_root=tmp_path, sandbox=True,
                                              network=False)
    (tmp_path / "candidate_0").mkdir()
    assert spawn(tmp_path / "candidate_0", "p") and seen["network"] is False


@linux_x86
def test_the_wrapper_is_verified_before_use(tmp_path):
    wrapper = no_network.write_shell_wrapper(tmp_path / "shell", sys.executable, shutil.which("bash"))
    no_network.verify_wrapper(wrapper, sys.executable)
    # A plain bash in its place (no filter) is caught.
    leaky = tmp_path / "leaky" / "bash"
    leaky.parent.mkdir()
    leaky.write_text(f'#!/bin/sh\nexec "{shutil.which("bash")}" "$@"\n')
    leaky.chmod(0o755)
    with pytest.raises(RuntimeError, match="left internet sockets open"):
        no_network.verify_wrapper(leaky, sys.executable)


def test_the_wrapper_works_from_wherever_its_directory_is_mounted(tmp_path):
    no_network.write_shell_wrapper(tmp_path / "shell", sys.executable, shutil.which("bash"))
    (tmp_path / "shell").rename(tmp_path / "moved")
    no_network.verify_wrapper(tmp_path / "moved" / "bash", sys.executable)


def _claude_log(path, *tools):
    events = [{"type": "assistant", "message": {"content": [{"type": "tool_use", "name": t, "input": {}}]}}
              for t in tools]
    events.append({"type": "result", "subtype": "success", "result": "done"})
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


def test_a_claude_tool_call_outside_the_offline_tools_is_found(tmp_path):
    log = _claude_log(tmp_path / "agent.jsonl", "Bash", "Monitor", "WebFetch", "Task")
    assert no_network.offline_tool_violations("claude", log) == ["WebFetch", "Task"]
    assert no_network.offline_tool_violations("claude", _claude_log(tmp_path / "ok.jsonl", "Bash", "Read")) == []


# ─────────────────────────────────────────────
# Integration: a sandboxed agent launched without network
# ─────────────────────────────────────────────

needs_bwrap = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")

# A stand-in for the claude CLI: it runs a socket probe through the shells it
# was given (as Claude Code's Bash tool does), opens a socket itself (the CLI
# must reach its API), tries to overwrite its shell, and reports all of it,
# with its argv, as the result of a stream-json run.
FAKE_CLAUDE = """#!{python} -I
import json, os, socket, sys, subprocess
PROBE = "import socket\\ntry:\\n    socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()\\n    print('open')\\nexcept OSError:\\n    print('refused')"
def through(shell):
    return subprocess.run([shell, "-c", sys.executable + ' -I -c "$PROBE"'], env={{**os.environ, "PROBE": PROBE}},
                          capture_output=True, text=True).stdout.strip()
socket.socket(socket.AF_INET, socket.SOCK_STREAM).close()
shell = os.environ["SHELL"]
try:
    open(shell, "a").close()
    shell_writable = True
except OSError:
    shell_writable = False
report = {{"argv": sys.argv[1:], "shell": through(shell),
          "claude_shell": through(os.environ.get("CLAUDE_CODE_SHELL", shell)), "shell_writable": shell_writable}}
for tool in {tools!r}:
    print(json.dumps({{"type": "assistant", "message": {{"content": [{{"type": "tool_use", "name": tool, "input": {{}}}}]}}}}))
print(json.dumps({{"type": "result", "subtype": "success", "result": json.dumps(report), "usage": {{}}}}))
"""


def _run_fake_claude(tmp_path, monkeypatch, *, network, tools=()):
    from src.runtime.coding_agent import run_coding_agent

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    cli = bin_dir / "claude"
    cli.write_text(FAKE_CLAUDE.format(python=sys.executable, tools=list(tools)))
    cli.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("CLAUDE_AUTH", "subscription")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test-token")
    tree = tmp_path / "tree"
    agent_dir = tree / "candidate_0"
    agent_dir.mkdir(parents=True)
    ok, result = run_coding_agent(
        "p", cwd=tree, log_path=agent_dir / "agent.jsonl", backend="claude", writable_dirs=[agent_dir],
        sandbox=True, network=network, on_summary=None, timeout_secs=120,
    )
    assert ok
    return json.loads(result)


@linux_x86
@needs_bwrap
def test_a_sandboxed_claude_agent_without_network_has_no_internet_in_its_shell(tmp_path, monkeypatch):
    report = _run_fake_claude(tmp_path, monkeypatch, network=False)
    assert report["shell"] == "refused" and report["claude_shell"] == "refused"
    assert not report["shell_writable"]
    tools = report["argv"][report["argv"].index("--tools") + 1].split(",")
    assert set(tools) == set(no_network.CLAUDE_OFFLINE_TOOLS)
    assert "WebFetch" not in tools and "WebSearch" not in tools


@linux_x86
@needs_bwrap
def test_a_sandboxed_claude_agent_with_network_keeps_it(tmp_path, monkeypatch):
    report = _run_fake_claude(tmp_path, monkeypatch, network=True)
    assert report["shell"] == "open" and "--tools" not in report["argv"]


@linux_x86
@needs_bwrap
def test_an_offline_agent_that_used_a_web_tool_stops_the_run(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="WebFetch"):
        _run_fake_claude(tmp_path, monkeypatch, network=False, tools=["Bash", "WebFetch"])
