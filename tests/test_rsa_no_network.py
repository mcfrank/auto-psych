"""Agents without internet access: the shell filter, the denied web tools, the log check."""

import json
import platform
import shutil
import subprocess
import sys

import pytest

from src.rsa.loop import no_network

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


def test_the_spawner_refuses_a_backend_it_cannot_cut_off(tmp_path):
    from src.rsa.loop.orchestrator import coding_agent_spawner

    with pytest.raises(ValueError, match="opencode backend only"):
        coding_agent_spawner(models_dir=tmp_path, responses_path=tmp_path / "r.csv", timeout_sec=1, backend="claude",
                             model=None, agent_root=None, sandbox=True, network=False, shell_dir=tmp_path / "s")
    spawn = coding_agent_spawner(models_dir=tmp_path, responses_path=tmp_path / "r.csv", timeout_sec=1,
                                 backend="opencode", model=None, agent_root=None, sandbox=True, network=False,
                                 shell_dir=tmp_path / "s")
    assert callable(spawn) and (tmp_path / "s" / "bash").exists()
