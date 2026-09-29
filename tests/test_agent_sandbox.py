"""A sandboxed loop agent can read and write its own run tree and nothing else.

The loop's agents are subjects of the experiment and run as the user with no
read restriction of their own: before the sandbox, an agent that listed its
parent directory could read another run's notes, models and hypotheses, and
`ps` showed the harness's --gt-models-dir. ``sandbox_command`` wraps the agent
CLI in a bubblewrap mount namespace holding only its working tree, a scratch
directory at /tmp, a private home, and read-only system software, Python and
CLI. The integration tests run a real sandbox around a shell; they skip where
bubblewrap is not installed (``ml load system bubblewrap`` on Sherlock).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from src.runtime import agent_sandbox
from src.runtime.agent_sandbox import sandbox_command

needs_bwrap = pytest.mark.skipif(
    shutil.which("bwrap") is None, reason="bubblewrap not installed"
)


def _layout(tmp_path: Path):
    """Two runs: this agent's tree (with its own candidate dir) and another's."""
    tree = tmp_path / "agent_trees" / "run_a" / "repo"
    agent_dir = tree / "_runs" / "cell_1" / "candidate_0"
    agent_dir.mkdir(parents=True)
    (tree / "brief.md").write_text("the brief\n", encoding="utf-8")
    other = tmp_path / "agent_trees" / "run_b" / "repo"
    other.mkdir(parents=True)
    (other / "notes.md").write_text("PERIWINKLE\n", encoding="utf-8")
    return tree, agent_dir, other


def _run(tmp_path: Path, script: str):
    tree, agent_dir, other = _layout(tmp_path)
    cmd, env = sandbox_command(
        ["bash", "-c", script],
        backend="opencode",
        cwd=tree,
        writable_dirs=[agent_dir],
        agent_dir=agent_dir,
        env=dict(os.environ),
    )
    result = subprocess.run(
        cmd, env=env, cwd=tree, capture_output=True, text=True, timeout=120
    )
    return result, tree, agent_dir, other


@needs_bwrap
def test_agent_reads_and_writes_its_own_tree(tmp_path):
    result, tree, _, _ = _run(tmp_path, "cat brief.md && echo made > made.txt")
    assert result.returncode == 0, result.stderr
    assert result.stdout == "the brief\n"
    assert (tree / "made.txt").read_text() == "made\n"


@needs_bwrap
def test_another_run_does_not_exist_inside(tmp_path):
    script = f"cat {tmp_path}/agent_trees/run_b/repo/notes.md; ls {tmp_path}/agent_trees"
    result, *_ = _run(tmp_path, script)
    assert "PERIWINKLE" not in result.stdout
    assert "No such file" in result.stderr
    assert result.stdout.split() == ["run_a"]  # only the path down to its own tree


@needs_bwrap
def test_another_runs_tree_cannot_be_written(tmp_path):
    other = tmp_path / "agent_trees" / "run_b" / "repo"
    result, *_ = _run(tmp_path, f"echo x > {other}/planted.txt")
    assert result.returncode != 0
    assert not (other / "planted.txt").exists()


@needs_bwrap
def test_the_python_install_is_read_only(tmp_path):
    import sys

    result, *_ = _run(tmp_path, f"touch {sys.prefix}/planted")
    assert result.returncode != 0
    assert not Path(sys.prefix, "planted").exists()


@needs_bwrap
def test_tmp_is_the_agents_own_scratch_dir_on_disk(tmp_path):
    """Agents habitually write analysis scripts to /tmp; they land in the
    agent's scratch dir, kept with the run for inspection."""
    result, _, agent_dir, _ = _run(tmp_path, "echo 'print(1)' > /tmp/explore.py && echo $TMPDIR")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/tmp"
    assert (agent_dir / "scratch" / "explore.py").read_text() == "print(1)\n"


@needs_bwrap
def test_home_is_private(tmp_path):
    result, _, agent_dir, _ = _run(tmp_path, "echo hi > ~/.hello && ls -A ~")
    assert result.returncode == 0, result.stderr
    assert ".claude" not in result.stdout.split()  # the user's own files are absent
    assert (agent_dir / ".home" / ".hello").exists()


@needs_bwrap
def test_only_the_agents_own_processes_are_visible(tmp_path):
    """The harness's command line (--gt-models-dir ...) is not in `ps`."""
    result, *_ = _run(tmp_path, "ps -e -o comm=")
    assert result.returncode == 0, result.stderr
    assert "python" not in result.stdout  # the pytest process running this test


@needs_bwrap
def test_the_venv_python_works_inside(tmp_path):
    """The self-test command and the agents' analysis scripts need it."""
    import sys

    result, *_ = _run(tmp_path, f"{sys.executable} -c 'import numpy; print(numpy.__name__)'")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "numpy"


# ── Credentials and preconditions (no sandbox needs to run) ───────────


def _fake_bwrap(monkeypatch):
    real_which = shutil.which
    monkeypatch.setattr(
        agent_sandbox.shutil, "which",
        lambda name: "/usr/bin/bwrap" if name == "bwrap" else real_which(name),
    )


def test_missing_bubblewrap_fails_loudly(tmp_path, monkeypatch):
    monkeypatch.setattr(agent_sandbox.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="bubblewrap"):
        sandbox_command(["bash"], backend="opencode", cwd=tmp_path,
                        writable_dirs=[], agent_dir=tmp_path, env={})


def test_a_sandboxed_claude_agent_needs_the_long_lived_token(tmp_path, monkeypatch):
    """Its credentials file cannot be refreshed from inside the sandbox; the
    token from `claude setup-token` needs no file and no refresh."""
    _fake_bwrap(monkeypatch)
    with pytest.raises(RuntimeError, match="CLAUDE_CODE_OAUTH_TOKEN"):
        sandbox_command(["claude", "-p", "x"], backend="claude", cwd=tmp_path,
                        writable_dirs=[], agent_dir=tmp_path, env={"HOME": str(tmp_path)})


def test_a_sandboxed_codex_agent_gets_a_private_codex_home_with_only_its_login(
    tmp_path, monkeypatch
):
    """~/.codex holds every past session and the user's rules; the agent gets a
    fresh CODEX_HOME containing only auth.json."""
    _fake_bwrap(monkeypatch)
    user_home = tmp_path / "home"
    (user_home / ".codex").mkdir(parents=True)
    (user_home / ".codex" / "auth.json").write_text("{}", encoding="utf-8")
    agent_dir = tmp_path / "agent"
    agent_dir.mkdir()
    cmd, env = sandbox_command(
        ["bash"], backend="codex", cwd=tmp_path, writable_dirs=[],
        agent_dir=agent_dir, env={"HOME": str(user_home)},
    )
    codex_home = Path(env["CODEX_HOME"])
    assert codex_home.is_relative_to(agent_dir)
    joined = " ".join(cmd)
    assert f"--bind {user_home}/.codex/auth.json {codex_home}/auth.json" in joined
    assert f"{user_home}/.codex " not in joined + " "  # never the whole directory
