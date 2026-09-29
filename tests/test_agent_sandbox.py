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
    zoo = tree / "_runs" / "cell_1" / "models"
    zoo.mkdir()
    (zoo / "admitted.py").write_text("model = 1\n", encoding="utf-8")
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
def test_agent_reads_its_tree_and_writes_its_own_dir(tmp_path):
    script = "cat brief.md && echo made > _runs/cell_1/candidate_0/made.txt"
    result, _, agent_dir, _ = _run(tmp_path, script)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "the brief\n"
    assert (agent_dir / "made.txt").read_text() == "made\n"


@needs_bwrap
def test_the_rest_of_its_tree_is_read_only(tmp_path):
    """A Gemini candidate once emptied an admitted model in the shared zoo with
    a broken heredoc (`cat << 'EOF' > models/x.py`), and the cell crashed on
    it at the next admission. The shared tree can be read, not written."""
    script = (
        "cat _runs/cell_1/models/admitted.py; "
        "cat << 'EOF' > _runs/cell_1/models/admitted.py; echo x > planted.txt"
    )
    result, tree, _, _ = _run(tmp_path, script)
    assert result.stdout == "model = 1\n"
    assert "Read-only file system" in result.stderr
    assert (tree / "_runs" / "cell_1" / "models" / "admitted.py").read_text() == "model = 1\n"
    assert not (tree / "planted.txt").exists()


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
def test_scratch_holds_only_the_agents_own_files_not_python_caches(tmp_path, monkeypatch):
    """Sherlock exports PYTHONPYCACHEPREFIX=/tmp for every user, so inside the
    sandbox every import wrote a .pyc into scratch/ (about 2,300 per candidate
    in the first sandboxed smoke, 4 M inodes a sweep). Python's and pytensor's
    caches go to the private home, which is removed when the agent exits."""
    import sys

    monkeypatch.setenv("PYTHONPYCACHEPREFIX", "/tmp")
    script = f"echo 'print(1)' > /tmp/explore.py && {sys.executable} -c 'import numpy, json'"
    result, _, agent_dir, _ = _run(tmp_path, script)
    assert result.returncode == 0, result.stderr
    scratch = agent_dir / "scratch"
    # Files only: where tmp_path is under /tmp, bwrap's mount points for the
    # tree show up in scratch/ as empty directories.
    assert sorted(p.name for p in scratch.rglob("*") if p.is_file()) == ["explore.py"]


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


# ── The private home is removed when the agent exits; scratch is kept ─


def test_removing_the_private_home_keeps_scratch(tmp_path):
    from src.runtime.agent_sandbox import remove_private_home

    agent_dir = tmp_path / "candidate_0"
    (agent_dir / ".home" / ".codex" / "skills").mkdir(parents=True)
    (agent_dir / "scratch").mkdir()
    (agent_dir / "scratch" / "explore.py").write_text("print(1)\n", encoding="utf-8")
    remove_private_home(agent_dir)
    assert not (agent_dir / ".home").exists()
    assert (agent_dir / "scratch" / "explore.py").exists()


def test_removing_the_private_home_waits_out_a_busy_mount_point(tmp_path, monkeypatch):
    """On Sherlock's 3.10 kernel a file that is still a mount point in the
    exiting sandbox's namespace cannot be deleted (EBUSY) until the kernel
    finishes tearing the namespace down. A real Opus cell died on exactly that
    (the claude binary's mount point); a moment later the file was deletable."""
    import errno

    from src.runtime.agent_sandbox import remove_private_home

    agent_dir = tmp_path / "candidate_0"
    (agent_dir / ".home").mkdir(parents=True)
    real_rmtree, calls = shutil.rmtree, []

    def busy_twice(path):
        calls.append(path)
        if len(calls) <= 2:
            raise OSError(errno.EBUSY, "Device or resource busy")
        real_rmtree(path)

    monkeypatch.setattr(agent_sandbox.shutil, "rmtree", busy_twice)
    monkeypatch.setattr(agent_sandbox.time, "sleep", lambda s: None)
    remove_private_home(agent_dir)
    assert not (agent_dir / ".home").exists() and len(calls) == 3


def test_a_mount_point_that_stays_busy_fails_loudly(tmp_path, monkeypatch):
    import errno

    from src.runtime.agent_sandbox import remove_private_home

    (tmp_path / ".home").mkdir()

    def always_busy(path):
        raise OSError(errno.EBUSY, "Device or resource busy")

    monkeypatch.setattr(agent_sandbox.shutil, "rmtree", always_busy)
    monkeypatch.setattr(agent_sandbox.time, "sleep", lambda s: None)
    with pytest.raises(OSError):
        remove_private_home(tmp_path)


@needs_bwrap
def test_removing_a_codex_agents_home_never_touches_the_real_login(tmp_path):
    """The login is mounted into the private home; only the mount point, an
    empty file on the host, may be deleted."""
    from src.runtime.agent_sandbox import remove_private_home

    user_home = tmp_path / "home"
    (user_home / ".codex").mkdir(parents=True)
    login = user_home / ".codex" / "auth.json"
    login.write_text('{"tokens": "real"}', encoding="utf-8")
    tree, agent_dir, _ = _layout(tmp_path)
    cmd, env = sandbox_command(
        ["bash", "-c", 'cat "$CODEX_HOME/auth.json"'], backend="codex", cwd=tree,
        writable_dirs=[agent_dir], agent_dir=agent_dir,
        env={**os.environ, "HOME": str(user_home)},
    )
    result = subprocess.run(cmd, env=env, cwd=tree, capture_output=True, text=True, timeout=60)
    assert result.stdout == '{"tokens": "real"}', result.stderr  # the agent had its login
    remove_private_home(agent_dir)
    assert login.read_text(encoding="utf-8") == '{"tokens": "real"}'
    assert not (agent_dir / ".home").exists()


def test_a_sandboxed_opencode_agent_is_kept_in_by_the_sandbox_not_its_own_prompt(
    tmp_path, monkeypatch
):
    """opencode's own external-directory guard answers a path outside the tree
    with a permission prompt, and `opencode run` auto-rejects it by ending the
    whole session: the agent stops working. Inside the sandbox the guard is
    redundant, so it is switched off (merged over opencode.json, which keeps
    its other rules); the agent then gets bwrap's plain "No such file" and
    carries on. Verified with a real Gemini agent."""
    import json

    _fake_bwrap(monkeypatch)
    _, env = sandbox_command(
        ["bash"], backend="opencode", cwd=tmp_path, writable_dirs=[],
        agent_dir=tmp_path, env={"OPENCODE_PERMISSION": json.dumps({"bash": "allow"})},
    )
    assert json.loads(env["OPENCODE_PERMISSION"]) == {
        "bash": "allow", "external_directory": "allow",
    }


def test_the_agents_environment_carries_no_slurm_variables(tmp_path, monkeypatch):
    """SLURM_ARRAY_TASK_ID maps to the held-out ground truth through the default
    ground-truth order; no Slurm variable is any use to an agent."""
    _fake_bwrap(monkeypatch)
    _, env = sandbox_command(
        ["bash"], backend="opencode", cwd=tmp_path, writable_dirs=[], agent_dir=tmp_path,
        env={"SLURM_ARRAY_TASK_ID": "2", "SLURM_JOB_ID": "1", "PATH": "/usr/bin"},
    )
    assert not any(key.startswith("SLURM_") for key in env)
    assert env["PATH"] == "/usr/bin"
