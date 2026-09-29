"""Run a loop agent inside a filesystem sandbox (bubblewrap).

A loop agent is a subject of the experiment. It runs as the user and the agent
CLIs impose no read restriction of their own, so without a sandbox an agent that
went looking could read other runs' notes, models and hypotheses, the user's
home, and — through `ps` — the harness's --gt-models-dir. ``sandbox_command``
wraps the CLI's argv in a bubblewrap mount namespace that contains only:

- read-write: the agent's working tree (and any allowed directory outside it),
  its own ``scratch/`` directory mounted at /tmp (agents habitually write their
  analysis scripts there; this keeps them on disk, with the run), and a private
  home directory;
- read-only: system software (/usr, /etc, and /share/software, where Sherlock's
  modules live), the Python venv the harness runs with and its base
  interpreter, and the agent's own CLI;
- the agent's login and nothing else of the user's configuration: Claude
  through the long-lived CLAUDE_CODE_OAUTH_TOKEN, codex through a private
  CODEX_HOME holding only auth.json, opencode through the provider key in its
  environment.

Nothing else exists inside, and a private PID namespace hides every other
process. The network is shared: the agent must reach its API, and Sherlock
caps network namespaces at 0 anyway.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

SYSTEM_DIRS = ("/usr", "/etc", "/share/software")
# On merged-/usr systems (Sherlock's el7 included) these are links into /usr.
SYSTEM_LINKS = {"/bin": "usr/bin", "/sbin": "usr/sbin", "/lib": "usr/lib", "/lib64": "usr/lib64"}
SCRATCH_NAME = "scratch"  # mounted at /tmp
HOME_NAME = ".home"  # mounted at $HOME


def sandbox_command(
    cmd: Sequence[str],
    *,
    backend: str,
    cwd: Path,
    writable_dirs: Sequence[Path],
    agent_dir: Path,
    env: Mapping[str, str],
) -> Tuple[List[str], Dict[str, str]]:
    """``cmd`` wrapped in a bubblewrap sandbox, and the environment to run it with.

    ``agent_dir`` is the agent's own directory (the one holding its log); its
    ``scratch/`` becomes /tmp and its ``.home/`` becomes $HOME.
    """
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError(
            "Sandboxed agents need bubblewrap on PATH (on Sherlock: "
            "`ml load system bubblewrap`)."
        )
    env = dict(env)
    user_home = Path(env.get("HOME") or Path.home())
    agent_dir = Path(agent_dir)
    scratch = agent_dir / SCRATCH_NAME
    private_home = agent_dir / HOME_NAME
    scratch.mkdir(parents=True, exist_ok=True)
    private_home.mkdir(parents=True, exist_ok=True)

    args = [bwrap, "--unshare-user", "--unshare-pid", "--die-with-parent"]
    args += ["--proc", "/proc", "--dev", "/dev"]
    for directory in SYSTEM_DIRS:
        args += ["--ro-bind-try", directory, directory]
    for link, target in SYSTEM_LINKS.items():
        args += ["--symlink", target, link]

    # Home and /tmp first: a working tree that itself lives under $HOME or /tmp
    # is then mounted on top of them rather than hidden by them.
    args += ["--bind", str(private_home), str(user_home)]
    args += ["--bind", str(scratch), "/tmp"]
    for directory in _outermost([Path(cwd), *map(Path, writable_dirs), agent_dir]):
        args += ["--bind", str(directory), str(directory)]

    # Read-only on top of all that.
    for source, destination in _python_install_mounts():
        args += ["--ro-bind", str(source), str(destination)]
    executable, cli_mount = _cli_install(cmd[0], backend)
    if cli_mount is not None:
        args += ["--ro-bind", str(cli_mount), str(cli_mount)]
    args += _login(backend, env, user_home, private_home)

    env["TMPDIR"] = "/tmp"
    # Caches go to the private home, removed when the agent exits, so scratch/
    # holds only the agent's own files. Sherlock exports PYTHONPYCACHEPREFIX=/tmp
    # for every user, which put a .pyc for every import into scratch/ (about
    # 2,300 files per agent). The harness's pytensor compile dir is not mounted.
    env["PYTHONPYCACHEPREFIX"] = str(user_home / ".cache" / "pycache")
    env["PYTENSOR_FLAGS"] = f"base_compiledir={user_home / '.cache' / 'pytensor'}"
    for var, default in (
        ("XDG_CACHE_HOME", ".cache"),
        ("XDG_STATE_HOME", ".local/state"),
        ("XDG_CONFIG_HOME", ".config"),
    ):
        env[var] = str(user_home / default)
    if backend != "opencode":  # opencode's own XDG_DATA_HOME is in the agent dir
        env["XDG_DATA_HOME"] = str(user_home / ".local" / "share")
    if backend == "opencode":
        # opencode's own guard answers a path outside the tree with a permission
        # prompt, and `opencode run` auto-rejects it by ending the whole session:
        # the agent stops working. The sandbox already keeps it in (with a plain
        # "No such file"), so the guard is switched off. OPENCODE_PERMISSION is
        # merged over opencode.json, whose other rules stay.
        permission = json.loads(env.get("OPENCODE_PERMISSION") or "{}")
        permission["external_directory"] = "allow"
        env["OPENCODE_PERMISSION"] = json.dumps(permission)

    args += ["--chdir", str(cwd), "--", executable, *cmd[1:]]
    return args, env


def remove_private_home(agent_dir: Path) -> None:
    """Delete a finished agent's private home; its ``scratch/`` is kept.

    The home only holds CLI state (codex installs about 200 files of stock
    skills and plugins per agent); the agent's log is its record. A login
    mounted into it (codex's auth.json) was a mount inside the namespace only:
    on the host it is an empty mount-point file, so this cannot touch the real
    one — a test runs a real sandbox to check exactly that.
    """
    home = Path(agent_dir) / HOME_NAME
    if home.exists():
        shutil.rmtree(home)


def _outermost(paths: Sequence[Path]) -> List[Path]:
    """The paths, minus any that lie inside another one (already mounted by it)."""
    resolved = [p.resolve() for p in paths]
    keep: List[Path] = []
    for path in resolved:
        if path in keep:
            continue
        if any(path.is_relative_to(other) for other in resolved if other != path):
            continue
        keep.append(path)
    return keep


def _outside_system_dirs(path: Path) -> bool:
    return not any(path.is_relative_to(d) for d in SYSTEM_DIRS)


def _python_install_mounts() -> List[Tuple[Path, Path]]:
    """(source, destination) mounts for the harness's venv and its interpreter.

    Agents run the self-test command and their own analysis scripts with this
    Python. The venv is mounted at the path the harness knows it by (the holdout
    task reaches it through an opaque link). A uv venv's bin/python links to a
    minor-version alias of the install (cpython-3.12-...) rather than the
    install itself (cpython-3.12.13-...), so the install is mounted there too.
    """
    venv, base = Path(sys.prefix), Path(sys.base_prefix)
    mounts = [(venv.resolve(), venv)]
    if base != venv:
        mounts.append((base.resolve(), base))
        venv_python = venv / "bin" / "python"
        if venv_python.is_symlink():
            linked_install = Path(os.readlink(venv_python)).parent.parent
            if linked_install.is_absolute() and linked_install != base:
                mounts.append((base.resolve(), linked_install))
    return [(src, dst) for src, dst in mounts if _outside_system_dirs(src)]


def _cli_install(name: str, backend: str) -> Tuple[str, Path | None]:
    """The CLI's absolute path and the directory (or file) to mount for it."""
    found = shutil.which(name)
    if found is None:
        raise RuntimeError(f"{name!r} (the {backend} CLI) is not on PATH.")
    if backend == "claude":
        # A single native binary, reached through a link in ~/.local/bin, which
        # the private home hides: run the binary itself.
        binary = Path(found).resolve()
        return str(binary), binary if _outside_system_dirs(binary) else None
    # codex (npm) and opencode (a module) run from <prefix>/bin/<name>; the
    # package, and codex's native binary within it, live under the prefix.
    prefix = Path(found).parent.parent
    return found, prefix if _outside_system_dirs(prefix.resolve()) else None


def _login(
    backend: str, env: Dict[str, str], user_home: Path, private_home: Path
) -> List[str]:
    """Mount (or require) the backend's credentials; adjust ``env`` in place."""
    if backend == "claude":
        # The login in ~/.claude/.credentials.json is an OAuth token the CLI
        # refreshes by rewriting the file, which a sandbox mount cannot promise
        # to allow; `claude setup-token` issues a long-lived subscription token
        # that needs no file and no refresh.
        if not env.get("CLAUDE_CODE_OAUTH_TOKEN"):
            raise RuntimeError(
                "A sandboxed Claude agent needs CLAUDE_CODE_OAUTH_TOKEN: run "
                "`claude setup-token` once and add the token to .secrets."
            )
        return []
    if backend == "codex":
        # ~/.codex also holds every past session transcript and the user's
        # rules; the agent gets a fresh CODEX_HOME with only the login in it.
        login = Path(env.get("CODEX_HOME") or user_home / ".codex") / "auth.json"
        if not login.exists():
            raise RuntimeError(f"codex is not logged in: {login} does not exist.")
        codex_home = private_home / ".codex"
        codex_home.mkdir(parents=True, exist_ok=True)
        env["CODEX_HOME"] = str(codex_home)
        return ["--bind", str(login), str(codex_home / "auth.json")]
    # opencode reads its provider key from the environment. A key stored by
    # `opencode auth login` reaches the agent's data home as a link
    # (coding_agent._link_opencode_credentials); mount what it points at.
    stored = Path(env.get("XDG_DATA_HOME", "")) / "opencode" / "auth.json"
    if stored.is_symlink():
        source = stored.resolve()
        return ["--ro-bind", str(source), str(source)]
    return []
