"""Run a loop agent inside a filesystem sandbox (bubblewrap).

A loop agent is a subject of the experiment. It runs as the user and the agent
CLIs impose no read restriction of their own, so without a sandbox an agent that
went looking could read other runs' notes, models and hypotheses, the user's
home, and — through `ps` — the harness's --gt-models-dir. ``sandbox_command``
wraps the CLI's argv in a bubblewrap mount namespace that contains only:

- read-write: the agent's own directories (its candidate or critique dir, its
  run's notes), its own ``scratch/`` directory mounted at /tmp (agents
  habitually write their analysis scripts there; this keeps them on disk, with
  the run), and a private home directory;
- read-only: the rest of its working tree (the shared model zoo, the data,
  other agents' directories), system software (/usr, /etc, and
  /share/software, where Sherlock's modules live), the Python venv the harness
  runs with and its base interpreter, and the agent's own CLI;
- the agent's login and nothing else of the user's configuration: Claude
  through the long-lived CLAUDE_CODE_OAUTH_TOKEN (``subscription``) or
  ANTHROPIC_API_KEY (``api``) — whichever the run's ``CLAUDE_AUTH`` names, and
  never both (``claude_login_environment``) — codex through a private
  CODEX_HOME holding only auth.json, opencode through the provider key in its
  environment;
- an allowlisted environment (``agent_environment``): the system basics, the
  toolchain PyTensor compiles with, network settings, and its own backend's
  login and configuration — not the harness's environment, which carries
  every key in ``.secrets``.

Nothing else exists inside, and a private PID namespace hides every other
process. The network is shared: the agent must reach its API, and Sherlock
caps network namespaces at 0 anyway.
"""

from __future__ import annotations

import errno
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

SYSTEM_DIRS = ("/usr", "/etc", "/share/software")
# On merged-/usr systems (Sherlock's el7 included) these are links into /usr.
SYSTEM_LINKS = {
    "/bin": "usr/bin",
    "/sbin": "usr/sbin",
    "/lib": "usr/lib",
    "/lib64": "usr/lib64",
}
SCRATCH_NAME = "scratch"  # mounted at /tmp
HOME_NAME = ".home"  # mounted at $HOME
HOME_REMOVAL_ATTEMPTS = 6  # about 30 s in all
HOME_REMOVAL_BACKOFF_SECS = 2.0

# The environment variables a sandboxed agent gets (``agent_environment``). The
# harness's environment holds every key in .secrets (Prolific, Firebase, the
# results token, ...), the Slurm task id that maps to the held-out ground truth,
# and whatever a retry job exported (ARRAY_TASKS); the network is open, so an
# agent must not hold what it has no use for. It gets:
# - the system basics, locale and XDG directories;
AGENT_ENV_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "TERM",
        "LANG",
        "LANGUAGE",
        "TZ",
        "PWD",
        "TMPDIR",
    }
    # - the compiler toolchain PyTensor compiles models with (the gcc module's
    #   CC/CXX and its libstdc++ on LD_LIBRARY_PATH) and the thread caps;
    | {
        "CC",
        "CXX",
        "LD_LIBRARY_PATH",
        "LIBRARY_PATH",
        "CPATH",
        "C_INCLUDE_PATH",
        "CPLUS_INCLUDE_PATH",
        "PKG_CONFIG_PATH",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "XLA_FLAGS",  # JAX's thread cap (RSA models check themselves with JAX)
        "PYTENSOR_FLAGS",
        "PYTHONPYCACHEPREFIX",
    }
    # - network settings, and node's (codex is a node script);
    | {
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "NODE_EXTRA_CA_CERTS",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "NODE_PATH",
        "NODE_OPTIONS",
    }
)
AGENT_ENV_PREFIXES = ("LC_", "XDG_")
# - and its own backend's login and configuration: (names, prefixes).
BACKEND_ENV = {
    "opencode": (
        frozenset(
            {
                "GOOGLE_API_KEY",
                "GEMINI_API_KEY",
                "GOOGLE_GENERATIVE_AI_API_KEY",
                "ANTHROPIC_API_KEY",
                "OPENAI_API_KEY",
                "OPENROUTER_API_KEY",
            }
        ),
        ("OPENCODE_",),
    ),
    "claude": (frozenset({"DISABLE_AUTOUPDATER"}), ("CLAUDE_CODE_", "ANTHROPIC_")),
    "codex": (frozenset({"OPENAI_API_KEY"}), ("CODEX_",)),
}


# How Claude agents are billed. The mode is stated per run (config key
# agent.claude_auth, CLI flag --claude-auth, or CLAUDE_AUTH in the job
# scripts) and reaches this module as CLAUDE_AUTH in the harness's
# environment; there is no default, because the credential decides who pays:
# the CLI prefers ANTHROPIC_API_KEY over the subscription's OAuth token, so a
# stray key in .secrets used to move a subscription run onto API billing.
CLAUDE_AUTH_ENV = "CLAUDE_AUTH"
CLAUDE_AUTH_CREDENTIAL = {
    "subscription": "CLAUDE_CODE_OAUTH_TOKEN",  # `claude setup-token`
    "api": "ANTHROPIC_API_KEY",
}


def _claude_auth_mode(mode: str | None) -> str:
    if mode is None or mode == "":
        raise RuntimeError(
            "Claude agents need a stated billing mode: set agent.claude_auth in the "
            "config, pass --claude-auth, or export CLAUDE_AUTH for the job scripts "
            f"(one of {sorted(CLAUDE_AUTH_CREDENTIAL)})."
        )
    if mode not in CLAUDE_AUTH_CREDENTIAL:
        raise ValueError(
            f"unknown Claude billing mode {mode!r} (expected one of "
            f"{sorted(CLAUDE_AUTH_CREDENTIAL)})."
        )
    return mode


def require_claude_auth(
    backend: str, mode: str | None, env: Dict[str, str] | None = None
) -> str | None:
    """Check, before any agent starts, that a ``claude`` run states how its
    agents are billed and holds that mode's credential; return the mode.

    ``mode`` is the run's setting (CLI flag, then config), else ``env``'s
    ``CLAUDE_AUTH``. The mode is written back to ``env`` (default: this
    process's environment), where the sandbox reads it for every agent. A
    backend other than ``claude`` needs no mode and returns None.
    """
    env = os.environ if env is None else env
    if backend != "claude":
        if mode is not None:
            _claude_auth_mode(mode)
        return None
    mode = _claude_auth_mode(mode or env.get(CLAUDE_AUTH_ENV))
    credential = CLAUDE_AUTH_CREDENTIAL[mode]
    if not env.get(credential):
        raise RuntimeError(
            f"Claude agents are billed by {mode!r} but {credential} is not set; add it "
            "to .secrets (the job scripts export every key there)."
        )
    env[CLAUDE_AUTH_ENV] = mode
    return mode


def claude_login_environment(env: Dict[str, str], mode: str | None) -> None:
    """Give a Claude agent's environment ``mode``'s credential and no other.

    ``subscription``: CLAUDE_CODE_OAUTH_TOKEN and no ``ANTHROPIC_*`` variable
    (an API key or base URL would take over the login); ``api``:
    ANTHROPIC_API_KEY and no OAuth token. Raises when the mode is missing or
    its credential is not in ``env``.
    """
    mode = _claude_auth_mode(mode)
    credential = CLAUDE_AUTH_CREDENTIAL[mode]
    if not env.get(credential):
        raise RuntimeError(
            f"A sandboxed Claude agent billed by {mode!r} needs {credential}"
            + (
                ": run `claude setup-token` once and add the token to .secrets."
                if mode == "subscription"
                else " in .secrets."
            )
        )
    for key in list(env):
        if mode == "subscription" and key.startswith("ANTHROPIC_"):
            del env[key]
        if mode == "api" and key == CLAUDE_AUTH_CREDENTIAL["subscription"]:
            del env[key]


def agent_environment(env: Mapping[str, str], backend: str) -> Dict[str, str]:
    """The allowlisted part of ``env`` a sandboxed ``backend`` agent runs with.

    See ``AGENT_ENV_NAMES`` and ``BACKEND_ENV``. Everything else is withheld:
    every other ``.secrets`` key, every ``SLURM_*`` variable
    (``SLURM_ARRAY_TASK_ID`` maps to the held-out ground truth through the
    default ground-truth order), the sweep's own variables.
    """
    if backend not in BACKEND_ENV:
        raise ValueError(
            f"No environment allowlist for backend {backend!r}; add it to BACKEND_ENV."
        )
    names, prefixes = BACKEND_ENV[backend]
    return {
        key: value
        for key, value in env.items()
        if key in AGENT_ENV_NAMES
        or key in names
        or key.startswith(AGENT_ENV_PREFIXES + prefixes)
    }


def sandbox_command(
    cmd: Sequence[str],
    *,
    backend: str,
    cwd: Path,
    writable_dirs: Sequence[Path],
    agent_dir: Path,
    env: Mapping[str, str],
    readable_dirs: Sequence[Path] = (),
) -> Tuple[List[str], Dict[str, str]]:
    """``cmd`` wrapped in a bubblewrap sandbox, and the environment to run it with.

    ``cwd`` (the agent's tree) and ``readable_dirs`` are mounted read-only;
    ``writable_dirs`` and ``agent_dir`` (the agent's own directory, the one
    holding its log) read-write. The agent dir's ``scratch/`` becomes /tmp and
    its ``.home/`` becomes $HOME.
    """
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError(
            "Sandboxed agents need bubblewrap on PATH (on Sherlock: "
            "`ml load system bubblewrap`)."
        )
    claude_auth = env.get(CLAUDE_AUTH_ENV)
    env = agent_environment(env, backend)
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
    # The tree is read-only and the agent's own directories are writable on
    # top of it: a Gemini candidate once emptied an admitted model in the
    # shared zoo with a broken heredoc, and the cell crashed on it.
    for directory in _outermost([Path(cwd), *map(Path, readable_dirs)]):
        args += ["--ro-bind", str(directory), str(directory)]
    for directory in _outermost([*map(Path, writable_dirs), agent_dir]):
        directory.mkdir(parents=True, exist_ok=True)
        args += ["--bind", str(directory), str(directory)]

    # Read-only on top of all that.
    for source, destination in _python_install_mounts():
        args += ["--ro-bind", str(source), str(destination)]
    executable, cli_mount = _cli_install(cmd[0], backend)
    if cli_mount is not None:
        args += ["--ro-bind", str(cli_mount), str(cli_mount)]
    args += _login(backend, env, user_home, private_home, claude_auth)

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
    # A login or CLI binary mounted into the home leaves a mount point there. On
    # Sherlock's 3.10 kernel it cannot be deleted (EBUSY) until the kernel has
    # finished tearing down the exited sandbox's namespace, a moment after the
    # process is gone; a real cell died on that. Wait it out, then fail loudly.
    for attempt in range(HOME_REMOVAL_ATTEMPTS):
        if not home.exists():
            return
        try:
            shutil.rmtree(home)
            return
        except OSError as error:
            if error.errno != errno.EBUSY or attempt == HOME_REMOVAL_ATTEMPTS - 1:
                raise
            time.sleep(HOME_REMOVAL_BACKOFF_SECS * (attempt + 1))


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
    backend: str,
    env: Dict[str, str],
    user_home: Path,
    private_home: Path,
    claude_auth: str | None = None,
) -> List[str]:
    """Mount (or require) the backend's credentials; adjust ``env`` in place."""
    if backend == "claude":
        # The login in ~/.claude/.credentials.json is an OAuth token the CLI
        # refreshes by rewriting the file, which a sandbox mount cannot promise
        # to allow; `claude setup-token` issues a long-lived subscription token
        # that needs no file and no refresh. An API key needs nothing either.
        claude_login_environment(env, claude_auth)
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
