"""Backend-agnostic launcher for coding agents (Claude Code or opencode).

Both pipeline loops spawn a coding-agent CLI as a subprocess, stream its
output to a log, and read back a (success, result) pair. The only thing that
differs between Claude Code and opencode is how the command line is built and
how the streamed output is interpreted. This module owns that difference so the
call sites stay backend-neutral.

Backend selection: an explicit argument wins, else the ``CODING_AGENT``
environment variable, else ``"opencode"`` (the default). Model names are
per-backend: each backend has its own default and any ``model`` argument is
passed through verbatim (opencode uses ``provider/model`` — defaults to Gemini;
Claude uses ``claude-sonnet-4-6``; Codex (``codex exec --json``) uses
``gpt-5.6-sol``).

Both backends stream JSON events (Claude via ``--output-format stream-json``,
opencode via ``--format json``), and both report token usage in that stream —
Claude in the terminal ``result`` event, opencode in per-step ``step_finish``
events. Every run records its usage to :mod:`src.runtime.token_usage` under
the caller's ``usage_label`` so a pipeline run can account for its total
token spend.

The launch contract (P34). ``opencode run`` roots its session — the directory
inside which paths are "internal" and whose ``opencode.json`` it loads — at
``process.env.PWD ?? process.cwd()``. ``subprocess.Popen(cwd=...)`` changes the
child's directory but leaves the inherited ``PWD`` alone, so an agent launched
from a harness whose shell sat elsewhere got a session rooted *there*, and its
own working directory was "external": asked for, and auto-rejected by the
non-interactive ``run`` command (104 of 360 candidate slots in the 2026-09
sweep ended as "no candidate.py written" that way). The launcher therefore

* pins ``PWD`` to the real cwd for every backend (:func:`child_environment`);
* for opencode, gives each agent a private ``XDG_DATA_HOME`` beside its log so
  concurrent agents never contend on opencode's shared sqlite store;
* for opencode, writes a grant for every ``allowed_dirs`` entry outside the cwd
  into the cwd's ``opencode.json`` (:func:`ensure_opencode_external_grants`);
* scans the agent's log afterwards and raises
  :class:`AgentPermissionDenied` on any auto-rejected permission — a denial is
  a misconfigured launch, never one unlucky candidate.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

from src.runtime import token_usage
from src.runtime.agent_sandbox import remove_private_home, sandbox_command

DEFAULT_BACKEND = "opencode"
_DEFAULT_MODEL = {
    "claude": "claude-sonnet-4-6",
    "opencode": "google/gemini-3.1-pro-preview",
    "codex": "gpt-5.6-sol",
}

# opencode >= 1.17 keeps its sessions in one shared sqlite database. When
# several opencode instances start at once (parallel candidate agents), the
# late arrivals can die instantly with "database is locked" during session
# creation. That is transient — retry with backoff instead of losing the
# agent's whole task. Other failures are NOT retried.
# Linux caps one argv string at 128 KiB (MAX_ARG_STRLEN). Prompts above this
# are delivered on stdin instead; codex always reads its prompt from stdin
# (its `-` argument), claude and opencode only when the prompt is long
# (`opencode run` with no message argument reads piped stdin as its message).
STDIN_PROMPT_THRESHOLD = 100_000

OPENCODE_LOCK_RETRIES = 3
OPENCODE_LOCK_BACKOFF_SECS = 2.0
_OPENCODE_LOCK_SIGNATURE = "database is locked"

# `opencode run` prints this (outside its JSON stream) whenever it refuses a
# permission it would otherwise have asked for; the tool call then fails with
# "The user rejected permission to use this specific tool call."
PERMISSION_DENIAL_SIGNATURE = "auto-rejecting"
# Name of the per-agent XDG_DATA_HOME, created beside the agent's log.
AGENT_DATA_HOME_NAME = ".xdg_data"
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
# Serialises read-modify-write of one opencode.json across the threads that
# spawn candidate agents in parallel (one process per run; cross-process
# writers of the same cwd are not expected).
_GRANTS_LOCK = threading.Lock()


class AgentPermissionDenied(RuntimeError):
    """The coding agent was refused access to its own directories.

    ``opencode run`` is non-interactive: it auto-rejects every permission it
    would have asked for, after which the agent simply cannot touch the path.
    Refused its own working tree or an allowed directory, it cannot do its job:
    that is a systemic misconfiguration of the launch (a session rooted in the
    wrong directory, a missing grant), never bad luck for one candidate, so the
    launcher raises it instead of letting the caller record an empty slot.
    """


# "! permission requested: external_directory (<dir>/*); auto-rejecting"
_EXTERNAL_DIRECTORY_DENIAL = re.compile(r"external_directory \((.+?)/?\*?\); auto-rejecting")


def check_for_permission_denials(log_path: Path, own_dirs: Sequence[Path]) -> list[str]:
    """Check the agent's log for refused permissions; return the refused outside dirs.

    A refusal of a directory outside ``own_dirs`` (the agent's working tree and
    allowed directories) is the agent reaching somewhere it may not go and being
    stopped: it is returned, for the caller to report, and the agent carries on.
    Any other refusal — of its own directories, or of a permission that is not
    a directory — means the launch is misconfigured and raises
    :class:`AgentPermissionDenied`, with the offending lines (ANSI colour
    stripped) and the log path in the message.
    """
    own = [Path(d).resolve() for d in own_dirs]
    text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    refused_outside: list[str] = []
    denials: list[str] = []
    for line in text.splitlines():
        if PERMISSION_DENIAL_SIGNATURE not in line:
            continue
        line = _ANSI_ESCAPE.sub("", line).strip()
        match = _EXTERNAL_DIRECTORY_DENIAL.search(line)
        if match and not any(Path(match.group(1)).resolve().is_relative_to(d) for d in own):
            refused_outside.append(match.group(1))
        else:
            denials.append(line)
    if not denials:
        return refused_outside
    shown = "\n".join(f"  {line}" for line in denials[:5])
    raise AgentPermissionDenied(
        f"the coding agent was denied {len(denials)} permission request(s); this "
        f"is a misconfigured launch (wrong session directory or missing grant), "
        f"not a per-candidate failure. See {log_path}:\n{shown}"
    )


def agent_data_home(log_path: Path) -> Path:
    """The private ``XDG_DATA_HOME`` of the agent whose log is ``log_path``.

    The directory holding an agent's log is that agent's own directory (its
    candidate, critique or stage dir), so its opencode store sits beside the
    log and no two concurrent agents share one sqlite database.
    """
    return Path(log_path).parent / AGENT_DATA_HOME_NAME


def _link_opencode_credentials(inherited_data_home: Path, private_data_home: Path) -> None:
    """Make credentials stored under the inherited data home visible to the agent.

    opencode reads ``$XDG_DATA_HOME/opencode/auth.json`` (else
    ``~/.local/share/opencode/auth.json``). Provider keys passed through the
    environment need nothing; keys stored by ``opencode auth login`` would be
    lost behind the private data home, so they are linked (not copied — the
    agent tree gets archived) into it.
    """
    source = inherited_data_home / "opencode" / "auth.json"
    if not source.exists():
        return
    target = private_data_home / "opencode" / "auth.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink() or target.exists():
        if target.resolve() == source.resolve():
            return
        raise RuntimeError(
            f"{target} exists and is not a link to the inherited credentials {source}"
        )
    target.symlink_to(source)


def child_environment(
    *,
    backend: str,
    cwd: Path,
    log_path: Path,
    env: Optional[dict],
    stock: bool = False,
    memory_dir: Optional[Path] = None,
) -> Dict[str, str]:
    """The environment the agent subprocess runs with.

    Starts from ``env`` (or this process's environment), then pins ``PWD`` to
    the resolved ``cwd`` — ``Popen(cwd=...)`` would otherwise leave the stale
    inherited value, which ``opencode run`` prefers over the real working
    directory when rooting its session. For opencode the agent also gets its
    own ``XDG_DATA_HOME`` (see :func:`agent_data_home`), with any credentials
    from the inherited one linked in.
    """
    child = dict(os.environ if env is None else env)
    child["PWD"] = str(Path(cwd).resolve())
    if backend == "claude" and stock and memory_dir is None:
        child.update(STOCK_CLAUDE_ENV)
    if backend == "opencode":
        inherited = Path(child.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
        private = agent_data_home(log_path)
        private.mkdir(parents=True, exist_ok=True)
        _link_opencode_credentials(inherited, private)
        child["XDG_DATA_HOME"] = str(private)
    return child


def ensure_opencode_external_grants(cwd: Path, dirs: Sequence[Path]) -> list[str]:
    """Grant opencode access to every directory in ``dirs`` that lies outside ``cwd``.

    opencode treats a path outside its session directory (the cwd, once
    ``PWD`` is pinned) as external and asks an ``external_directory``
    permission for ``<dir>/*`` — which ``opencode run`` auto-rejects — unless
    the session's ``opencode.json`` allows it. Directories inside the cwd need
    nothing and add nothing. For each outside directory the cwd's
    ``opencode.json`` gains ``<dir>/**`` and ``<dir>/*`` set to ``allow``;
    everything already in the file is preserved. Returns the patterns added.

    Raises ``FileNotFoundError`` when a grant is needed but the cwd has no
    ``opencode.json`` (the agent would run without the repo's permission
    config at all), and ``ValueError`` when a needed pattern is already
    present with a different action — overriding an explicit deny silently
    would hide a deliberate decision.
    """
    root = Path(cwd).resolve()
    external = []
    for directory in dirs:
        resolved = Path(directory).resolve()
        if resolved == root or root in resolved.parents:
            continue
        external.append(resolved)
    if not external:
        return []
    config_path = root / "opencode.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} does not exist, so opencode cannot be granted access to "
            f"{', '.join(str(d) for d in external)}; the agent's cwd must carry the "
            f"repo's opencode.json"
        )
    with _GRANTS_LOCK:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        permission = config.setdefault("permission", {})
        grants = permission.setdefault("external_directory", {})
        if not isinstance(grants, dict):
            raise ValueError(
                f"{config_path}: permission.external_directory must map globs to "
                f"actions, got {grants!r}"
            )
        added: list[str] = []
        for directory in external:
            for pattern in (f"{directory}/**", f"{directory}/*"):
                action = grants.get(pattern)
                if action == "allow":
                    continue
                if action is not None:
                    raise ValueError(
                        f"{config_path}: {pattern!r} is already {action!r}; refusing "
                        f"to override an explicit permission"
                    )
                grants[pattern] = "allow"
                added.append(pattern)
        if added:
            # Atomic replace: an opencode process starting concurrently must
            # never read a half-written config.
            tmp_path = config_path.with_name("opencode.json.tmp")
            tmp_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp_path, config_path)
    return added


def select_backend(explicit: Optional[str]) -> str:
    """Resolve the coding-agent backend: explicit arg, then env, then default."""
    backend = explicit or os.environ.get("CODING_AGENT") or DEFAULT_BACKEND
    if backend not in _DEFAULT_MODEL:
        raise ValueError(
            f"unknown coding-agent backend: {backend!r} "
            f"(expected one of {sorted(_DEFAULT_MODEL)})"
        )
    return backend


def prompt_via_stdin(backend: str, prompt: str) -> bool:
    """Whether ``prompt`` is delivered on stdin rather than as an argument."""
    if backend == "codex":
        return True
    return len(prompt.encode("utf-8")) > STDIN_PROMPT_THRESHOLD


# A *stock* Claude agent runs with none of the user's personal configuration:
# no ~/.claude settings or CLAUDE.md, no plugins, no MCP servers (the user's
# claude.ai Gmail, Drive and Calendar connectors included), and nothing written
# back to ~/.claude. The loop's candidate and critique agents are subjects of
# the experiment and run this way; the user's own agents (campaign driver,
# review panel) do not. --bare would also drop all of it, but it refuses OAuth,
# which would move the agents off the subscription onto per-token API billing.
STOCK_CLAUDE_ARGS = (
    "--setting-sources", "project,local",  # skips user settings and ~/.claude/CLAUDE.md
    "--strict-mcp-config",                 # no MCP servers unless --mcp-config names them
    "--no-session-persistence",            # no session transcripts in ~/.claude/projects
)
# Auto-memory has no flag; this variable removes it. A stock agent keeps it only
# when given a memory_dir: its notes then live in that directory (the loop
# passes one per run) instead of the user's ~/.claude, where every run sharing
# a working directory would share them.
STOCK_CLAUDE_ENV = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}


def build_command(
    backend: str,
    *,
    prompt: str,
    allowed_dirs: list[Path],
    model: Optional[str],
    extra_args: Sequence[str] = (),
    stock: bool = False,
    memory_dir: Optional[Path] = None,
) -> list[str]:
    """Build the CLI argv for the given backend.

    The prompt is the final element so callers can locate it — or, when
    :func:`prompt_via_stdin` says so, the final element is codex's ``-``
    marker / absent for claude and opencode, and the caller writes the prompt
    to stdin.
    opencode has no ``--add-dir`` equivalent (it operates on the working
    directory), so ``allowed_dirs`` reaches the command line only for Claude
    Code; :func:`run_coding_agent` honours it for opencode by granting the
    directories in the cwd's ``opencode.json``.
    ``extra_args`` are backend CLI flags appended verbatim before the prompt
    (e.g. Claude's ``--max-turns`` / ``--max-budget-usd`` / ``--disallowedTools``
    for a long-running supervisor session). ``stock`` runs Claude without the
    user's personal configuration (see ``STOCK_CLAUDE_ARGS``); codex and
    opencode have no user-level instruction files here, so it adds nothing to
    their command lines. ``memory_dir`` is where a Claude agent keeps its
    auto-memory (notes for later sessions); codex and opencode have no
    equivalent.
    """
    if backend not in _DEFAULT_MODEL:
        raise ValueError(f"unknown coding-agent backend: {backend!r}")
    model = model or _DEFAULT_MODEL[backend]
    if backend == "claude":
        cmd = [
            "claude",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]
        for d in allowed_dirs:
            cmd += ["--add-dir", str(d)]
        if stock:
            cmd += STOCK_CLAUDE_ARGS
        if memory_dir is not None:
            cmd += ["--settings", json.dumps({"autoMemoryDirectory": str(memory_dir)})]
        cmd += ["--model", model, *extra_args, "-p"]
        if not prompt_via_stdin(backend, prompt):
            cmd.append(prompt)
        return cmd
    if backend == "opencode":
        cmd = ["opencode", "run", "--format", "json", "-m", model, *extra_args]
        if not prompt_via_stdin(backend, prompt):
            cmd.append(prompt)
        return cmd
    if backend == "codex":
        # `codex exec` never prompts; the sandbox flag is its only permission
        # knob. --skip-git-repo-check lets it run outside a repository. The
        # prompt goes on stdin (`-`): briefs with an inlined evidence pack are
        # far over the argv limit.
        return [
            "codex", "exec", "--model", model, "--sandbox", "danger-full-access",
            "--skip-git-repo-check", "--json", *extra_args, "-",
        ]
    # Reachable only if _DEFAULT_MODEL gains a backend without a branch here.
    # Fail loudly rather than returning None into subprocess.Popen.
    raise ValueError(f"no command builder for coding-agent backend: {backend!r}")


def _summarise_claude_event(event: dict) -> Optional[str]:
    """One-line human summary of a Claude stream-json event, or None to skip."""
    t = event.get("type")
    if t == "assistant":
        parts = event.get("message", {}).get("content", [])
        lines = []
        for part in parts if isinstance(parts, list) else []:
            if part.get("type") == "tool_use":
                name = part.get("name", "?")
                inp = part.get("input", {})
                detail = ""
                for key in ("command", "file_path", "pattern", "path", "query"):
                    if key in inp:
                        val = str(inp[key])
                        detail = f" {val[:120]}" if len(val) > 120 else f" {val}"
                        break
                lines.append(f"  → {name}{detail}")
            elif part.get("type") == "text":
                text = part.get("text", "").strip()
                if text:
                    lines.append(f"  … {text.splitlines()[0][:120]}")
        return "\n".join(lines) if lines else None
    if t == "result":
        subtype = event.get("subtype", "")
        cost = event.get("total_cost_usd", event.get("cost_usd"))
        cost_str = f"  cost=${cost:.4f}" if cost is not None else ""
        turns = event.get("num_turns", "?")
        result_text = str(event.get("result", ""))[:200]
        return f"  [result] {subtype}{cost_str}  turns={turns}\n  {result_text}"
    return None


def _summarise_codex_event(event: dict) -> Optional[str]:
    """One-line human summary of a ``codex exec --json`` event, or None."""
    t = event.get("type")
    if t == "item.completed":
        item = event.get("item", {})
        kind = item.get("type")
        if kind == "command_execution":
            return f"  → {str(item.get('command', ''))[:120]}"
        if kind == "agent_message":
            text = str(item.get("text", "")).strip()
            return f"  … {text.splitlines()[0][:120]}" if text else None
        if kind in ("file_change", "mcp_tool_call", "web_search"):
            return f"  → {kind}"
        return None
    if t == "turn.completed":
        usage = event.get("usage", {})
        return f"  [turn] tokens in={usage.get('input_tokens', '?')} out={usage.get('output_tokens', '?')}"
    if t == "error":
        return f"  [error] {str(event.get('message', ''))[:200]}"
    return None


def _summarise_opencode_event(event: dict) -> Optional[str]:
    """One-line human summary of an opencode ``--format json`` event, or None."""
    t = event.get("type")
    part = event.get("part", {})
    if t == "text":
        text = str(part.get("text", "")).strip()
        return f"  … {text.splitlines()[0][:120]}" if text else None
    if t == "tool":
        name = part.get("tool", part.get("name", "?"))
        return f"  → {name}"
    if t == "step_finish":
        tokens = part.get("tokens", {})
        cost = part.get("cost")
        cost_str = f"  cost=${cost:.4f}" if cost is not None else ""
        return f"  [step] tokens={tokens.get('total', '?')}{cost_str}"
    return None


class _ClaudeStream:
    """Interprets Claude Code stream-json events: result text, success, usage.

    Usage comes from the terminal ``result`` event (cumulative for the whole
    run). If the run dies before emitting it (timeout, crash), fall back to
    summing the per-call ``usage`` of the assistant events seen so far, so the
    tokens already paid for still get counted.
    """

    def __init__(self) -> None:
        self.final_result = ""
        self.success = False
        self._result_usage: Optional[Dict[str, Any]] = None
        self._result_cost: Optional[float] = None
        self._assistant_usages: list[Dict[str, Any]] = []

    def feed(self, event: dict) -> None:
        t = event.get("type")
        if t == "assistant":
            usage = event.get("message", {}).get("usage")
            if isinstance(usage, dict):
                self._assistant_usages.append(usage)
        elif t == "result":
            self.final_result = str(event.get("result", ""))
            self.success = event.get("subtype") == "success"
            usage = event.get("usage")
            if isinstance(usage, dict):
                self._result_usage = usage
            self._result_cost = event.get("total_cost_usd", event.get("cost_usd"))

    def usage_fields(self) -> Dict[str, Any]:
        if self._result_usage is not None:
            usages = [self._result_usage]
        elif self._assistant_usages:
            usages = self._assistant_usages
        else:
            return {"usage_missing": True}
        # Anthropic counts reasoning inside output_tokens and cached prompt
        # tokens outside input_tokens, so the components are already disjoint.
        return {
            "input_tokens": sum(int(u.get("input_tokens", 0)) for u in usages),
            "output_tokens": sum(int(u.get("output_tokens", 0)) for u in usages),
            "cache_write_tokens": sum(
                int(u.get("cache_creation_input_tokens", 0)) for u in usages
            ),
            "cache_read_tokens": sum(
                int(u.get("cache_read_input_tokens", 0)) for u in usages
            ),
            "cost_usd": self._result_cost,
        }


class _CodexStream:
    """Interprets ``codex exec --json`` events: final message and usage.

    The result text is the last ``agent_message`` item; usage is summed over
    ``turn.completed`` events (Codex reports cached input inside
    ``input_tokens``, so it is split back out to keep the components disjoint).
    """

    def __init__(self) -> None:
        self._messages: list[str] = []
        self._token_sums = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
        self._saw_usage = False
        self.error: Optional[str] = None

    def feed(self, event: dict) -> None:
        t = event.get("type")
        if t == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                text = str(item.get("text", ""))
                if text.strip():
                    self._messages.append(text)
        elif t == "turn.completed":
            usage = event.get("usage")
            if isinstance(usage, dict):
                self._saw_usage = True
                cached = int(usage.get("cached_input_tokens", 0))
                self._token_sums["input_tokens"] += int(usage.get("input_tokens", 0)) - cached
                self._token_sums["cache_read_tokens"] += cached
                self._token_sums["output_tokens"] += int(usage.get("output_tokens", 0))
                self._token_sums["reasoning_tokens"] += int(
                    usage.get("reasoning_output_tokens", 0)
                )
        elif t == "error":
            self.error = str(event.get("message", ""))

    def result_text(self) -> str:
        return self._messages[-1].strip() if self._messages else ""

    def usage_fields(self) -> Dict[str, Any]:
        if not self._saw_usage:
            return {"usage_missing": True}
        return {**self._token_sums, "cost_usd": None}


class _OpencodeStream:
    """Interprets opencode ``--format json`` events: result text and usage.

    Each ``step_finish`` event reports that step's tokens (disjoint components:
    input / output / reasoning / cache); a run's usage is their sum. The result
    text is the concatenation of the ``text`` parts.
    """

    def __init__(self) -> None:
        self._texts: list[str] = []
        self._token_sums = {
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
        self._cost = 0.0
        self._saw_usage = False

    def feed(self, event: dict) -> None:
        t = event.get("type")
        part = event.get("part", {})
        if t == "text":
            text = str(part.get("text", ""))
            if text.strip():
                self._texts.append(text)
        elif t == "step_finish":
            tokens = part.get("tokens")
            if isinstance(tokens, dict):
                self._saw_usage = True
                cache = tokens.get("cache", {})
                self._token_sums["input_tokens"] += int(tokens.get("input", 0))
                self._token_sums["output_tokens"] += int(tokens.get("output", 0))
                self._token_sums["reasoning_tokens"] += int(
                    tokens.get("reasoning", 0)
                )
                self._token_sums["cache_read_tokens"] += int(cache.get("read", 0))
                self._token_sums["cache_write_tokens"] += int(cache.get("write", 0))
            cost = part.get("cost")
            if cost is not None:
                self._cost += float(cost)

    def result_text(self) -> str:
        return "\n".join(self._texts).strip()

    def usage_fields(self) -> Dict[str, Any]:
        if not self._saw_usage:
            return {"usage_missing": True}
        return {**self._token_sums, "cost_usd": self._cost}


def run_coding_agent(
    prompt: str,
    *,
    cwd: Path,
    log_path: Path,
    allowed_dirs: Optional[list[Path]] = None,
    model: Optional[str] = None,
    timeout_secs: int = 900,
    backend: Optional[str] = None,
    env: Optional[dict] = None,
    on_summary: Optional[Callable[[str], None]] = print,
    usage_label: str = "coding_agent",
    extra_args: Sequence[str] = (),
    stock: bool = False,
    memory_dir: Optional[Path] = None,
    sandbox: bool = False,
    writable_dirs: Optional[list[Path]] = None,
) -> tuple[bool, str]:
    """Spawn the selected coding agent, stream output to ``log_path``.

    ``stock=True`` runs a Claude agent with none of the user's personal
    configuration (``STOCK_CLAUDE_ARGS``, ``STOCK_CLAUDE_ENV``);
    ``memory_dir`` gives it an auto-memory there instead of none.
    ``sandbox=True`` runs the CLI inside a bubblewrap filesystem sandbox that
    holds only its working tree and ``allowed_dirs`` (read-only), the
    ``writable_dirs`` and ``memory_dir`` (read-write), a scratch dir at /tmp
    and a private home (see :mod:`src.runtime.agent_sandbox`).

    Returns ``(success, result_text)``. For Claude, success and the final
    result come from the terminal ``result`` stream-json event; for opencode
    success is a zero exit code and the result is the text parts of its JSON
    event stream (falling back to raw output if no events parsed). On timeout
    returns ``(False, <message>)``.

    An opencode start that dies with "database is locked" (concurrent agents
    contending for opencode's shared sqlite store) is retried up to
    ``OPENCODE_LOCK_RETRIES`` times with backoff; any other failure is final.

    Whatever usage the stream reported is recorded to
    :mod:`src.runtime.token_usage` under ``usage_label`` — also on timeout or
    failure, since those tokens were spent all the same. One logical call
    records exactly one usage entry, from the attempt that ran.

    The child runs with ``PWD`` pinned to ``cwd`` and, for opencode, a private
    ``XDG_DATA_HOME`` and grants for every ``allowed_dirs`` entry outside the
    cwd (see the module docstring). Afterwards the log is scanned and an
    auto-rejected permission raises :class:`AgentPermissionDenied` — after the
    usage was recorded, since those tokens were spent too.
    """
    backend = select_backend(backend)
    model = model or _DEFAULT_MODEL[backend]
    cmd = build_command(
        backend,
        prompt=prompt,
        allowed_dirs=list(allowed_dirs or []),
        model=model,
        extra_args=extra_args,
        stock=stock,
        memory_dir=memory_dir,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    child_env = child_environment(
        backend=backend, cwd=cwd, log_path=log_path, env=env, stock=stock,
        memory_dir=memory_dir,
    )
    if sandbox:
        if writable_dirs is None:
            raise ValueError(
                "A sandboxed agent needs writable_dirs: the directories it may "
                "write (the rest of its tree is read-only)."
            )
        cmd, child_env = sandbox_command(
            cmd,
            backend=backend,
            cwd=cwd,
            readable_dirs=list(allowed_dirs or []),
            writable_dirs=[*writable_dirs, *([memory_dir] if memory_dir else [])],
            agent_dir=log_path.parent,
            env=child_env,
        )
    if backend == "opencode":
        granted = ensure_opencode_external_grants(cwd, list(allowed_dirs or []))
        if granted and on_summary:
            on_summary(
                f"  [oc] granted external_directory in {Path(cwd) / 'opencode.json'}: "
                + ", ".join(granted)
            )

    stdin_text = prompt if prompt_via_stdin(backend, prompt) else None
    for attempt in range(1 + OPENCODE_LOCK_RETRIES):
        outcome = _run_agent_once(
            cmd,
            backend=backend,
            cwd=cwd,
            log_path=log_path,
            timeout_secs=timeout_secs,
            env=child_env,
            on_summary=on_summary,
            log_mode="w" if attempt == 0 else "a",
            stdin_text=stdin_text,
        )
        stream, captured, timed_out, returncode = outcome
        raw_output = "".join(captured)
        lock_hit = (
            backend == "opencode"
            and not timed_out
            and returncode != 0
            and _OPENCODE_LOCK_SIGNATURE in raw_output
        )
        if lock_hit and attempt < OPENCODE_LOCK_RETRIES:
            delay = OPENCODE_LOCK_BACKOFF_SECS * (attempt + 1)
            message = (
                f"  [oc] opencode session store locked (attempt {attempt + 1}); "
                f"retrying in {delay:.0f}s"
            )
            if on_summary:
                on_summary(message)
            time.sleep(delay)
            continue
        break

    usage = stream.usage_fields()
    token_usage.record_usage(
        source=usage_label, backend=backend, model=model, **usage
    )
    if usage.get("usage_missing") and on_summary:
        on_summary(
            f"  [tokens] WARNING: {backend} run for {usage_label!r} reported no "
            f"token usage; this run's spend is uncounted"
        )

    if sandbox:
        remove_private_home(log_path.parent)

    # A refusal of the agent's own directories is a misconfigured launch: raise,
    # whatever else the run did (including timing out after it). A refusal of a
    # directory outside them is the agent being kept in: report it, carry on.
    refused_outside = check_for_permission_denials(
        log_path, own_dirs=[cwd, *(allowed_dirs or [])]
    )
    if refused_outside and on_summary:
        on_summary(
            f"  [oc] refused access outside the agent's directories (agent carried on): "
            + ", ".join(refused_outside)
        )

    if timed_out:
        return False, f"coding agent ({backend}) timed out after {timeout_secs}s"

    if backend == "claude":
        return stream.success, stream.final_result
    if backend == "codex":
        # A non-zero exit or an error event is a failure; the error text is
        # the result so callers (and the session-limit detector) can read it.
        if stream.error and returncode != 0:
            return False, stream.error
        return returncode == 0, stream.result_text() or raw_output.strip()
    success = returncode == 0
    final_result = stream.result_text() or raw_output.strip()
    return success, final_result


def _run_agent_once(
    cmd: list[str],
    *,
    backend: str,
    cwd: Path,
    log_path: Path,
    timeout_secs: int,
    env: Optional[dict],
    on_summary: Optional[Callable[[str], None]],
    log_mode: str,
    stdin_text: Optional[str] = None,
) -> tuple[Any, list[str], bool, Optional[int]]:
    """One subprocess pass: spawn, stream, kill on timeout.

    Returns ``(stream, captured_lines, timed_out, returncode)``. Retry attempts
    append to the log file (``log_mode="a"``) so the evidence of earlier
    failures survives. ``stdin_text`` (a prompt too long for argv) is written
    from a helper thread so a child that emits output before draining stdin
    cannot deadlock against us.
    """
    streams = {"claude": _ClaudeStream, "codex": _CodexStream, "opencode": _OpencodeStream}
    summarisers = {
        "claude": _summarise_claude_event,
        "codex": _summarise_codex_event,
        "opencode": _summarise_opencode_event,
    }
    stream = streams[backend]()
    summarise = summarisers[backend]
    captured: list[str] = []

    with open(log_path, log_mode, encoding="utf-8") as log_file:
        if log_mode == "a":
            log_file.write("\n--- retry attempt ---\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            # Closed unless we deliver the prompt on it: codex exec reads stdin
            # whenever it is not a tty.
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            # Put the agent in its own process group so a timeout can kill the
            # whole tree. The coding-agent CLIs (claude/opencode/npx) spawn their
            # own children; proc.kill() would SIGKILL only the direct child, leak
            # the grandchildren, and — because a leaked grandchild can hold the
            # stdout pipe open — let the read loop / proc.wait() below hang forever.
            start_new_session=True,
        )
        if stdin_text is not None:
            def _feed_stdin(text: str = stdin_text) -> None:
                try:
                    proc.stdin.write(text)
                finally:
                    proc.stdin.close()

            threading.Thread(target=_feed_stdin, daemon=True).start()
        timed_out = threading.Event()

        def _kill_after():
            timed_out.set()
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()

        timer = threading.Timer(timeout_secs, _kill_after)
        timer.start()
        try:
            for raw_line in proc.stdout:
                log_file.write(raw_line)
                log_file.flush()
                captured.append(raw_line)
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    if on_summary:
                        prefix = {"claude": "cc", "codex": "cx"}.get(backend, "oc")
                        on_summary(f"  [{prefix}] {line}")
                    continue
                if not isinstance(event, dict):
                    continue
                stream.feed(event)
                summary = summarise(event)
                if summary and on_summary:
                    on_summary(summary)
        finally:
            timer.cancel()
            proc.wait()

    return stream, captured, timed_out.is_set(), proc.returncode
