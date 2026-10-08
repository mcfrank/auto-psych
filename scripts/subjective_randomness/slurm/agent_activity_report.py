"""What a cell's agents looked at: web tool calls, URLs, and paths outside their own directory.

    python agent_activity_report.py --agent-dir A --out RUN_DIR/agent_activity.md

Warn-only, for a person to skim: agents may look up published papers, and the
sandbox keeps them out of other runs, but nothing recorded either. Scans the
raw text of every agent log under the agent directory, so it works for every
backend's log format. Prints a WARNING line when an outside path appears.

Web tool calls (webfetch, websearch, ...: what an agent actually fetched or
searched, with each call's status) are listed apart from URLs that merely
appear in log text (a warning message, a model's docstring, a grepped
``uv.lock``): Sherlock run 2's no-network check could not be read from one list.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence, Tuple

import tyro

URL = re.compile(r"https?://[^\s\"'\\)>\]]+")
ABSOLUTE_PATH = re.compile(
    r"(?<![\w.:])/(?:scratch|home|oak|share|tmp|users|groups)/[^\s\"'\;|&)]+"
)
# Where an agent is expected to be: its scratch (/tmp) and the modules tree.
EXPECTED_PREFIXES = ("/tmp", "/share/software")
# Web tools by name (opencode's and Claude Code's), lower-cased.
WEB_TOOLS = {"webfetch", "websearch", "codesearch", "web_fetch", "web_search"}


def web_calls(event) -> list[str]:
    """Each web tool call in one log event: ``"<tool> <url or query> (<status>)"``.

    Finds any mapping whose ``tool`` or ``name`` is a web tool (opencode puts
    the call in ``part``, Claude Code in a ``tool_use`` content block) and
    reads its input (``input`` or ``state.input``) and status.
    """
    found = []
    if isinstance(event, dict):
        tool = event.get("tool") or event.get("name")
        if isinstance(tool, str) and tool.lower() in WEB_TOOLS:
            state = event.get("state") if isinstance(event.get("state"), dict) else {}
            given = event.get("input") or state.get("input") or {}
            if isinstance(given, dict):
                target = given.get("url") or given.get("query") or json.dumps(given)[:200]
            else:
                target = str(given)
            found.append(f"{tool.lower()} {target} ({state.get('status', 'called')})")
        for value in event.values():
            found += web_calls(value)
    elif isinstance(event, list):
        for value in event:
            found += web_calls(value)
    return found


def activity(
    agent_dir: Path, also_expected: Sequence[str] = ()
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """(web call -> count, URL in log text -> count, outside path -> count)
    over the agent logs in ``agent_dir``.

    ``also_expected`` adds path prefixes an agent may use: the home directory
    (inside the sandbox it is the agent's private home, and the Python install
    is mounted read-only under it) and the venv.
    """
    agent_dir = Path(agent_dir)
    expected = EXPECTED_PREFIXES + tuple(str(p) for p in also_expected)
    own = str(agent_dir.resolve())
    calls: Counter = Counter()
    urls: Counter = Counter()
    outside: Counter = Counter()
    for log in sorted(agent_dir.rglob("*.jsonl")):
        text = log.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            try:
                calls.update(web_calls(json.loads(line)))
            except json.JSONDecodeError:
                continue
        urls.update(URL.findall(text))
        for path in ABSOLUTE_PATH.findall(text):
            path = path.rstrip(".,:")
            if path.startswith(own) or path.startswith(str(agent_dir)):
                continue
            if path.startswith(expected):
                continue
            outside[path] += 1
    return dict(calls), dict(urls), dict(outside)


def render(calls: Dict[str, int], urls: Dict[str, int], outside: Dict[str, int]) -> str:
    lines = ["# What the agents looked at", "", "## Web tool calls", "",
             "What agents fetched or searched, with each call's status (a denied call is an error).", ""]
    lines += [f"- {call} ({n}×)" for call, n in sorted(calls.items())] or ["(none)"]
    lines += ["", "## URLs that appear in the log text", "",
              "Anywhere in the logs: tool output, warnings, file contents, model text. Not fetches.", ""]
    lines += [f"- {url} ({n}×)" for url, n in sorted(urls.items())] or ["(none)"]
    lines += ["", "## Paths outside the agent's own directory", ""]
    lines += [f"- {path} ({n}×)" for path, n in sorted(outside.items())] or ["(none)"]
    return "\n".join(lines) + "\n"


@dataclass
class Args:
    agent_dir: Path
    """The cell's agent directory (agent_trees/<hash>), holding repo/_runs."""
    out: Path
    """Where to write the report (outside the agent tree)."""
    expected: Tuple[str, ...] = ()
    """Further path prefixes agents may use (their private home, the venv)."""


def main(args: Args) -> None:
    calls, urls, outside = activity(args.agent_dir, args.expected)
    args.out.write_text(render(calls, urls, outside), encoding="utf-8")
    if outside:
        print(
            f"WARNING: agents touched {len(outside)} path(s) outside their own "
            f"directory; see {args.out}"
        )
    print(
        f"[activity] {sum(calls.values())} web tool call(s), {len(urls)} URL(s) in log text, "
        f"{len(outside)} outside path(s) -> {args.out}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
