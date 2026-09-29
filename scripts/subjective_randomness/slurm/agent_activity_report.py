"""What a cell's agents looked at: URLs, and paths outside their own directory.

    python agent_activity_report.py --agent-dir A --out RUN_DIR/agent_activity.md

Warn-only, for a person to skim: agents may look up published papers, and the
sandbox keeps them out of other runs, but nothing recorded either. Scans the
raw text of every agent log under the agent directory, so it works for every
backend's log format. Prints a WARNING line when an outside path appears.
"""

from __future__ import annotations

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


def activity(
    agent_dir: Path, also_expected: Sequence[str] = ()
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """(URL -> count, outside path -> count) over the agent logs in ``agent_dir``.

    ``also_expected`` adds path prefixes an agent may use: the home directory
    (inside the sandbox it is the agent's private home, and the Python install
    is mounted read-only under it) and the venv.
    """
    agent_dir = Path(agent_dir)
    expected = EXPECTED_PREFIXES + tuple(str(p) for p in also_expected)
    own = str(agent_dir.resolve())
    urls: Counter = Counter()
    outside: Counter = Counter()
    for log in sorted(agent_dir.rglob("*.jsonl")):
        text = log.read_text(encoding="utf-8", errors="replace")
        urls.update(URL.findall(text))
        for path in ABSOLUTE_PATH.findall(text):
            path = path.rstrip(".,:")
            if path.startswith(own) or path.startswith(str(agent_dir)):
                continue
            if path.startswith(expected):
                continue
            outside[path] += 1
    return dict(urls), dict(outside)


def render(urls: Dict[str, int], outside: Dict[str, int]) -> str:
    lines = ["# What the agents looked at", "", "## URLs in the agent logs", ""]
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
    urls, outside = activity(args.agent_dir, args.expected)
    args.out.write_text(render(urls, outside), encoding="utf-8")
    if outside:
        print(
            f"WARNING: agents touched {len(outside)} path(s) outside their own "
            f"directory; see {args.out}"
        )
    print(
        f"[activity] {len(urls)} URL(s), {len(outside)} outside path(s) -> {args.out}"
    )


if __name__ == "__main__":
    main(tyro.cli(Args))
