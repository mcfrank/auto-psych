"""Build a paradigm's page: the generic player (`page.html`) with its trial lists baked in.

    uv run python -m src.rsa_open.page --lists lists.json --out page.html [--preview]

The page needs nothing from the network but jsPsych from unpkg (as the closed page). Without
``consent_html`` the consent screen is a marked placeholder (`CONSENT_PLACEHOLDER_MARKER`),
which a live deployment must replace; the live deploy passes ``None`` and shows its own gate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import tyro

TEMPLATE = Path(__file__).with_name("page.html")
CONSENT_PLACEHOLDER_MARKER = "ro-placeholder"
CONSENT_PLACEHOLDER = (
    '<div class="ro-placeholder"><h1>Consent</h1><p><strong>[CONSENT PLACEHOLDER]</strong> The approved '
    "consent text goes here. This build is a prototype and must not be shown to real participants.</p></div>"
)
_DEFAULT = object()


def _js(value: Any) -> str:
    """JSON safe inside a <script> element."""
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/").replace(" ", "\\u2028").replace(
        " ", "\\u2029")


def render(lists: Mapping[str, Any], *, preview: bool = False, consent_html: Any = _DEFAULT) -> str:
    from src.rsa_open.screens import SCHEMA

    if lists.get("schema") != SCHEMA:
        raise ValueError(f"not trial lists of schema {SCHEMA}: {lists.get('schema')!r}")
    consent = CONSENT_PLACEHOLDER if consent_html is _DEFAULT else consent_html
    values = {"LISTS_JSON": lists, "PREVIEW_JSON": bool(preview), "CONSENT_HTML_JSON": consent}
    page = TEMPLATE.read_text(encoding="utf-8")
    for key in values:
        if page.count("{{%s}}" % key) != 1:
            raise ValueError(f"the page template must contain {{{{{key}}}}} exactly once")
    # One pass, so text inside the lists can never be taken for a placeholder.
    return re.sub(r"\{\{(%s)\}\}" % "|".join(values), lambda m: _js(values[m.group(1)]), page)


@dataclass
class Args:
    lists: Path
    out: Path
    preview: bool = False


def main(args: Args) -> Path:
    args.out.write_text(render(json.loads(args.lists.read_text()), preview=args.preview), encoding="utf-8")
    print(f"page -> {args.out}")
    return args.out


if __name__ == "__main__":
    main(tyro.cli(Args))
