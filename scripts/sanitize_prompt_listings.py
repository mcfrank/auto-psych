"""Sanitize the appendix prompt listings so ``listings`` renders them correctly.

The paper appendix embeds each pipeline prompt verbatim with
``\\lstinputlisting[style=prompt]{prompts/<name>.txt}``. The ``listings`` package
mis-handles the multi-byte UTF-8 punctuation those prompts contain: em-/en-dashes
appear raw in the monospace block, the ``≤``/``→``/``…`` glyphs are not covered by
the package's ``literate`` rules, and an en-dash between digits gets scrambled
(the source ``2-4 sentence`` came out as ``-24 sentence`` in the compiled PDF).

This script rewrites those characters to ASCII equivalents in the prompt files
that the appendix includes. It only ever touches the characters in
``SUBSTITUTIONS`` — every other byte is preserved exactly — so it is idempotent
and safe to re-run. By default it edits the files in place; pass ``--check`` to
preview which files would change without writing.

Usage::

    python scripts/sanitize_prompt_listings.py           # rewrite in place
    python scripts/sanitize_prompt_listings.py --check    # dry run, report only
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro
from pyprojroot import here

# Ordered map from the non-ASCII characters that appear in the prompts to ASCII
# equivalents that render predictably in a monospace listing. Keep this list in
# sync with the characters actually present (the script asserts the output is
# pure ASCII, so an unmapped character fails loudly rather than slipping through).
SUBSTITUTIONS: dict[str, str] = {
    "—": "--",   # — EM DASH
    "–": "-",    # – EN DASH
    "…": "...",  # … HORIZONTAL ELLIPSIS
    "≤": "<=",   # ≤ LESS-THAN OR EQUAL TO
    "→": "->",   # → RIGHTWARDS ARROW
}

# Directory holding the prompt files included by appendix.tex.
PROMPTS_DIR = here("paper-source") / "prompts"


def sanitize_text(text: str) -> str:
    """Replace the non-ASCII punctuation in ``text`` with ASCII equivalents.

    Fails loudly if any non-ASCII character remains, so a newly introduced glyph
    is caught here rather than producing another silently mangled listing.
    """
    for char, replacement in SUBSTITUTIONS.items():
        text = text.replace(char, replacement)
    if not text.isascii():
        offending = sorted({c for c in text if not c.isascii()})
        raise ValueError(
            "sanitize_text left unmapped non-ASCII characters: "
            + ", ".join(f"U+{ord(c):04X} {c!r}" for c in offending)
            + " — add them to SUBSTITUTIONS."
        )
    return text


@dataclass
class Args:
    """Command-line arguments."""

    prompts_dir: Path = PROMPTS_DIR
    """Directory of prompt .txt files included by the appendix."""

    check: bool = False
    """Report which files would change without writing them."""


def main(args: Args) -> None:
    prompt_files = sorted(args.prompts_dir.glob("*.txt"))
    if not prompt_files:
        raise FileNotFoundError(f"No prompt .txt files found in {args.prompts_dir}")

    changed = []
    for path in prompt_files:
        original = path.read_text(encoding="utf-8")
        cleaned = sanitize_text(original)
        if cleaned == original:
            continue
        changed.append(path)
        if not args.check:
            path.write_text(cleaned, encoding="utf-8")

    verb = "would change" if args.check else "rewrote"
    if changed:
        print(f"{verb} {len(changed)} of {len(prompt_files)} prompt file(s):")
        for path in changed:
            print(f"  {path.relative_to(args.prompts_dir.parent)}")
    else:
        print(f"All {len(prompt_files)} prompt file(s) already ASCII-clean.")


if __name__ == "__main__":
    main(tyro.cli(Args))
