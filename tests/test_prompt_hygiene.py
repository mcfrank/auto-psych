"""Guard against committed git conflict markers in src/.

Commit 00276a1 shipped literal ``<<<<<<< HEAD`` / ``=======`` / ``>>>>>>> ...``
conflict markers inside ``src/pipelines/outer_loop/prompts/3_implement.md``, so
three live runs sent a corrupted prompt to the implement agent (resolved in
473ac17). Agent prompts are plain Markdown that no interpreter ever parses, so
a committed conflict silently reaches the LLM. This test fails loudly if any
tracked text file under ``src/`` contains conflict markers.
"""

import subprocess
from pathlib import Path

from pyprojroot import here

# ``<<<<<<< `` and ``>>>>>>> `` (seven markers + space + ref) are unambiguous.
# A bare ``=======`` line is only treated as a marker when the file also has a
# start/end marker, because ``=======`` alone is a legitimate Markdown setext
# heading underline.
_START = "<" * 7 + " "
_END = ">" * 7 + " "
_MID = "=" * 7


def find_conflict_marker_lines(text: str) -> list[int]:
    """Return 1-based line numbers of git conflict markers in ``text``."""
    lines = text.splitlines()
    has_start_or_end = any(
        line.startswith(_START) or line.startswith(_END) for line in lines
    )
    flagged = []
    for lineno, line in enumerate(lines, start=1):
        if line.startswith(_START) or line.startswith(_END):
            flagged.append(lineno)
        elif line == _MID and has_start_or_end:
            flagged.append(lineno)
    return flagged


def _tracked_src_text_files() -> list[Path]:
    root = here()
    out = subprocess.run(
        ["git", "ls-files", "src"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    return [root / line for line in out.stdout.splitlines() if line]


def test_scanner_flags_full_conflict_block():
    text = (
        "regular line\n"
        f"{_START}HEAD\n"
        "ours\n"
        f"{_MID}\n"
        "theirs\n"
        f"{_END}be48bff (some branch)\n"
    )
    assert find_conflict_marker_lines(text) == [2, 4, 6]


def test_scanner_ignores_markdown_setext_heading():
    text = "A Heading\n=======\n\nbody text\n"
    assert find_conflict_marker_lines(text) == []


def test_scanner_flags_start_marker_alone():
    text = f"prose\n{_START}HEAD\nmore prose\n"
    assert find_conflict_marker_lines(text) == [2]


def test_no_conflict_markers_in_tracked_src_files():
    offenders = []
    for path in _tracked_src_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:  # binary asset — nothing to scan
            continue
        for lineno in find_conflict_marker_lines(text):
            offenders.append(f"{path.relative_to(here())}:{lineno}")
    assert not offenders, (
        "Git conflict markers committed in tracked files (resolve the merge "
        "before committing): " + ", ".join(offenders)
    )
