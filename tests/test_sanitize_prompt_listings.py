"""Unit tests for the prompt-listing sanitizer.

The appendix includes the pipeline prompts verbatim via ``\\lstinputlisting``.
The ``listings`` package mis-tokenizes the multi-byte UTF-8 punctuation those
prompts contain: em-/en-dashes render raw in a monospace block, and an en-dash
between digits gets scrambled (the source ``2-4 sentence`` rendered as
``-24 sentence`` in the compiled PDF). These tests pin the ASCII substitutions
that make the listings render predictably, and the idempotency that lets the
script be re-run safely.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "sanitize_prompt_listings.py"


def _load_cli():
    """Load the standalone script as a module so its helpers are the units."""
    spec = importlib.util.spec_from_file_location("sanitize_prompt_listings", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


def test_em_dash_becomes_double_hyphen():
    assert cli.sanitize_text("here — the") == "here -- the"


def test_en_dash_becomes_hyphen():
    assert cli.sanitize_text("roughly 100–300 pairs") == "roughly 100-300 pairs"


def test_ellipsis_becomes_three_dots():
    assert cli.sanitize_text("PARAGRAPH_2 …</p>") == "PARAGRAPH_2 ...</p>"


def test_less_than_or_equal_becomes_ascii():
    assert cli.sanitize_text("p ≤ alpha") == "p <= alpha"


def test_rightwards_arrow_becomes_ascii():
    assert cli.sanitize_text("Hypothesis → PyMC") == "Hypothesis -> PyMC"


def test_digit_dash_digit_regression():
    """The exact string whose en-dash scrambled to '-24' in the rendered PDF."""
    assert cli.sanitize_text("write a 2–4 sentence") == "write a 2-4 sentence"


def test_pure_ascii_is_unchanged():
    text = "def test_statistic(df):\n    return value  # a single float\n"
    assert cli.sanitize_text(text) == text


def test_sanitize_is_idempotent():
    raw = "discrepancy — distinct; p ≤ a; 2–4; x → y; foo …"
    once = cli.sanitize_text(raw)
    assert cli.sanitize_text(once) == once


def test_result_is_pure_ascii():
    raw = "all of: — – … ≤ →"
    assert cli.sanitize_text(raw).isascii()
