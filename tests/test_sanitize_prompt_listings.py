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

import pytest

from tests.paths import SCRIPTS_DIR, load_script_module


@pytest.fixture(scope="module")
def cli():
    """The standalone script, loaded as a module — its helpers are the units."""
    return load_script_module(SCRIPTS_DIR / "sanitize_prompt_listings.py")


def test_em_dash_becomes_double_hyphen(cli):
    assert cli.sanitize_text("here — the") == "here -- the"


def test_en_dash_becomes_hyphen(cli):
    assert cli.sanitize_text("roughly 100–300 pairs") == "roughly 100-300 pairs"


def test_ellipsis_becomes_three_dots(cli):
    assert cli.sanitize_text("PARAGRAPH_2 …</p>") == "PARAGRAPH_2 ...</p>"


def test_less_than_or_equal_becomes_ascii(cli):
    assert cli.sanitize_text("p ≤ alpha") == "p <= alpha"


def test_rightwards_arrow_becomes_ascii(cli):
    assert cli.sanitize_text("Hypothesis → PyMC") == "Hypothesis -> PyMC"


def test_digit_dash_digit_regression(cli):
    """The exact string whose en-dash scrambled to '-24' in the rendered PDF."""
    assert cli.sanitize_text("write a 2–4 sentence") == "write a 2-4 sentence"


def test_pure_ascii_is_unchanged(cli):
    text = "def test_statistic(df):\n    return value  # a single float\n"
    assert cli.sanitize_text(text) == text


def test_sanitize_is_idempotent(cli):
    raw = "discrepancy — distinct; p ≤ a; 2–4; x → y; foo …"
    once = cli.sanitize_text(raw)
    assert cli.sanitize_text(once) == once


def test_result_is_pure_ascii(cli):
    raw = "all of: — – … ≤ →"
    assert cli.sanitize_text(raw).isascii()
