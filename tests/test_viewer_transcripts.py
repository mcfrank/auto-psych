"""Unit tests for cleaning the agent terminal transcripts.

The ``*.jsonl`` log files in a run are misnamed — they are raw colored terminal
output from the coding agent, not JSON. The viewer must strip the ANSI escape
codes so the transcript is readable in the browser, while preserving the actual
text and line structure.
"""

from __future__ import annotations

from src.viewer.transcripts import strip_ansi


def test_strips_color_codes_but_keeps_text():
    raw = "\x1b[0m> build · gemini-3.1-pro-preview\x1b[0m\n\x1b[0m$ \x1b[0mls -la\n"
    cleaned = strip_ansi(raw)
    assert "\x1b" not in cleaned
    assert "[0m" not in cleaned
    assert "> build · gemini-3.1-pro-preview" in cleaned
    assert "$ ls -la" in cleaned


def test_preserves_lines_and_plain_text():
    raw = "line one\nline two\n"
    assert strip_ansi(raw) == "line one\nline two\n"


def test_handles_empty_string():
    assert strip_ansi("") == ""
