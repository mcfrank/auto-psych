"""Tests for shared LLM output parsing helpers (YAML and code blocks)."""

import pytest

from src.agents.llm_output_parsing import (
    ensure_str,
    extract_fenced_blocks,
    extract_yaml_from_response,
    normalize_escaped_newlines,
    normalize_extracted_code,
    try_load_yaml,
)


def test_ensure_str_from_list():
    class Part:
        def __init__(self, text):
            self.text = text
    out = ensure_str([Part("hello"), Part(" world")])
    assert "hello" in out and "world" in out


def test_normalize_escaped_newlines():
    assert normalize_escaped_newlines("a\\nb") == "a\nb"
    assert normalize_escaped_newlines("a\\tb") == "a\tb"
    assert normalize_escaped_newlines("") == ""


def test_try_load_yaml_raw():
    data = try_load_yaml("models:\n  - name: x\nrationale: ok")
    assert data == {"models": [{"name": "x"}], "rationale": "ok"}


def test_try_load_yaml_with_literal_backslash_n():
    # LLM sometimes returns literal \n in the string
    block = "models:\\n  - name: x\\nrationale: ok"
    data = try_load_yaml(block)
    assert data == {"models": [{"name": "x"}], "rationale": "ok"}


def test_extract_yaml_from_response_begin_end():
    response = "prefix\n---BEGIN YAML---\nmodels:\n  - name: m1\n---END YAML---\nsuffix"
    data = extract_yaml_from_response(response)
    assert data == {"models": [{"name": "m1"}]}


def test_extract_yaml_from_response_fence():
    response = "prefix\n```yaml\nmodels:\n  - name: m1\n```\nsuffix"
    data = extract_yaml_from_response(response)
    assert data == {"models": [{"name": "m1"}]}


def test_normalize_extracted_code():
    raw = "def foo():\\n    return 1"
    out = normalize_extracted_code(raw)
    assert out == "def foo():\n    return 1"


def test_normalize_extracted_code_strips_python_line():
    raw = "python\ndef foo():\n    pass"
    out = normalize_extracted_code(raw)
    assert out == "def foo():\n    pass"


def test_extract_fenced_blocks_python():
    text = "text\n```python\ndef f(stimulus, response_options):\n    return {}\n```\nmore"
    blocks = extract_fenced_blocks(text, "python")
    assert len(blocks) == 1
    assert "def f(stimulus, response_options):" in blocks[0]
    assert blocks[0].count("\n") >= 1


def test_extract_fenced_blocks_normalizes_literal_newlines():
    text = "```python\ndef f():\\n    pass\\n```"
    blocks = extract_fenced_blocks(text, "python")
    assert len(blocks) == 1
    assert blocks[0] == "def f():\n    pass"
