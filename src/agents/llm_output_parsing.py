"""
Shared helpers for parsing structured output from LLM responses.

LLM/API responses can contain literal \\n instead of newlines, or inconsistent
fence formatting. Use these helpers so all agents that parse YAML, JSON, or
code blocks from LLM output behave consistently.
"""

import re
from typing import Any, Dict, List, Optional

import yaml


def ensure_str(response: Any) -> str:
    """Normalize LLM response to a single string (handles list content from API)."""
    if isinstance(response, list):
        return " ".join(
            (getattr(part, "text", None) or str(part)) for part in response
        )
    return str(response or "")


def normalize_escaped_newlines(text: str) -> str:
    """Replace literal \\n, \\t, and escaped quotes in text with real chars."""
    if not text:
        return text
    text = text.replace("\\n", "\n").replace("\\t", "\t")
    # LLMs sometimes emit \\' and \\" in code; in Python source we need ' and "
    text = text.replace("\\'", "'").replace('\\"', '"')
    return text


def try_load_yaml(block: str) -> Optional[Dict[str, Any]]:
    """
    Parse block as YAML; try raw first, then with normalized newlines.
    Returns dict or None.
    """
    if not block or not block.strip():
        return None
    for raw in (block, normalize_escaped_newlines(block)):
        try:
            data = yaml.safe_load(raw)
            if isinstance(data, dict):
                return data
        except yaml.YAMLError:
            continue
    return None


def extract_yaml_from_response(
    response: str,
    begin_marker: str = "---BEGIN YAML---",
    end_marker: str = "---END YAML---",
    fence_start_alternatives: tuple = ("```yaml", "```"),
) -> Optional[Dict[str, Any]]:
    """
    Extract a YAML block from the response and parse it.
    Tries begin/end markers first, then line-by-line for fence alternatives.
    """
    text = ensure_str(response).strip()
    if begin_marker in text and end_marker in text:
        block = text.split(begin_marker)[1].split(end_marker)[0].strip()
        data = try_load_yaml(block)
        if data is not None:
            return data
    lines = text.split("\n")
    in_block = False
    block_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped in (end_marker, "```"):
            break
        if stripped in (begin_marker,) + fence_start_alternatives:
            in_block = True
            continue
        if in_block:
            block_lines.append(line)
    if block_lines:
        data = try_load_yaml("\n".join(block_lines))
        if data is not None:
            return data
    return None


def normalize_extracted_code(raw: str, strip_language_line: bool = True) -> str:
    """
    Turn literal \\n/\\t in extracted code into real newlines/tabs.
    If strip_language_line is True, remove a first line that is only the
    language tag (e.g. "python").
    """
    if not raw:
        return raw
    code = normalize_escaped_newlines(raw)
    if strip_language_line:
        first_line, _, rest = code.partition("\n")
        if first_line.strip().lower() == "python":
            code = rest.lstrip("\n")
    return code.strip()


def extract_fenced_blocks(
    text: str,
    language: str = "python",
    *,
    normalize: bool = True,
    min_length: int = 10,
) -> List[str]:
    """
    Extract contents of fenced code blocks from text.
    Looks for ```language ... ``` (case-insensitive), then falls back to
    any ``` ... ``` block that looks like code (e.g. contains "def " for python,
    or "expected_information_gain" / "out_dir" for designer scripts).
    Returns list of block contents; if normalize is True, applies
    normalize_extracted_code to each.
    """
    text = ensure_str(text)
    pattern_lang = re.compile(
        rf"```\s*{re.escape(language)}\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )
    blocks = pattern_lang.findall(text)
    if not blocks and language == "python":
        pattern_any = re.compile(r"```\s*\n?(.*?)```", re.DOTALL)
        for raw in pattern_any.findall(text):
            raw_stripped = raw.strip()
            # Original: def + (stimulus or response_options)
            if "def " in raw_stripped and ("stimulus" in raw_stripped or "response_options" in raw_stripped):
                blocks.append(raw_stripped)
                break
            # Designer script: uses pipeline helpers
            if ("expected_information_gain" in raw_stripped or "out_dir" in raw_stripped) and len(raw_stripped) >= min_length:
                blocks.append(raw_stripped)
                break
    result = []
    for raw in blocks:
        content = normalize_extracted_code(raw.strip()) if normalize else raw.strip()
        if len(content) >= min_length:
            result.append(content)
    return result
