"""Code-size complexity of a theory file (docs/auto_open/RSA_GRID.md; PI 2026-11-16).

Measures of what a theory's author wrote, to set against how well it fits (the complexity frontier), not a penalty:
an exchange rate between nats and code would be a hand-set number. Every measure reads the file's own code with its
comments and docstrings removed (a hypothesis written out at length costs nothing):

- ``lines``: non-blank lines of code;
- ``ast_nodes``: nodes of the parsed module (a size that ignores layout and naming);
- ``halstead_volume``: Halstead's volume, (N1 + N2) log2(n1 + n2) over the code's tokens, operators (operator symbols
  and keywords) and operands (names, numbers, strings), N total and n distinct;
- ``compressed_bytes``: the zlib-compressed code (level 9), a crude description length that discounts repetition;
- ``n_params``: free parameters, when ``PARAMS`` is a literal dict (else ``None``; the caller counts them by loading);
- ``hand_set_numbers``: numbers set by hand outside ``PARAMS`` (`src.rsa_open.loop.gates.magic_number_problems`), which
  are fitted parameters in all but name when an agent tuned them on data it had seen.
"""

from __future__ import annotations

import ast
import io
import keyword
import math
import re
import tokenize
import zlib
from typing import Dict, Optional

MEASURES = ("lines", "ast_nodes", "halstead_volume", "compressed_bytes")


def _docstring_nodes(tree: ast.AST) -> set:
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                out.add(first)
    return out


def code_only(source: str) -> str:
    """``source`` without comments, docstrings or blank lines (the code a measure reads)."""
    lines = source.splitlines()
    for node in _docstring_nodes(ast.parse(source)):
        for i in range(node.lineno - 1, node.end_lineno):
            lines[i] = ""
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if tok.type == tokenize.COMMENT:
            row, col = tok.start
            lines[row - 1] = lines[row - 1][:col]
    return "\n".join(line.rstrip() for line in lines if line.strip())


def halstead_volume(code: str) -> float:
    operators, operands = [], []
    for tok in tokenize.generate_tokens(io.StringIO(code).readline):
        if tok.type == tokenize.OP:
            operators.append(tok.string)
        elif tok.type == tokenize.NAME:
            (operators if keyword.iskeyword(tok.string) else operands).append(tok.string)
        elif tok.type in (tokenize.NUMBER, tokenize.STRING):
            operands.append(tok.string)
    n = len(set(operators)) + len(set(operands))
    return (len(operators) + len(operands)) * math.log2(n) if n > 1 else 0.0


def literal_params(source: str) -> Optional[int]:
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "PARAMS" for t in node.targets):
            return len(node.value.keys) if isinstance(node.value, ast.Dict) else None
    return None


def hand_set_numbers(source: str) -> int:
    from src.rsa_open.loop.gates import magic_number_problems

    problems = magic_number_problems(source)
    if not problems:
        return 0
    m = re.search(r"sets (\d+) number", problems[0])
    if m is None:
        raise ValueError(f"unexpected gate message: {problems[0][:120]}")
    return int(m.group(1))


def measure(source: str) -> Dict[str, Optional[float]]:
    code = code_only(source)
    return dict(lines=len(code.splitlines()),
                ast_nodes=sum(1 for _ in ast.walk(ast.parse(code))),
                halstead_volume=round(halstead_volume(code), 1),
                compressed_bytes=len(zlib.compress(code.encode(), 9)),
                n_params=literal_params(source),
                hand_set_numbers=hand_set_numbers(source))
