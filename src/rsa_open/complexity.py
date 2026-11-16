"""Code-size complexity of a theory file (docs/auto_open/RSA_GRID.md; PI 2026-11-16).

Measures of what a theory's author wrote, to set against how well it fits (the complexity frontier), not a penalty:
an exchange rate between nats and code would be a hand-set number. Every measure reads the theory's **active code**
(`active_code`): the file without comments and docstrings (a hypothesis written out at length costs nothing), without
dead code (helpers, assignments and imports nothing uses), with switched-off mechanisms removed (the closed
consolidation's variant files carry the whole superset and switch mechanisms off by fixing their weights at 0 or by
``if "<mechanism>" in on`` blocks; `prune_mechanisms` partially evaluates them, and a test checks the pruned file
predicts exactly as the original). The other measures read it in one layout (``ast.unparse``), so formatting costs
nothing; lines are counted in the author's layout:

- ``lines``: lines of the active code in the author's own layout (`active_lines`);
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


class _PriorArgs(ast.NodeTransformer):
    """Numbers inside a prior distribution (``dist.X(...)``) are priors, not hand-set values: replaced by a name."""

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "dist":
            return ast.Call(func=node.func, args=[ast.Name(id="prior", ctx=ast.Load()) for _ in node.args],
                            keywords=[ast.keyword(arg=k.arg, value=ast.Name(id="prior", ctx=ast.Load())) for k in node.keywords])
        return self.generic_visit(node)


def hand_set_numbers(source: str) -> int:
    """Numbers set by hand outside the priors, by the open loop's gate (numerical-safety sentinels such as a mask of
    999 or a floor of 1e-15 count, as the gate counts them)."""
    from src.rsa_open.loop.gates import magic_number_problems

    source = ast.unparse(ast.fix_missing_locations(_PriorArgs().visit(ast.parse(source))))
    problems = magic_number_problems(source)
    if not problems:
        return 0
    m = re.search(r"sets (\d+) number", problems[0])
    if m is None:
        raise ValueError(f"unexpected gate message: {problems[0][:120]}")
    return int(m.group(1))


# ---------- active code ----------

class _Zero(ast.Constant):
    """A literal 0 from a switched-off weight; ``shape`` is an expression of the array it multiplied, if known."""

    def __init__(self, shape=None):
        super().__init__(value=0.0)
        self.shape = shape


def _is_zero(n) -> bool:
    return isinstance(n, _Zero) or (isinstance(n, ast.Constant) and n.value == 0.0 and not isinstance(n.value, bool))


def _is_one(n) -> bool:
    return isinstance(n, ast.Constant) and n.value == 1.0 and not isinstance(n.value, bool)


class _PruneMechanisms(ast.NodeTransformer):
    def __init__(self, on, off_params):
        self.on, self.off = set(on), set(off_params)

    def _mechanism_test(self, test):
        """True/False for ``"m" in on`` / ``"m" not in on``; None for any other test."""
        if (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.left, ast.Constant)
                and isinstance(test.left.value, str) and isinstance(test.comparators[0], ast.Name)
                and test.comparators[0].id == "on"):
            inside = test.left.value in self.on
            if isinstance(test.ops[0], ast.In):
                return inside
            if isinstance(test.ops[0], ast.NotIn):
                return not inside
        return None

    def visit_If(self, node):
        verdict = self._mechanism_test(node.test)
        if verdict is None:
            return self.generic_visit(node)
        kept = node.body if verdict else node.orelse
        return [self.visit(n) for n in kept] or None

    def visit_Subscript(self, node):
        self.generic_visit(node)
        if (isinstance(node.value, ast.Name) and node.value.id == "q" and isinstance(node.slice, ast.Constant)
                and node.slice.value in self.off):
            return _Zero()
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.USub) and _is_zero(node.operand):
            return node.operand
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        a, b, op = node.left, node.right, node.op
        if isinstance(op, ast.Mult):
            if _is_zero(a) or _is_zero(b):
                return _Zero(shape=b if _is_zero(a) else a)
            if _is_one(a):
                return b
            if _is_one(b):
                return a
        if isinstance(op, ast.Add):
            if _is_zero(a):
                return b
            if _is_zero(b):
                return a
        if isinstance(op, ast.Sub):
            if _is_zero(b):
                return a
            if _is_zero(a):
                return ast.UnaryOp(op=ast.USub(), operand=b)
        if isinstance(op, ast.Div):
            if _is_one(b):
                return a
            if _is_zero(a):
                return _Zero(shape=getattr(a, "shape", None))
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        if (isinstance(node.func, ast.Attribute) and node.func.attr == "exp" and len(node.args) == 1
                and _is_zero(node.args[0])):
            shape = getattr(node.args[0], "shape", None)
            if shape is None:
                return ast.Constant(value=1.0)
            return ast.Call(func=ast.Attribute(value=ast.Name(id="jnp", ctx=ast.Load()), attr="ones_like", ctx=ast.Load()),
                            args=[shape], keywords=[])
        return node


def _literal(tree: ast.Module, name: str):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node
    return None


def prune_mechanisms(tree: ast.Module) -> ast.Module:
    """A consolidation variant file (``MECHANISMS_ON``, ``MECHANISM_PARAMS``, ``OFF``) with its switched-off
    mechanisms removed: their ``if "<m>" in on`` blocks resolved, their weights (fixed at 0 in ``OFF``) substituted
    and simplified away, and their parameters dropped from the tables. Any other file is returned unchanged."""
    on_node, table, off = _literal(tree, "MECHANISMS_ON"), _literal(tree, "MECHANISM_PARAMS"), _literal(tree, "OFF")
    if on_node is None or table is None or off is None:
        return tree
    on = set(ast.literal_eval(on_node.value))
    off_params = set()
    for k, v in zip(table.value.keys, table.value.values):
        if k.value not in on:
            off_params |= {kk.value for kk in v.keys}
    table.value.keys, table.value.values = map(list, zip(*[(k, v) for k, v in zip(table.value.keys, table.value.values)
                                                          if k.value in on])) if on else ([], [])
    off.value.keys, off.value.values = map(list, zip(*[(k, v) for k, v in zip(off.value.keys, off.value.values)
                                                      if k.value not in off_params])) or ([], [])
    return _PruneMechanisms(on, off_params).visit(tree)


ROOTS = {"PARAMS", "listener", "speaker", "choice_probs", "MECHANISMS_ON"}


def _loads(nodes) -> set:
    return {n.id for root in nodes for n in ast.walk(root) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} \
        | {n.value.id for root in nodes for n in ast.walk(root)
           if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}


def _assigned(stmt) -> set:
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {stmt.name}
    if isinstance(stmt, ast.Assign):
        return {n.id for t in stmt.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)) and isinstance(stmt.target, ast.Name):
        return {stmt.target.id}
    return set()


def _drop_dead(body: list, keep: set) -> list:
    """Statements of ``body`` whose names nothing else in ``body`` (or ``keep``) reads, removed until none is left."""
    while True:
        out = []
        for i, stmt in enumerate(body):
            names = _assigned(stmt)
            if names and not names & keep and not names & _loads([s for j, s in enumerate(body) if j != i]):
                continue
            out.append(stmt)
        if len(out) == len(body):
            return out
        body = out


class _DeadLocals(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        returns = [s for s in node.body if isinstance(s, ast.Return)]
        node.body = _drop_dead(node.body, set()) or returns or [ast.Pass()]
        return node


def remove_dead_code(tree: ast.Module) -> ast.Module:
    """The module without functions, assignments or imported names that nothing reachable from
    ``PARAMS``/``listener``/``speaker`` uses, and without local assignments a function never reads."""
    tree = _DeadLocals().visit(tree)
    tree.body = _drop_dead(tree.body, ROOTS)
    used = _loads(tree.body)
    body = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            stmt.names = [a for a in stmt.names if (a.asname or a.name).split(".")[0] in used]
            if not stmt.names:
                continue
        body.append(stmt)
    tree.body = body
    return tree


def _active(source: str):
    """(code without comments and docstrings, the active tree over it: mechanisms pruned, dead code removed)."""
    code = code_only(source)
    return code, ast.fix_missing_locations(remove_dead_code(prune_mechanisms(ast.parse(code))))


def active_code(source: str) -> str:
    """What the measures read, in ``ast.unparse``'s layout."""
    return ast.unparse(_active(source)[1])


def active_lines(source: str) -> int:
    """Lines of the active code in the author's own layout: every line of a surviving simple statement, the header
    lines (decorators, signature) of a surviving compound one. A statement simplified in place keeps its lines."""
    _, tree = _active(source)
    covered = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.stmt) or not hasattr(node, "end_lineno") or getattr(node, "lineno", 0) == 0:
            continue
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and hasattr(body[0], "lineno"):
            first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            covered.update(range(first, max(first, body[0].lineno - 1) + 1) if body[0].lineno > node.lineno else [node.lineno])
        else:
            covered.update(range(node.lineno, node.end_lineno + 1))
    return len(covered)


def measure(source: str) -> Dict[str, Optional[float]]:
    code = active_code(source)
    return dict(lines=active_lines(source),
                ast_nodes=sum(1 for _ in ast.walk(ast.parse(code))),
                halstead_volume=round(halstead_volume(code), 1),
                compressed_bytes=len(zlib.compress(code.encode(), 9)),
                n_params=literal_params(source),
                hand_set_numbers=hand_set_numbers(code))
