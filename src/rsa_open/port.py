"""Port a closed-campaign RSA model file to a paradigm-free theory (docs/auto_open/PLAN.md §3.1).

A closed model file defines ``PARAMS`` and ``choice_probs(params, ctx)``, the probability of
clicking each object. The port keeps the theory and leaves the response process to the link
(`src.rsa_open.links`: a decision temperature and a uniform lapse, PI 2026-10-21). Two rules:

- **(a) a final uniform lapse.** Every ``return`` of ``choice_probs`` is
  ``with_lapse(X, params["lapse"])`` and ``PARAMS["lapse"]`` is the choice link's prior,
  ``Beta(1.0, 9.0)``: ``lapse`` leaves ``PARAMS`` and each return becomes ``return X``.
  Composed with the choice link, the port equals the original at temperature 1.
- **(b) no final lapse.** No return is ``with_lapse(...)``: the whole function is the theory,
  and a parameter named ``lapse`` (a latent mixture of the theory's own) is renamed
  ``theory_lapse`` so the link's lapse cannot
  clash. Composed with the choice link, the port equals the original at link lapse 0 and
  temperature 1; the theory's mixture stays only if it earns its parameter over the link's.

Either way the function is renamed ``listener``. The port edits only those spans of the source
text: memo parses its own source, so the memo definitions, and every other line, stay byte for
byte. A file that fits neither rule is refused with the reason (`PortRefused`); there is no
fallback.

**No hand ports** (PI 2026-10-22): a decision temperature written into a closed theory stays in
it, under its own name (`softmax_surprisal_isolated_costly_l4`'s ``beta``). E, the literature, is
the reference paradigm and its link's temperature is fixed at 1 (`src.rsa_open.corpus`), so such a
parameter is the theory's own sharpness, shared by every paradigm. (From 2026-10-21 to 10-22 that
``beta`` was cut by hand and left to the link.)

    python -m src.rsa_open.port <model.py> ... --out <dir>
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import tyro

LAPSE = "lapse"
LINK_LAPSE_PRIOR = ("Beta", (1.0, 9.0))  # the choice link's prior on its lapse
LISTENER = "listener"
THEORY_LAPSE = "theory_lapse"
HEADER = "# Ported from {name} by src.rsa_open.port, rule ({rule}) (docs/auto_open/PLAN.md §3.1).\n"


class PortRefused(ValueError):
    """The file does not have the shape the mechanical port requires."""


def _line_starts(source: str) -> List[int]:
    starts, offset = [0], 0
    for line in source.splitlines(keepends=True):
        offset += len(line)
        starts.append(offset)
    return starts


def _span(node: ast.AST, starts: List[int]) -> Tuple[int, int]:
    return (starts[node.lineno - 1] + node.col_offset,
            starts[node.end_lineno - 1] + node.end_col_offset)


def _is_lapse_param(node: ast.AST) -> bool:
    """``params["lapse"]``."""
    return (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
            and node.value.id == "params" and isinstance(node.slice, ast.Constant)
            and node.slice.value == LAPSE)


def _own_returns(fn: ast.FunctionDef) -> List[ast.Return]:
    """The returns of ``fn`` itself, not of functions or lambdas defined inside it."""
    found, stack = [], list(fn.body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Return):
            found.append(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
            continue
        stack.extend(ast.iter_child_nodes(node))
    return found


def _params_dict(tree: ast.Module, name: str) -> ast.Dict:
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "PARAMS"):
            if not isinstance(node.value, ast.Dict):
                raise PortRefused(f"{name}: PARAMS must be a dict literal to be ported")
            return node.value
    raise PortRefused(f"{name}: no module-level PARAMS")


def _check_lapse_prior(value: ast.AST, name: str) -> None:
    family, args = LINK_LAPSE_PRIOR
    ok = (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
          and value.func.attr == family and not value.keywords and len(value.args) == len(args)
          and all(isinstance(a, ast.Constant) and float(a.value) == b for a, b in zip(value.args, args)))
    if not ok:
        raise PortRefused(
            f"{name}: PARAMS['lapse'] is {ast.unparse(value)}, not the choice link's "
            f"{family}{args}; a port with another lapse prior would not reproduce the original")


def port(source: str, name: str) -> str:
    """The theory file for one closed model file's ``source`` (``name`` labels errors)."""
    tree = ast.parse(source)
    params = _params_dict(tree, name)
    keys = [k.value if isinstance(k, ast.Constant) else None for k in params.keys]
    fns = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "choice_probs"]
    if not fns:
        raise PortRefused(f"{name}: no module-level choice_probs(params, ctx)")
    defined = {getattr(n, "name", None) for n in tree.body} | {
        getattr(t, "id", None) for n in tree.body for t in getattr(n, "targets", [])}
    if LISTENER in defined:
        raise PortRefused(f"{name}: already defines '{LISTENER}', the name the port gives choice_probs")
    if THEORY_LAPSE in keys:
        raise PortRefused(f"{name}: PARAMS already has '{THEORY_LAPSE}', the name rule (b) uses")
    fn = fns[0]
    returns = _own_returns(fn)

    def is_final_lapse(v):
        return (isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "with_lapse"
                and len(v.args) == 2 and not v.keywords and _is_lapse_param(v.args[1]))

    wrapped = [is_final_lapse(r.value) for r in returns]
    starts = _line_starts(source)
    edits = []  # (start, end, replacement)
    mentions = [n for n in ast.walk(tree) if isinstance(n, ast.Constant) and n.value == LAPSE]
    if returns and all(wrapped):
        rule = "a"
        if LAPSE not in keys:
            raise PortRefused(f"{name}: returns with_lapse(..., params['lapse']) but PARAMS has no 'lapse'")
        lapse_index = keys.index(LAPSE)
        _check_lapse_prior(params.values[lapse_index], name)
        if len(mentions) != 1 + len(returns):
            raise PortRefused(
                f"{name}: 'lapse' is used outside PARAMS and the final with_lapse "
                f"({len(mentions)} mentions); the port would change the model")
        # The PARAMS entry, with its trailing comma, and its whole line when it stands alone.
        start, _ = _span(params.keys[lapse_index], starts)
        _, end = _span(params.values[lapse_index], starts)
        comma = re.match(r"[ \t]*,", source[end:])
        if comma:
            end += comma.end()
        line_start = source.rfind("\n", 0, start) + 1
        line_end = source.find("\n", end)
        line_end = len(source) if line_end == -1 else line_end + 1
        if not source[line_start:start].strip() and not source[end:line_end].strip():
            start, end = line_start, line_end
        edits.append((start, end, ""))
        # Each return's value becomes the lapse's argument.
        for ret in returns:
            call = ret.value
            s, e = _span(call, starts)
            a, b = _span(call.args[0], starts)
            edits.append((s, e, source[a:b]))
    elif not any(wrapped):
        rule = "b"
        allowed = {id(k) for k in params.keys if isinstance(k, ast.Constant) and k.value == LAPSE}
        allowed |= {id(n.slice) for n in ast.walk(tree) if _is_lapse_param(n)}
        stray = [m for m in mentions if id(m) not in allowed]
        if stray:
            raise PortRefused(f"{name}: 'lapse' appears outside PARAMS and params['lapse'] "
                              f"(line {stray[0].lineno}); rule (b) renames only those")
        for m in mentions:
            s, e = _span(m, starts)
            quote = source[s]
            edits.append((s, e, f"{quote}{THEORY_LAPSE}{quote}"))
    else:
        raise PortRefused(
            f"{name}: some returns of choice_probs are with_lapse(X, params['lapse']) and some are "
            f"not; neither rule applies")
    # The function's name.
    def_start = starts[fn.lineno - 1] + fn.col_offset
    name_at = source.index("choice_probs", def_start)
    edits.append((name_at, name_at + len("choice_probs"), LISTENER))

    out = source
    for s, e, text in sorted(edits, reverse=True):
        out = out[:s] + text + out[e:]
    result = HEADER.format(name=name, rule=rule) + out
    _check_theory(result, name)
    return result


def port_rule(theory_source: str) -> str:
    """'a' or 'b': how a theory file was ported (from its header)."""
    first = theory_source.splitlines()[0] if theory_source else ""
    m = re.search(r"rule \((a|b)\)", first)
    if m:
        return m.group(1)
    raise ValueError("not a ported theory file (no port header)")


def port_file(path: Path) -> str:
    """The theory for one closed model file (`port`)."""
    path = Path(path)
    return port(path.read_text(), name=path.name)


def _check_theory(source: str, name: str) -> None:
    tree = ast.parse(source)
    defined = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if LISTENER not in defined or "choice_probs" in defined:
        raise PortRefused(f"{name}: the port did not give a theory with {LISTENER}() and no choice_probs")
    if any(isinstance(n, ast.Constant) and n.value == LAPSE for n in ast.walk(tree)):
        raise PortRefused(f"{name}: the ported theory still mentions 'lapse'")


@dataclass
class Args:
    models: List[Path]
    """Closed-campaign model files to port."""
    out: Path
    """Directory for the theory files (one per model, same file name)."""


def main(args: Args) -> List[Path]:
    args.out.mkdir(parents=True, exist_ok=True)
    written = []
    for path in args.models:
        target = args.out / path.name
        target.write_text(port_file(path))
        written.append(target)
        print(f"ported {path} -> {target}")
    return written


if __name__ == "__main__":
    main(tyro.cli(Args))
