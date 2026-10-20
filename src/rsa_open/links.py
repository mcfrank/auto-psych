"""Links: from a theory's latent quantities to the likelihood of one response type
(docs/auto_open/PLAN.md §3).

A theory is paradigm-free: it defines ``PARAMS`` and ``listener(params, ctx)``, the listener's
posterior over the objects of one scene. A link is a response process with its own parameters,
shared by every theory on a paradigm and frozen with the paradigm before any data. Theorists
never write links.

`compose` puts a theory and a link together into an ordinary RSA model file (``PARAMS`` and
``choice_probs``), so the existing fitting, cross-validation and loop code runs it unchanged.
The link's parameters are appended to the theory's, so a theory may not use their names
(`LinkClash`).

Only the choice link exists so far; the others in PLAN.md §3 come with their recovery tests.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Dict, List

from src.rsa_open.port import LISTENER


class LinkClash(ValueError):
    """A theory defines a parameter, or a function, that the link defines."""


@dataclass(frozen=True)
class Link:
    name: str
    description: str
    params: Dict[str, str]  # name -> numpyro prior, as source over ``_link_dist``
    imports: str
    body: str  # defines choice_probs(params, ctx) in terms of listener(params, ctx)


CHOICE = Link(
    name="choice",
    description=("Pick one referent: the listener posterior, mixed with a uniform guess with "
                 "probability lapse (the closed campaign's response process)."),
    params={"lapse": "_link_dist.Beta(1.0, 9.0)"},
    imports=("import numpyro.distributions as _link_dist\n"
             "from src.rsa.memo_kit import with_lapse as _link_with_lapse\n"),
    body=("def choice_probs(params, ctx):\n"
          "    return _link_with_lapse(listener(params, ctx), params[\"lapse\"])\n"),
)

LINKS: Dict[str, Link] = {CHOICE.name: CHOICE}


def _theory_params(tree: ast.Module) -> List[str]:
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "PARAMS"):
            if not isinstance(node.value, ast.Dict) or not all(
                    isinstance(k, ast.Constant) and isinstance(k.value, str) for k in node.value.keys):
                raise ValueError("a theory's PARAMS must be a dict literal with string keys")
            return [k.value for k in node.value.keys]
    raise ValueError("a theory must define PARAMS")


def compose(theory_source: str, link: str) -> str:
    """The model file for ``theory_source`` under the named link."""
    if link not in LINKS:
        raise KeyError(f"no link {link!r}; the library has {sorted(LINKS)}")
    spec = LINKS[link]
    tree = ast.parse(theory_source)
    functions = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    if LISTENER not in functions:
        raise ValueError(f"a theory must define {LISTENER}(params, ctx)")
    if "choice_probs" in functions:
        raise LinkClash("a theory may not define choice_probs: the link does")
    clash = sorted(set(_theory_params(tree)) & set(spec.params))
    if clash:
        raise LinkClash(f"theory parameter(s) {', '.join(clash)} belong to the {link!r} link")
    entries = ", ".join(f'"{k}": {v}' for k, v in spec.params.items())
    footer = (f"\n\n# --- link {link!r}, composed by src.rsa_open.links.compose "
              f"(docs/auto_open/PLAN.md §3) ---\n"
              f"{spec.imports}\n"
              f"PARAMS = {{**PARAMS, {entries}}}\n\n\n"
              f"{spec.body}")
    return theory_source.rstrip("\n") + footer
