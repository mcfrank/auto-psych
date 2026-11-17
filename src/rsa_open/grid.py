"""A factorial grid of standard RSA theories (docs/auto_open/RSA_GRID.md; PI 2026-11-15).

Not part of the discovery loop: a basic comparison of textbook RSA forms on all the reference-game data the
campaigns hold. Each grid point is a self-contained theory file in the open campaign's format (`PARAMS`,
`listener`), so the same fits, cross-validation and links score it on every paradigm. Axes:

- **depth** 0-3: the literal listener L0, or a pragmatic listener L1, L2, L3 (Frank & Goodman 2012; each level
  inverts a softmax-rational speaker who simulates the level below);
- **salience** where it enters: ``none``; ``listener`` (the top listener's prior over referents only, the speakers
  it simulates take a uniform one, as `rsa_l1_salience`); ``shared`` (common knowledge: every level's prior, the
  literal listener's included, as `rsa_l1_shared_prior`). At depth 0 the two are the same, so only ``listener``;
- **kind** of salience: ``fc`` (an object's feature count) or ``singleton`` (an object no other object in the scene
  looks like), each with one weight;
- **semantics**: ``bool`` (the literal lexicon) or ``graded`` (a word is true of an object to 1 / its feature
  count: the less of an object a word describes, the less true it is; the open arm's recurring discovery);
- **no utterance cost.** Every utterance is one word, so there is no production cost to model. An earlier version
  had a cost axis, a word's extension share with its own weight outside the rationality; it was a second knob on the
  speaker's preference for specific words and traded off with recursion depth, so the PI removed it (2026-11-17).
  Names keep their ``_nocost`` suffix so that every fit made before stays matched.

A theory's speaker is the format's default, the one derived from its listener (`src.rsa_open.fit.TheoryModel`).
On a prior trial (no word) a theory with salience gives its prior, one without a uniform choice, as the seeds do.
The corners reproduce existing theories exactly: depth 0 / 1 / 2 with nothing else are `literal_listener`,
`rsa_l1`, `rsa_l2`; depth 1 with feature-count salience is `rsa_l1_salience` (``listener``) or
`rsa_l1_shared_prior` (``shared``) with their familiarization weight at 0 (`tests/test_rsa_open_grid.py`).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml

DEPTHS = (0, 1, 2, 3)
SALIENCE = ("none", "listener", "shared")
KINDS = ("fc", "singleton")
SEMANTICS = ("bool", "graded")


@dataclass(frozen=True)
class GridPoint:
    depth: int
    salience: str = "none"
    kind: str = ""
    semantics: str = "bool"

    def __post_init__(self) -> None:
        if self.depth not in DEPTHS or self.salience not in SALIENCE or self.semantics not in SEMANTICS:
            raise ValueError(f"not a grid point: {self}")
        if (self.salience == "none") != (self.kind == ""):
            raise ValueError(f"a salience placement needs a kind, and only then: {self}")
        if self.kind and self.kind not in KINDS:
            raise ValueError(f"unknown salience kind {self.kind!r}")
        if self.depth == 0 and self.salience == "shared":
            raise ValueError(f"at depth 0 salience is the listener's prior; there is no level to share it with: {self}")

    @property
    def name(self) -> str:
        sal = "nosal" if self.salience == "none" else {"listener": "lst", "shared": "shr"}[self.salience] + self.kind
        return f"grid_d{self.depth}_{sal}_{self.semantics}_nocost"  # the suffix keeps earlier fits matched

    @property
    def params(self) -> Dict[str, str]:
        out = {}
        if self.depth >= 1:
            out["alpha"] = "dist.LogNormal(0.0, 1.0)"
        if self.salience != "none":
            out["w_salience"] = "dist.Normal(0.0, 1.0)"
        return out

    def hypothesis(self) -> str:
        level = "the literal listener" if self.depth == 0 else f"a depth-{self.depth} pragmatic listener (RSA)"
        sal = ("no salience prior" if self.salience == "none" else
               f"a salience prior ({self.kind_text()}) " +
               ("in the top listener only" if self.salience == "listener" else "shared by every level"))
        sem = "Boolean word meanings" if self.semantics == "bool" else "graded word meanings (1 / feature count)"
        return f"Grid point: {level} with {sal}, {sem}."

    def kind_text(self) -> str:
        return {"fc": "feature count", "singleton": "singleton objects"}[self.kind]


def all_points() -> List[GridPoint]:
    """Every distinct grid point: 2 x 3 at depth 0, 2 x 5 at each of depths 1-3 (36)."""
    points = [GridPoint(0, sal, kind, sem) for sem in SEMANTICS
              for sal, kind in [("none", "")] + [("listener", k) for k in KINDS]]
    for depth, sem in itertools.product((1, 2, 3), SEMANTICS):
        points += [GridPoint(depth, sal, kind, sem)
                   for sal, kind in [("none", "")] + [(s, k) for s in ("listener", "shared") for k in KINDS]]
    return points


# ---------- the source ----------

def _level(k: int, p: GridPoint) -> str:
    """The memo of level k (0 = the literal listener)."""
    shared = p.salience == "shared"
    top = k == p.depth
    with_prior = shared or (p.salience == "listener" and top)
    if k == 0:
        args = "lex: ..." + (", prior: ..." if with_prior else "")
        wpp = ("vec(prior, r) * " if with_prior else "") + "at(lex, u, r)"
        return (f"@memo\ndef L0[u: UTT, r: OBJ]({args}):\n"
                f"    listener: knows(u)\n    listener: chooses(r in OBJ, wpp={wpp})\n    return Pr[listener.r == r]\n")
    inner_prior = shared  # the level below carries the prior only when it is common knowledge
    inner_args = ("alpha, " if k - 1 >= 1 else "") + "lex" + (", prior" if inner_prior else "")
    args = "alpha, lex: ..." + (", prior: ..." if with_prior else "")
    given = "vec(prior, r)" if with_prior else "1"
    utility = f"exp(alpha * log(L{k - 1}[u, r]({inner_args}) + {{EPS}}))"
    return (f"@memo\ndef L{k}[u: UTT, r: OBJ]({args}):\n"
            f"    listener: thinks[\n"
            f"        speaker: given(r in OBJ, wpp={given}),\n"
            f"        speaker: chooses(u in UTT, wpp=at(lex, u, r) * {utility}),\n"
            f"    ]\n"
            f"    listener: observes [speaker.u] is u\n"
            f"    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])\n"
            f"    return Pr[listener.r == r]\n")


def source(p: GridPoint) -> str:
    """The theory file of grid point ``p``."""
    if p.salience == "listener" and p.depth >= 2:
        # inner levels take no prior, the top one does: write the inner chain without it, then the top level
        levels = [_level(k, GridPoint(p.depth, "none", "", p.semantics)) for k in range(p.depth)]
        levels.append(_level(p.depth, p))
    else:
        levels = [_level(k, p) for k in range(p.depth + 1)]
    imports = ["EPS", "at"] + (["softmax_prior"] if p.salience != "none" else []) \
        + (["vec"] if p.salience != "none" else [])
    params = "".join(f'    "{k}": {v},\n' for k, v in p.params.items())
    lines = [f'"""{p.hypothesis()} Generated by src.rsa_open.grid (not an agent\'s theory)."""', "",
             "import jax.numpy as jnp", "import numpyro.distributions as dist", "from memo import memo", "",
             f"from src.rsa.memo_kit import {', '.join(sorted(set(imports)))}", "",
             "PARAMS = {\n" + params + "}", "", ""]
    body = "\n\n".join(levels)
    helpers = ["def lexicon(ctx):"]
    if p.semantics == "bool":
        helpers.append("    return ctx.lex")
    else:
        helpers.append("    return ctx.lex / jnp.maximum(ctx.feature_count, 1.0)[None, :]")
    if p.salience != "none":
        helpers += ["", "", "def salience(ctx):"]
        if p.kind == "fc":
            helpers.append("    return ctx.feature_count")
        else:
            helpers += ["    diff = ctx.lex[:, :, None] - ctx.lex[:, None, :]",
                        "    alike = jnp.sum(jnp.abs(diff), axis=0) < EPS",
                        "    return jnp.where(jnp.sum(alike, axis=1) == 1, 1.0, 0.0)"]
    call_args = []
    if p.depth >= 1:
        call_args.append('params["alpha"]')
    call_args.append("lexicon(ctx)")
    if p.salience != "none":
        call_args.append("prior")
    listener = ["", "", "def listener(params, ctx):"]
    if p.salience != "none":
        listener.append('    prior = softmax_prior(params["w_salience"] * salience(ctx))')
    listener.append(f"    heard = L{p.depth}({', '.join(call_args)})[ctx.utterance]")
    if p.salience != "none":
        listener.append("    return jnp.where(ctx.is_prior > 0, prior, heard)")
    else:
        listener += ["    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])",
                     "    return jnp.where(ctx.is_prior > 0, uniform, heard)"]
    return "\n".join(lines) + body + "\n\n" + "\n".join(helpers) + "\n" + "\n".join(listener) + "\n"


def write_grid(out_dir: Path, points: List[GridPoint] = None) -> Path:
    """Every grid point's theory file and a `models_manifest.yaml` (name, rationale, axes)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for p in points or all_points():
        (out_dir / f"{p.name}.py").write_text(source(p), encoding="utf-8")
        entries.append(dict(name=p.name, rationale=p.hypothesis(), depth=p.depth, salience=p.salience,
                            kind=p.kind or None, semantics=p.semantics))
    (out_dir / "models_manifest.yaml").write_text(yaml.safe_dump({"models": entries}, sort_keys=False))
    return out_dir


if __name__ == "__main__":
    import sys

    print(write_grid(Path(sys.argv[1] if len(sys.argv) > 1 else "data/rsa_open/grid")))
