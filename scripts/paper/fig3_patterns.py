"""Figure 3 row (c)'s exploratory search: in which stimuli does the best discovered model's advantage over the best
seed concentrate?

From Figure 3's own condition tables (``scripts.paper.figures.fig3.load``: the same models and predictions as rows
(a) and (b)), each condition's advantage is the discovered model's log-likelihood minus the seed's. For a fixed
list of stimulus features per domain (``FEATURES``, written before looking), the conditions are split by each
feature's value, and each value's *excess* advantage is its advantage minus its trials' share of the domain's
total, with the stimulus-clustered SE of Figure 2 (``cluster_dse``; clusters: the pair, the example set, the
display). The candidates are the ``CANDIDATES`` values with the largest effect size (how much closer the discovered model's
predictions are to people's than the seed's, condition by condition: ``effect``) among those whose excess is at
least ``MIN_Z`` SEs; row (c) shows the candidate per domain with the clearest behavioural reading
(``scripts.paper.figures.fig3.ROW_C``, which must be a candidate). Exploratory: the search
uses all the data the models were fit on.

    uv run python -m scripts.paper.fig3_patterns
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import tyro
from pyprojroot import here

from scripts.paper.figures import Args as FigureArgs
from scripts.paper.figures import fig3
from src.models.clustered_se import cluster_dse


@dataclass
class Args:
    out: Path = here() / "data" / "paper" / "figure_data" / "fig3_patterns.json"
    top: int = 10
    """Rows of each domain's ranking printed."""


def loglik(cond: pd.DataFrame, col: str, domain: str) -> np.ndarray:
    """Each condition's log-likelihood under ``col``'s predictions (reference games: the class's choices)."""
    p = cond[col].clip(1e-9, 1 - 1e-9).to_numpy()
    k = (cond.observed * cond.n).to_numpy()
    if domain == "rsa":
        return k * np.log(p)
    return k * np.log(p) + (cond.n.to_numpy() - k) * np.log(1 - p)


def clusters(cond: pd.DataFrame, domain: str) -> np.ndarray:
    key = cond.index.to_series() if domain == "subjective_randomness" else cond[
        {"number_game": "number_set", "rsa": "display"}[domain]]
    return pd.factorize(key)[0]


# --- the features, fixed before looking ---------------------------------------------------------------

def _runs(q):
    return [len(x) for x in re.findall(r"H+|T+", q)]


def _sr(c):
    a, b = c.pair
    pure = sum(len(set(q)) == 1 for q in (a, b))
    switches = [sum(x != y for x, y in zip(q, q[1:])) for q in (a, b)]
    longest = max(max(_runs(q)) for q in (a, b))
    flip = str.maketrans("HT", "TH")
    sym = any(q == q[::-1] or q == q[::-1].translate(flip) for q in (a, b))
    return {"a pure streak in the pair": {0: "none", 1: "one", 2: "both"}[pure],
            "sequence length": str(len(a)),
            "difference in switches": str(min(abs(switches[0] - switches[1]), 3)).replace("3", "3+"),
            "longest run in the pair": "1-2" if longest <= 2 else "5+" if longest >= 5 else str(longest),
            "a symmetric sequence": "yes" if sym else "no"}


def _ng(c):
    ex = [int(v) for v in str(c.number_set).split(",")]
    t = int(c.target)
    digits, parity = {e % 10 for e in ex}, {e % 2 for e in ex}
    lo, hi = min(ex), max(ex)
    gap = max(0, lo - t, t - hi)
    single = ("n/a" if len(ex) > 1 else "the example" if t == ex[0] else "a multiple" if t % ex[0] == 0
              else "a factor" if ex[0] % t == 0 else "other")
    return {"number of examples": str(len(ex)),
            "the number is an example": "yes" if t in ex else "no",
            "after one example, the number is": single,
            "examples share a last digit; the number": ("n/a" if len(ex) < 2 or len(digits) > 1
                                                        else "keeps it" if t % 10 in digits else "breaks it"),
            "examples share parity; the number": ("n/a" if len(ex) < 2 or len(parity) > 1
                                                  else "keeps it" if t % 2 in parity else "breaks it"),
            "distance outside the examples' range": "inside" if gap == 0 else "1-10" if gap <= 10 else "11+",
            "the number is a multiple of 10": "yes" if t % 10 == 0 else "no"}


def _rsa(c):
    objects, classes, u = c.objects, c.choice_classes, c.utterance
    members = [i for i, k in enumerate(classes) if k == c.cls]
    obj = objects[members[0]]
    if u is None or np.isnan(u):
        return {"query": "no word", "the object": "n/a (no word)", "objects shown": str(len(objects)),
                "objects the word fits": "n/a", "identical copies": "yes" if len(members) > 1 else "no"}
    u = int(u)
    fits = [o for o in objects if o[u]]
    counts = sorted({sum(o) for o in fits})
    role = ("doesn't fit the word" if not obj[u] else "all fitting alike" if len(counts) == 1
            else "fewest features" if sum(obj) == counts[0] else "most features" if sum(obj) == counts[-1]
            else "in between")
    return {"query": "a word", "the object": role, "objects shown": str(len(objects)),
            "objects the word fits": str(len(fits)), "identical copies": "yes" if len(members) > 1 else "no"}


MIN_Z = 2.0  # a value counts only if its excess is at least this many clustered SEs
CANDIDATES = 3  # the reliable values with the largest effects: row (c) shows one of them, chosen by hand
FEATURES: Dict[str, Callable] = {"subjective_randomness": _sr, "number_game": _ng, "rsa": _rsa}


def rank(cond: pd.DataFrame, domain: str) -> pd.DataFrame:
    """One row per (feature, value): its trials, its excess advantage over its share of the total, that excess's
    clustered SE and z, and people's and each model's mean in it."""
    gain = loglik(cond, "best_model", domain) - loglik(cond, "best_seed", domain)
    g = clusters(cond, domain)
    # each condition's trials: reference games' class rows carry their display's n, but only the trials choosing
    # the class enter its log-likelihood
    n = (cond.observed * cond.n if domain == "rsa" else cond.n).to_numpy(dtype=float)
    rate = gain.sum() / n.sum()
    feats = pd.DataFrame([FEATURES[domain](c) for c in cond.itertuples()], index=cond.index)
    rows: List[dict] = []
    for f in feats:
        for v, idx in feats.groupby(f).groups.items():
            if str(v).startswith("n/a"):  # the feature does not apply there
                continue
            m = cond.index.isin(idx)
            excess = gain[m] - rate * n[m]
            se = cluster_dse(excess, np.zeros(m.sum()), pd.factorize(g[m])[0])
            w = cond.n.to_numpy(dtype=float)[m]
            rows.append({"feature": f, "value": v, "trials": int(w.sum()), "excess": float(excess.sum()),
                         "se": se, "z": float(excess.sum() / se) if se > 0 else 0.0,
                         **{k: float(np.average(cond[k][m], weights=w)) for k in ("observed", "best_seed", "best_model")},
                         # effect size: how much closer the discovered model is to people than the seed, condition by
                         # condition (a group's mean would cancel out: a pair's probability is for an arbitrary side)
                         "effect": float(np.average((cond.observed[m] - cond.best_seed[m]).abs()
                                                    - (cond.observed[m] - cond.best_model[m]).abs(), weights=w))})
    r = pd.DataFrame(rows)
    r["share"] = r.trials / cond.n.sum()
    r["candidate"] = False
    top = r[r.z >= MIN_Z].sort_values("effect", ascending=False).head(CANDIDATES).index
    r.loc[top, "candidate"] = True
    return r.sort_values(["candidate", "effect"], ascending=False).reset_index(drop=True)


def main(args: Args) -> Dict[str, list]:
    fa = FigureArgs()
    tables = fig3.load(fa.figure_data, fa.rsa_displays)
    out = {}
    for d in fig3.ORDER:
        r = rank(tables[d], d)
        out[d] = r.to_dict("records")
        print(f"\n== {d}\n" + r.head(args.top).round(3).to_string(index=False), flush=True)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {args.out}", flush=True)
    return out


if __name__ == "__main__":
    main(tyro.cli(Args))
