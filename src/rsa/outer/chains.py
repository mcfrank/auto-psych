"""Split the promoted seeds across the live campaign's chains (PI 2026-10-09).

    uv run python -m src.rsa.outer.chains --promoted data/rsa/live_seeds/models \
        --data <all existing trials> --cache <fit cache> --out data/rsa/live_seeds/chains

The rule, fixed before any live data: every chain gets the reference model
(``rsa_l2``, the baseline every claim is measured against); the other promoted
seeds are dealt into ``n_chains`` chains of sizes differing by at most one,
choosing, among all such splits, the one whose closest pair of models within a
chain is farthest apart (RMSE of posterior-mean choice-class probabilities on
the design pool, the displays the live experiments can show), ties broken by
the larger mean within-chain distance, then by name. Near-twins on the design
pool (models that differ only in terms plain displays cannot reach) end up in
different chains, where each can be tested against models it actually
disagrees with, and the chains start from different hypotheses rather than
three copies of one. Every chain is still scored against all promoted seeds.

Writes ``<out>/chain_<k>/`` (the chain's model files and a manifest, the
promoted manifest's entries in its order) and ``<out>/chains.json``.
"""

from __future__ import annotations

import itertools
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import tyro
import yaml

from src.rsa.fit import FitSettings
from src.rsa.outer.ground_truth import pool_predictions

REFERENCE = "rsa_l2"


@dataclass
class Args:
    promoted: Path
    data: Path
    cache: Path
    out: Path
    n_chains: int = 3
    reference: str = REFERENCE
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    fit_seed: int = 0


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _score(groups: Sequence[Sequence[str]], dist: Dict[str, Dict[str, float]]):
    pairs = [dist[a][b] for g in groups for a, b in itertools.combinations(g, 2)]
    return (min(pairs), float(np.mean(pairs))) if pairs else (np.inf, 0.0)


def split(preds: Dict[str, np.ndarray], n_chains: int, reference: str = REFERENCE) -> List[List[str]]:
    """The chains, each a sorted list of model names with ``reference`` last."""
    if reference not in preds:
        raise ValueError(f"the reference model {reference!r} has no predictions")
    names = sorted(n for n in preds if n != reference)
    if len(names) < n_chains:
        raise ValueError(f"{len(names)} models besides {reference} cannot fill {n_chains} chains")
    dist = {a: {b: _rmse(preds[a], preds[b]) for b in names} for a in names}
    lo, extra = divmod(len(names), n_chains)
    sizes = sorted([lo + 1] * extra + [lo] * (n_chains - extra))
    best = None
    # Chain labels are interchangeable: the first name always opens chain 0,
    # and each later name may open at most the next empty chain.
    def deal(i, groups):
        nonlocal best
        if i == len(names):
            if sorted(len(g) for g in groups) != sizes:
                return
            canon = sorted(tuple(g) for g in groups)
            score = _score(groups, dist)
            if best is None or score > best[0] or (score == best[0] and canon < best[1]):
                best = (score, canon)
            return
        opened = sum(1 for g in groups if g)
        for k in range(min(opened + 1, n_chains)):
            if len(groups[k]) < sizes[-1]:
                groups[k].append(names[i])
                deal(i + 1, groups)
                groups[k].pop()

    deal(0, [[] for _ in range(n_chains)])
    return [list(g) + [reference] for g in best[1]]


def write_chains(promoted: Path, chains: List[List[str]], out: Path) -> List[Path]:
    entries = {e["name"]: e for e in yaml.safe_load((Path(promoted) / "models_manifest.yaml").read_text())["models"]}
    dirs = []
    for k, chain in enumerate(chains):
        d = Path(out) / f"chain_{k}"
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for name in chain:
            shutil.copyfile(Path(promoted) / f"{name}.py", d / f"{name}.py")
        (d / "models_manifest.yaml").write_text(
            yaml.safe_dump({"models": [entries[n] for n in chain]}, sort_keys=False))
        dirs.append(d)
    return dirs


def main(args: Args) -> dict:
    settings = FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                           seed=args.fit_seed)
    preds = pool_predictions(args.promoted, args.data, args.cache, settings)
    chains = split(preds, args.n_chains, args.reference)
    write_chains(args.promoted, chains, args.out)
    others = sorted(n for n in preds if n != args.reference)
    dist = {a: {b: _rmse(preds[a], preds[b]) for b in others} for a in others}
    within = [[dist[a][b] for a, b in itertools.combinations(c[:-1], 2)] for c in chains]
    record = dict(
        rule=__doc__.split("The rule, fixed before any live data: ", 1)[1].split("\n\nWrites", 1)[0],
        reference=args.reference, n_chains=args.n_chains,
        distance="RMSE of posterior-mean choice-class probabilities over the design pool",
        chains=[dict(chain=k, models=c, min_within_rmse=min(w) if w else None,
                     mean_within_rmse=float(np.mean(w)) if w else None) for k, c, w in zip(range(len(chains)), chains, within)],
        min_rmse_all_pairs=min(dist[a][b] for a, b in itertools.combinations(others, 2)),
        design_pool_rmse=dist,
    )
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "chains.json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "design_pool_rmse"}, indent=1))
    return record


if __name__ == "__main__":
    main(tyro.cli(Args))
