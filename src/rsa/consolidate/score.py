"""Score the lean model prospectively on the pre-registered experiment's data.

    python -m src.rsa.consolidate.score --train <pooled.csv> --test <prereg modelled.csv> \
        --cache <fit cache> --models lean=<lean.py> full=<superset.py> c0=<final.py> ... \
        --reference c0 --out score.json

Each model is fitted on ``train`` (the campaign's pooled data, as the frozen
predictions were; `src.rsa.loop.fitting.loop_fit`, no time limit) and scored
on ``test`` by log predictive density (`src.rsa.evaluate_heldout.heldout_lpd`),
nothing refitted. Each model's difference from ``reference`` carries an SE
clustered by display (the pre-registration's convention). Writes per model:
its parameter count, convergence, lpd, and the difference from the reference
overall and on each half of the design (EIG, random). Trial data stay where
they are; the output holds only sums.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal

import numpy as np
import pandas as pd
import tyro

HALVES = {"eig": "eig_", "random": "rand_"}


@dataclass
class Args:
    train: Path
    test: Path
    cache: Path
    models: List[str]
    """``<label>=<model file>``."""
    reference: str
    out: Path
    workers: int = 8
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    seed: int = 0
    init: Literal["uniform", "map"] = "uniform"
    """Where each chain starts (`src.rsa.fit.FitSettings.init`). The recorded 2026-10-17 run used
    "uniform"; "map" starts every chain in the full superset's main mode, not its minor one."""


def versus(a: np.ndarray, b: np.ndarray, units: np.ndarray) -> dict:
    from src.models.clustered_se import cluster_dse

    diff = float(a.sum() - b.sum())
    se = float(cluster_dse(a, b, units))
    return dict(diff=diff, se=se, z=diff / se if se > 0 else 0.0)


def main(args: Args) -> dict:
    from src.rsa.dataset import load_forced_choice
    from src.rsa.evaluate_heldout import heldout_lpd
    from src.rsa.fit import FitSettings
    from src.rsa.loop.fitting import NO_TIME_LIMIT, describe, loop_fit
    from src.rsa.model_file import RSAModel

    entries = []
    for spec in args.models:
        label, sep, path = spec.partition("=")
        if not sep:
            raise ValueError(f"--models {spec!r} is not <label>=<file>")
        entries.append((label, Path(path)))
    labels = [e[0] for e in entries]
    if args.reference not in labels or len(set(labels)) != len(labels):
        raise ValueError(f"labels {labels} must be distinct and include the reference {args.reference!r}")
    test = load_forced_choice(args.test)
    conditions = test.frame["condition"].astype(str).to_numpy()
    if (conditions == "catch").any():
        raise ValueError(f"{args.test} has catch trials: score the modelled trials")
    units = pd.factorize(conditions)[0]
    settings = FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                           seed=args.seed, dense_mass=True, init=args.init)

    def one(entry):
        label, path = entry
        model = RSAModel(path, name=path.stem)
        fitted = loop_fit(path, path.stem, args.train, settings, args.cache, time_limit_sec=NO_TIME_LIMIT)
        lpd = heldout_lpd(model, fitted, test.contexts, test.choices)
        print(f"  {label} ({path.stem}): lpd {lpd.sum():.1f}", flush=True)
        return label, dict(model=path.stem, n_params=len(model.params), fit=describe(fitted),
                           sha256=hashlib.sha256(path.read_bytes()).hexdigest()), lpd

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(one, entries))
    lpd = {label: v for label, _, v in results}
    ref = lpd[args.reference]
    models = {}
    for label, info, v in results:
        row = dict(info, lpd=float(v.sum()))
        if label != args.reference:
            row["vs_reference"] = versus(v, ref, units)
            row["vs_reference_by_half"] = {}
            for half, prefix in HALVES.items():
                keep = np.char.startswith(conditions.astype(str), prefix)
                if keep.any():
                    row["vs_reference_by_half"][half] = versus(v[keep], ref[keep], pd.factorize(conditions[keep])[0])
        models[label] = row
    out = dict(reference=args.reference, n_trials=len(conditions), n_displays=int(units.max() + 1),
               train_sha256=hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
               test_sha256=hashlib.sha256(Path(args.test).read_bytes()).hexdigest(),
               models=dict(sorted(models.items(), key=lambda kv: -kv[1]["lpd"])))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    for label, row in out["models"].items():
        vs = row.get("vs_reference")
        print(f"{label:10s} {row['n_params']:2d} params  lpd {row['lpd']:.1f}"
              + (f"  vs {args.reference}: {vs['diff']:+.1f} ± {vs['se']:.1f}" if vs else "  (reference)"), flush=True)
    return out


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()
    main(tyro.cli(Args))
