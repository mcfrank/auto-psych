"""Backward elimination over the consolidated superset, by grouped cross-validation.

    python -m src.rsa.consolidate.eliminate --data <pooled.csv> --out <dir> --cache <fit cache> --workers 32

The rule (docs/auto_rsa/CONSOLIDATION.md, fixed before any fit):

1. Start from every mechanism on (`src.rsa.consolidate.superset`).
2. Each round, score the current model and every model with one mechanism
   off by ELPD from grouped CV (`src.rsa.loop.cv`: the loop's folds of whole
   displays and conditions, the loop's fit rule; out-of-fold lpd per trial).
3. Of the converged one-off models, take the one with the highest ELPD-CV.
   Its cost is the current model's ELPD-CV minus its own; its SE is clustered
   by unit. If the cost is at most 0 (a gain, or no change) or below
   ``threshold_dse`` (2) SEs, that mechanism goes, and the next round starts from the smaller
   model.
4. Stop when every removal costs at least ``threshold_dse`` SEs (or none
   converges): the current model is the lean model.

Every model's CV is saved (``cv/<name>.npz``) and reused, so a killed run
resumes. Writes ``elimination.json`` (every round: each candidate's ELPD-CV,
cost, SE and convergence; the decision) and the lean model file under
``lean/``. The pre-registered experiment's data are never read here.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import tyro

from src.rsa.consolidate.superset import MECHANISMS
from src.rsa.consolidate.variants import canonical, variant_name, write_variant

THRESHOLD_DSE = 2.0


@dataclass
class Scored:
    pointwise: np.ndarray
    converged: bool

    @property
    def elpd(self) -> float:
        return float(self.pointwise.sum())


def eliminate(start: Tuple[str, ...], score: Callable[[List[Tuple[str, ...]]], Dict[Tuple[str, ...], Scored]],
              units: np.ndarray, threshold_dse: float = THRESHOLD_DSE) -> dict:
    """The rule above. ``score`` maps mechanism sets to their CV (in parallel, cached)."""
    from src.models.clustered_se import cluster_dse

    current = canonical(start)
    rounds = []
    while True:
        drops = [canonical(set(current) - {m}) for m in current]
        scored = score([current] + drops)
        cur = scored[current]
        rows = []
        for m, d in zip(current, drops):
            s = scored[d]
            cost = cur.elpd - s.elpd
            se = float(cluster_dse(cur.pointwise, s.pointwise, units))
            rows.append(dict(mechanism=m, model=variant_name(d), elpd_cv=s.elpd, cost=float(cost), se=se,
                             cost_in_se=float(cost / se) if se > 0 else (0.0 if cost == 0 else float(np.sign(cost)) * np.inf),
                             converged=s.converged))
        rows.sort(key=lambda r: r["cost"])
        record = dict(model=variant_name(current), mechanisms=list(current), elpd_cv=cur.elpd,
                      converged=cur.converged, candidates=rows)
        best = next((r for r in rows if r["converged"]), None)
        if best is not None and (best["cost"] <= 0 or best["cost"] < threshold_dse * best["se"]):
            record["decision"] = f"drop {best['mechanism']}"
            rounds.append(record)
            current = canonical(set(current) - {best["mechanism"]})
            if not current:
                rounds.append(dict(model=variant_name(current), mechanisms=[], elpd_cv=score([current])[current].elpd,
                                   decision="stop: nothing left"))
                break
            continue
        record["decision"] = ("stop: no one-off model converged" if best is None else
                              f"stop: the cheapest removal ({best['mechanism']}) costs {best['cost']:.1f} nats, "
                              f"{best['cost_in_se']:.2f} SE >= {threshold_dse}")
        rounds.append(record)
        break
    return dict(rule=dict(threshold_dse=threshold_dse, start=list(canonical(start))),
                rounds=rounds, lean=dict(model=variant_name(current), mechanisms=list(current)))


@dataclass
class Args:
    data: Path
    """The campaign's pooled data (the finals' pooled.csv): never the pre-registered experiment's."""
    out: Path
    cache: Path
    workers: int = 8
    """Fits at once (each fit is one process on one core)."""
    folds: int = 5
    fold_seed: int = 0
    threshold_dse: float = THRESHOLD_DSE
    start: List[str] = field(default_factory=lambda: list(MECHANISMS))
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    seed: int = 0


def main(args: Args) -> dict:
    from src.rsa.fit import FitSettings
    from src.rsa.loop.cv import cv_pointwise, make_folds
    from src.rsa.loop.fitting import NO_TIME_LIMIT

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    settings = FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                           seed=args.seed, dense_mass=True)
    folds = make_folds(args.data, out / "folds", k=args.folds, seed=args.fold_seed)
    (out / "cv").mkdir(exist_ok=True)

    def one(ms: Tuple[str, ...]) -> Scored:
        name, path = write_variant(ms, out / "models")
        saved = out / "cv" / f"{name}.npz"
        if saved.exists():
            z = np.load(saved)
            return Scored(z["pointwise"], bool(z["converged"]))
        r = cv_pointwise(path, name, args.data, folds, settings, args.cache, time_limit_sec=NO_TIME_LIMIT,
                         workers=args.folds)
        np.savez(saved.with_suffix(".tmp.npz"), pointwise=r.pointwise, converged=r.converged)
        saved.with_suffix(".tmp.npz").replace(saved)
        print(f"  {name}: ELPD-CV {r.elpd:.1f}{'' if r.converged else ' (UNCONVERGED)'}", flush=True)
        return Scored(r.pointwise, r.converged)

    def score(sets: List[Tuple[str, ...]]) -> Dict[Tuple[str, ...], Scored]:
        with ThreadPoolExecutor(max_workers=max(1, args.workers // args.folds)) as pool:
            return dict(zip(sets, pool.map(one, sets)))

    record = eliminate(tuple(args.start), score, folds.units, args.threshold_dse)
    lean_name, lean_path = write_variant(record["lean"]["mechanisms"], out / "models")
    (out / "lean").mkdir(exist_ok=True)
    shutil.copyfile(lean_path, out / "lean" / f"{lean_name}.py")
    (out / "lean" / "models_manifest.yaml").write_text(f"models:\n- name: {lean_name}\n")
    record.update(data=str(args.data), data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                  folds=dict(k=args.folds, seed=args.fold_seed), fit_settings=vars(settings),
                  lean_file_sha256=hashlib.sha256(lean_path.read_bytes()).hexdigest())
    (out / "elimination.json").write_text(json.dumps(record, indent=1))
    for r in record["rounds"]:
        print(f"[{r['model']}] ELPD-CV {r['elpd_cv']:.1f}: {r['decision']}", flush=True)
    print(f"lean model: {lean_name} ({len(record['lean']['mechanisms'])} mechanisms)", flush=True)
    return record


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()
    main(tyro.cli(Args))
