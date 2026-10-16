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
from typing import Callable, Dict, List, Literal, Tuple

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


class Missing(Exception):
    """A round needs models whose CV is not saved yet (`plan`)."""

    def __init__(self, sets: List[Tuple[str, ...]]):
        super().__init__(f"{len(sets)} models need their CV")
        self.sets = sets


@dataclass
class Args:
    data: Path
    """The campaign's pooled data (the finals' pooled.csv): never the pre-registered experiment's."""
    out: Path
    cache: Path
    mode: Literal["run", "plan", "worker"] = "run"
    """``run``: the whole elimination in this process (one node). ``plan``: replay the rule
    over the saved CVs; write ``next.json`` with the models the next round still needs, or,
    when none is missing, the result (as ``run``). ``worker``: compute the CV of model
    ``task`` of ``next.json``. Plan and workers let a round's models run as a Slurm array
    across nodes (`scripts/rsa/slurm/consolidate_array.sbatch`); the rule is the same."""
    task: int = -1
    """worker: the index into ``next.json``'s models (the array task id)."""
    workers: int = 8
    """run: fits at once (each fit is one process on one core). worker: the folds run at once."""
    folds: int = 5
    fold_seed: int = 0
    threshold_dse: float = THRESHOLD_DSE
    start: List[str] = field(default_factory=lambda: list(MECHANISMS))
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    seed: int = 0


def _settings(args: Args):
    from src.rsa.fit import FitSettings

    return FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                       seed=args.seed, dense_mass=True)


def read_folds(data: Path, fold_dir: Path, k: int, seed: int):
    """The folds `make_folds` wrote, without writing anything (workers run side by side)."""
    import pandas as pd

    from src.rsa.dataset import load_forced_choice
    from src.rsa.loop.cv import Folds, assign_folds
    from src.rsa.split import unit_keys

    meta = json.loads((Path(fold_dir) / "folds.json").read_text())
    frame = load_forced_choice(data).frame
    if (meta["k"], meta["seed"], meta["n_trials"]) != (k, seed, len(frame)):
        raise ValueError(f"{fold_dir}/folds.json records other folds ({meta}) than k={k}, seed={seed}, "
                         f"{len(frame)} trials")
    paths = [Path(fold_dir) / f"fold_{f}_train.csv" for f in range(k)]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"fold files missing: {missing} (the plan step writes them)")
    return Folds(k=k, seed=seed, fold=assign_folds(frame, k, seed), units=pd.factorize(unit_keys(frame))[0],
                 train_paths=paths)


def _saved(out: Path, ms: Tuple[str, ...]):
    path = out / "cv" / f"{variant_name(ms)}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return Scored(z["pointwise"], bool(z["converged"]))


def _compute(args: Args, out: Path, folds, ms: Tuple[str, ...], workers: int) -> Scored:
    from src.rsa.loop.cv import cv_pointwise
    from src.rsa.loop.fitting import NO_TIME_LIMIT

    done = _saved(out, ms)
    if done is not None:
        return done
    name, path = write_variant(ms, out / "models")
    r = cv_pointwise(path, name, args.data, folds, _settings(args), args.cache, time_limit_sec=NO_TIME_LIMIT,
                     workers=workers)
    saved = out / "cv" / f"{name}.npz"
    tmp = saved.with_name(f"{name}.tmp.npz")
    np.savez(tmp, pointwise=r.pointwise, converged=r.converged)
    tmp.replace(saved)
    print(f"  {name}: ELPD-CV {r.elpd:.1f}{'' if r.converged else ' (UNCONVERGED)'}", flush=True)
    return Scored(r.pointwise, r.converged)


def _finish(args: Args, out: Path, record: dict) -> dict:
    lean_name, lean_path = write_variant(record["lean"]["mechanisms"], out / "models")
    (out / "lean").mkdir(exist_ok=True)
    shutil.copyfile(lean_path, out / "lean" / f"{lean_name}.py")
    (out / "lean" / "models_manifest.yaml").write_text(f"models:\n- name: {lean_name}\n")
    record.update(data=str(args.data), data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                  folds=dict(k=args.folds, seed=args.fold_seed), fit_settings=vars(_settings(args)),
                  lean_file_sha256=hashlib.sha256(lean_path.read_bytes()).hexdigest())
    (out / "elimination.json").write_text(json.dumps(record, indent=1))
    for r in record["rounds"]:
        print(f"[{r['model']}] ELPD-CV {r['elpd_cv']:.1f}: {r['decision']}", flush=True)
    print(f"lean model: {lean_name} ({len(record['lean']['mechanisms'])} mechanisms)", flush=True)
    return record


def plan(args: Args) -> dict:
    """Replay the rule over the saved CVs: the next round's missing models, or the result."""
    from src.rsa.loop.cv import make_folds

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cv").mkdir(exist_ok=True)
    folds = make_folds(args.data, out / "folds", k=args.folds, seed=args.fold_seed)

    def score(sets):
        found = {s: _saved(out, s) for s in sets}
        missing = [s for s, v in found.items() if v is None]
        if missing:
            raise Missing(missing)
        return found

    nxt = out / "next.json"
    before = json.loads(nxt.read_text()) if nxt.exists() else None
    try:
        record = eliminate(tuple(args.start), score, folds.units, args.threshold_dse)
    except Missing as m:
        names = [variant_name(s) for s in m.sets]
        for s in m.sets:
            write_variant(s, out / "models")  # written once here, read by the workers
        attempt = before["attempt"] + 1 if before and before.get("names") == names else 1
        doc = dict(status="need", sets=[list(s) for s in m.sets], names=names, attempt=attempt)
        nxt.write_text(json.dumps(doc, indent=1))
        print(f"[plan] {len(names)} models to score (attempt {attempt}): {', '.join(names)}", flush=True)
        return doc
    _finish(args, out, record)
    doc = dict(status="done", lean=record["lean"])
    nxt.write_text(json.dumps(doc, indent=1))
    return doc


def worker(args: Args) -> Scored:
    """The CV of model ``task`` of ``next.json`` (its folds side by side)."""
    out = Path(args.out)
    doc = json.loads((out / "next.json").read_text())
    if doc["status"] != "need" or not 0 <= args.task < len(doc["sets"]):
        raise ValueError(f"task {args.task} is not one of next.json's {len(doc.get('sets', []))} models")
    folds = read_folds(args.data, out / "folds", args.folds, args.fold_seed)
    return _compute(args, out, folds, canonical(doc["sets"][args.task]), workers=args.workers)


def main(args: Args):
    if args.mode == "plan":
        return plan(args)
    if args.mode == "worker":
        return worker(args)
    from src.rsa.loop.cv import make_folds

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "cv").mkdir(exist_ok=True)
    folds = make_folds(args.data, out / "folds", k=args.folds, seed=args.fold_seed)

    def score(sets: List[Tuple[str, ...]]) -> Dict[Tuple[str, ...], Scored]:
        with ThreadPoolExecutor(max_workers=max(1, args.workers // args.folds)) as pool:
            return dict(zip(sets, pool.map(lambda ms: _compute(args, out, folds, ms, args.folds), sets)))

    return _finish(args, out, eliminate(tuple(args.start), score, folds.units, args.threshold_dse))


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()
    main(tyro.cli(Args))
