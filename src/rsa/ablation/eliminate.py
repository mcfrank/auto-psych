"""Backward elimination over chain 0's final, by the consolidation's grouped CV.

    python -m src.rsa.ablation.eliminate --data <pooled.csv> --out <dir> --cache <fit cache> --mode plan
    python -m src.rsa.ablation.eliminate --data <pooled.csv> --out <dir> --cache <fit cache> --mode worker --task <i>

The rule is the sr-oct26 ablation's, unchanged (`src.ablation.eliminate.eliminate`): only removals
that leave a valid model are candidates; the cheapest converged removal goes while it costs less
than 2 clustered SEs; the starting model must converge on every fold. What is RSA's:

- the final: chain 0's, the best of the three on the pre-registered experiment
  (`src.rsa.ablation.spec_c0`, `final_c0.py`);
- the data: the campaign's pooled data, as the consolidation (never the pre-registered experiment's);
- the CV: the consolidation's folds of whole displays and conditions (`src.rsa.loop.cv`) and its
  clusters (``Folds.units``) for the SE of a difference;
- the fits: numpyro (`src.rsa.fit`), the loop's fit rule (dense NUTS, one refit at target_accept
  0.95), every chain started at the best optimum with the mode guard (``init="map"``,
  docs/auto_rsa/CONSOLIDATION.md §6).

``plan`` replays the rule over the saved CVs and writes ``next.json`` with the models the next
round still needs (``cv/<model>.npz``), or, when none is missing, ``elimination.json``,
``summary.md`` and ``lean/``. ``worker`` computes the 5-fold CV of model ``task`` of ``next.json``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import tyro

from src.ablation import eliminate as E
from src.rsa.ablation import spec_c0
from src.rsa.ablation.variants import VARIANTS as V

FINAL = "c0"


@dataclass
class Args:
    data: Path
    """The campaign's pooled data (the consolidation's pooled.csv): never the pre-registered experiment's."""
    out: Path
    cache: Path
    mode: Literal["plan", "worker"] = "plan"
    task: int = -1
    """worker: the index into ``next.json``'s models (the array task id)."""
    workers: int = 5
    """worker: the folds fitted at once."""
    folds: int = 5
    fold_seed: int = 0
    threshold_dse: float = E.THRESHOLD_DSE
    start: Optional[List[str]] = None
    """The mechanisms the elimination starts from (default: every one, the final itself)."""
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    seed: int = 0
    init: Literal["uniform", "map"] = "map"


def _settings(args: Args):
    from src.rsa.fit import FitSettings

    return FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                       seed=args.seed, dense_mass=True, init=args.init)


def _config(args: Args) -> dict:
    import hashlib

    return dict(final=FINAL, data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                folds=args.folds, fold_seed=args.fold_seed, threshold_dse=args.threshold_dse,
                start=list(V.canonical(FINAL, args.start or spec_c0.MECHANISMS)), fit=asdict(_settings(args)))


def _check_config(args: Args, write: bool) -> None:
    """Every CV under ``out`` was made with one configuration (``config.json``)."""
    path = Path(args.out) / "config.json"
    mine = _config(args)
    if not path.exists():
        if not write:
            raise FileNotFoundError(f"{path} is missing (the plan step writes it)")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(mine, indent=1) + "\n")
        return
    recorded = json.loads(path.read_text())
    if recorded != mine:
        diff = {k: (recorded.get(k), mine.get(k)) for k in set(recorded) | set(mine) if recorded.get(k) != mine.get(k)}
        raise ValueError(f"{args.out} was made with another configuration (recorded, now): {diff}; use a fresh --out")


def _saved(out: Path, name: str) -> Optional[E.Scored]:
    path = out / "cv" / f"{name}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return E.Scored(z["pointwise"], bool(z["converged"]))


def plan(args: Args) -> dict:
    """Replay the rule over the saved CVs: the next round's missing models, or the result."""
    from src.rsa.loop.cv import make_folds

    out = Path(args.out)
    _check_config(args, write=True)
    (out / "cv").mkdir(exist_ok=True)
    folds = make_folds(args.data, out / "folds", k=args.folds, seed=args.fold_seed)

    def score(sets):
        found = {s: _saved(out, V.variant_name(FINAL, s)) for s in sets}
        missing = [s for s, v in found.items() if v is None]
        if missing:
            raise E.Missing(missing)
        return found

    nxt = out / "next.json"
    before = json.loads(nxt.read_text()) if nxt.exists() else None
    try:
        record = E.eliminate(V, FINAL, _config(args)["start"], score, folds.units, args.threshold_dse)
    except E.Missing as m:
        names = [V.write_variant(FINAL, s, out / "models")[0] for s in m.sets]  # written once here, read by workers
        attempt = before["attempt"] + 1 if before and before.get("names") == names else 1
        doc = dict(status="need", sets=[list(s) for s in m.sets], names=names, attempt=attempt)
        nxt.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"[plan] {len(names)} models to score (attempt {attempt}): {', '.join(names)}", flush=True)
        return doc
    _finish(args, out, record, n_trials=len(folds.fold))
    doc = dict(status="done", lean=record["lean"])
    nxt.write_text(json.dumps(doc, indent=1) + "\n")
    return doc


def _finish(args: Args, out: Path, record: dict, n_trials: int) -> None:
    import hashlib

    name, path = V.write_variant(FINAL, record["lean"]["mechanisms"], out / "models")
    (out / "lean").mkdir(exist_ok=True)
    shutil.copyfile(path, out / "lean" / f"{name}.py")
    (out / "lean" / "models_manifest.yaml").write_text(f"models:\n  - name: {name}\n")
    record.update(config=_config(args), n_trials=n_trials,
                  lean_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (out / "elimination.json").write_text(json.dumps(record, indent=1) + "\n")
    summary = E.summary_markdown(V, record)
    (out / "summary.md").write_text(summary)
    print(summary, flush=True)


def worker(args: Args) -> E.Scored:
    """The 5-fold CV of model ``task`` of ``next.json`` (its folds side by side)."""
    from src.rsa.consolidate.eliminate import read_folds
    from src.rsa.loop.cv import cv_pointwise
    from src.rsa.loop.fitting import NO_TIME_LIMIT

    out = Path(args.out)
    _check_config(args, write=False)
    doc = json.loads((out / "next.json").read_text())
    if doc["status"] != "need" or not 0 <= args.task < len(doc["names"]):
        raise ValueError(f"task {args.task} is not one of next.json's {len(doc.get('names', []))} models")
    name = doc["names"][args.task]
    done = _saved(out, name)
    if done is not None:
        return done
    path = out / "models" / f"{name}.py"
    if V.variant_name(FINAL, doc["sets"][args.task]) != name or not path.exists():
        raise ValueError(f"next.json's task {args.task} ({name}) has no model file of that name under {out / 'models'}")
    folds = read_folds(args.data, out / "folds", args.folds, args.fold_seed)
    r = cv_pointwise(path, name, args.data, folds, _settings(args), args.cache, time_limit_sec=NO_TIME_LIMIT,
                     workers=args.workers)
    tmp = out / "cv" / f"{name}.tmp.npz"
    np.savez(tmp, pointwise=r.pointwise, converged=r.converged)
    tmp.replace(out / "cv" / f"{name}.npz")
    print(f"  {name}: ELPD-CV {r.elpd:.1f}{'' if r.converged else ' (UNCONVERGED)'}", flush=True)
    return E.Scored(r.pointwise, r.converged)


def main(args: Args):
    return plan(args) if args.mode == "plan" else worker(args)


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()
    main(tyro.cli(Args))
