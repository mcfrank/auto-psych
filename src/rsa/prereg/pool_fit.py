"""Fit models once on the pooled data and freeze their predictions on the design pool.

    python -m src.rsa.prereg.pool_fit --data $WORK_ROOT/finals_pool/pooled.csv \
        --models seed:src/pipelines/outer_loop/projects/rsa_reference/seed_models \
                 promoted:data/rsa/live_seeds/models \
        --cache $WORK_ROOT/finals_pool/.fit_cache --out data/rsa/live/bar_pool_preds.json

Each model in each folder's manifest is fitted with the loop's own fit
(`src.rsa.loop.fitting.loop_fit`, the campaign's settings, no time limit, in a
child pinned to one core) and its posterior-mean choice-class probabilities on
`src.rsa.loop.novelty.plain_pool` are written in the format of
``finals_pool_preds_final.json`` (`src.rsa.prereg.predictions`), labelled
``<label>:<name>``. With the finals' data file and cache the fits are the
finals' fits' twins: same data, same settings, same rule.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List

import tyro

PREDICTIONS_NOTE = (
    "Posterior-mean choice-class probabilities on novelty.plain_pool(), flattened by context in pool order: "
    "context i occupies n_entries[i] entries (one per object); a class's probability sits at its first object's "
    "index and the class's other objects get 0 (src.rsa.fit.class_probs; choice_classes[i] gives each object's "
    "class). Each model fitted once on the pooled data below (src.rsa.loop.novelty.posterior_mean_class_probs, "
    "max 100 draws).")


@dataclass
class Args:
    data: Path
    """The pooled trials (the finals' pooled.csv)."""
    models: List[str]
    """``<label>:<folder>``: every model in the folder's manifest, labelled ``<label>:<name>``."""
    cache: Path
    out: Path
    workers: int = 8
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    seed: int = 0
    dense_mass: bool = True


def main(args: Args) -> dict:
    from src.models.model_manifest import read_manifest_names
    from src.rsa.fit import FitSettings
    from src.rsa.loop.fitting import NO_TIME_LIMIT, describe, loop_fit
    from src.rsa.loop.novelty import plain_pool, pool_digest, pool_record, posterior_mean_class_probs
    from src.rsa.model_file import RSAModel

    entries = []
    for spec in args.models:
        label, sep, folder = spec.partition(":")
        if not sep or not label or not folder:
            raise ValueError(f"--models {spec!r} is not <label>:<folder>")
        entries += [(f"{label}:{name}", name, Path(folder) / f"{name}.py") for name in read_manifest_names(folder)]
    labels = [e[0] for e in entries]
    if len(set(labels)) != len(labels):
        raise ValueError(f"a label repeats among {labels}")
    settings = FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                           seed=args.seed, dense_mass=args.dense_mass)
    pool = plain_pool()

    def run(entry):
        label, name, path = entry
        fitted = loop_fit(path, name, args.data, settings, args.cache, time_limit_sec=NO_TIME_LIMIT)
        probs = posterior_mean_class_probs(RSAModel(path, name=name), fitted, pool)
        print("done", label, describe(fitted), flush=True)
        return label, path, fitted, probs

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(run, entries))
    n_entries = [len(ctx.objects) for ctx in pool]
    out = dict(
        note=PREDICTIONS_NOTE,
        pool=dict(size=len(pool), digest=pool_digest(pool), n_entries=n_entries,
                  choice_classes=[list(ctx.choice_classes()) for ctx in pool],
                  contexts=[pool_record(c) for c in pool]),
        data=dict(path=str(args.data), sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest()),
        fit_settings=dict(vars(settings), refit="loop_fit: one refit at target_accept 0.95 if the first fails "
                                                "the convergence gate"),
        models={},
    )
    for label, path, fitted, probs in results:
        assert len(probs) == sum(n_entries), (label, len(probs))
        out["models"][label] = dict(source=str(path), model_sha256=hashlib.sha256(path.read_bytes()).hexdigest()[:16],
                                    fit=describe(fitted), probs=[round(float(p), 6) for p in probs])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out))
    print("wrote", args.out, flush=True)
    return out


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()  # before JAX starts its thread pools (src.rsa.cpus)
    main(tyro.cli(Args))
