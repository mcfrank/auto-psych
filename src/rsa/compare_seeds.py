"""Fit memo models to the pragmods forced-choice trials and compare them.

    uv run python -m src.rsa.compare_seeds --out-dir data/rsa/seed_comparison

Writes, under ``--out-dir``:

* ``comparison.csv`` — ``az.compare`` (PSIS-LOO) over the models, with each
  model's LOO reliability verdict and convergence problems;
* ``params.csv`` — posterior mean and 94% interval of every parameter;
* ``cells.csv`` — per display cell (experiment, condition, game, word): the
  observed choice proportions and each model's posterior-mean prediction;
* ``summary.json`` — the above plus each model's Pearson r between predicted
  and observed proportions over cells (the pragmods paper's fit measure).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import tyro

from src.models.model_manifest import read_manifest_names
from src.rsa.dataset import DEFAULT_TRIALS_CSV, load_forced_choice
from src.rsa.fit import FitSettings, RSAFit, compare, fit, posterior_mean_probs
from src.rsa.model_file import RSAModel
from src.runtime.config import PROJECT_ASSETS_DIR

SEED_DIR = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
CELL_KEYS = ["experiment", "condition", "objects", "query", "utterance"]


@dataclass
class Args:
    out_dir: Path
    models_dir: Path = SEED_DIR
    models: Optional[List[str]] = None  # default: every model in the manifest
    trials_csv: Path = DEFAULT_TRIALS_CSV
    experiments: Optional[List[str]] = None  # default: every experiment
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    seed: int = 0
    min_cell_n: int = 10  # cells with fewer trials are left out of the r


def cell_table(trials, predictions: dict) -> pd.DataFrame:
    frame = trials.frame.copy()
    frame["utterance"] = frame["utterance"].fillna(-1).astype(int)
    rows = []
    for key, g in frame.groupby(CELL_KEYS, sort=True):
        idx = g.index.to_numpy()
        n_obj = trials.contexts[idx[0]].shape[0]
        classes = trials.contexts[idx[0]].choice_classes()
        # Identical objects are one choice class (see src.rsa.fit): report the
        # class, at its first object, so the data's arbitrary copy does not count.
        counts = np.bincount([classes[c] for c in g["choice"].astype(int)], minlength=n_obj)
        for obj in sorted(set(classes)):
            row = dict(zip(CELL_KEYS, key))
            row.update(object=obj, n=len(idx), count=int(counts[obj]), observed=counts[obj] / len(idx))
            for name, preds in predictions.items():
                members = [r for r in range(n_obj) if classes[r] == obj]
                row[name] = float(np.mean([preds[i][members].sum() for i in idx]))
            rows.append(row)
    return pd.DataFrame(rows)


def main(args: Args) -> None:
    names = args.models or read_manifest_names(args.models_dir)
    trials = load_forced_choice(args.trials_csv, args.experiments)
    settings = FitSettings(
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        seed=args.seed,
    )
    print(f"{len(trials.contexts)} trials; fitting {names}")
    fits: dict[str, RSAFit] = {}
    predictions = {}
    param_rows = []
    for name in names:
        model = RSAModel(Path(args.models_dir) / f"{name}.py", name=name)
        fitted = fit(model, trials.contexts, trials.choices, settings)
        fits[name] = fitted
        predictions[name] = posterior_mean_probs(model, fitted, trials.contexts)
        for p in fitted.param_names:
            draws = np.asarray(fitted.idata.posterior[p]).ravel()
            param_rows.append(
                dict(model=name, param=p, mean=draws.mean(),
                     lo=np.quantile(draws, 0.03), hi=np.quantile(draws, 0.97))
            )
        print(f"  {name}: converged={fitted.converged} {fitted.convergence_problems}")

    table = compare(fits)
    table["loo_reliable"] = [not fits[n].loo().unreliable for n in table.index]
    table["convergence_problems"] = ["; ".join(fits[n].convergence_problems) for n in table.index]
    cells = cell_table(trials, predictions)
    big = cells[cells["n"] >= args.min_cell_n]
    r = {name: float(np.corrcoef(big["observed"], big[name])[0, 1]) for name in names}

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison.csv")
    pd.DataFrame(param_rows).to_csv(out / "params.csv", index=False)
    cells.to_csv(out / "cells.csv", index=False)
    summary = {
        "n_trials": len(trials.contexts),
        "experiments": args.experiments or sorted(trials.frame["experiment"].unique()),
        "settings": vars(settings),
        "pearson_r_over_cells": r,
        "n_cells_in_r": int(big.groupby(CELL_KEYS).ngroups),
        "comparison": json.loads(table.drop(columns=["convergence_problems"]).to_json(orient="index")),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(table[["rank", "elpd_loo", "elpd_diff", "dse", "weight", "loo_reliable"]])
    print("Pearson r over cells:", {k: round(v, 3) for k, v in r.items()})


if __name__ == "__main__":
    from src.rsa.cpus import pin_main_thread

    pin_main_thread()  # one core per process (src.rsa.cpus)
    main(tyro.cli(Args))
