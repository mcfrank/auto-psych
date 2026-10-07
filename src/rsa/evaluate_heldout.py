"""Score models on held-out conditions: did the loop find models that generalise?

    uv run python -m src.rsa.evaluate_heldout --loop-dir data/rsa/loop_run1 \
        --test data/rsa/split/test.csv --out-dir data/rsa/loop_run1/heldout

Every model (the loop's final live set, its pruned models and the seeds) is
fitted to the loop's training trials (``<loop-dir>/responses.csv``; its fit
cache is reused) and scored on the held-out trials by the log posterior
predictive density of each observed choice,

    lpd_i = log( mean over posterior draws of P(choice_i | display_i, params) ),

with the choice observed as its class of identical objects, as in the fit.
The table gives each model's total and per-trial lpd, overall and per
source, and its difference from the best seed, with a standard error that
treats each held-out unit (src.rsa.split) as one observation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import tyro

from src.models.clustered_se import cluster_dse
from src.models.model_manifest import read_manifest_names
from src.rsa.context import Context, group_by_shape
from src.rsa.dataset import load_forced_choice
from src.rsa.fit import FitSettings, RSAFit, class_probs, prepare
from src.rsa.loop.fitting import fit_cached
from src.rsa.model_file import RSAModel
from src.rsa.split import unit_keys
from src.runtime.config import PROJECT_ASSETS_DIR

SEED_DIR = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
MAX_DRAWS = 400


def heldout_lpd(model: RSAModel, fitted: RSAFit, contexts: Sequence[Context],
                choices: Sequence[int], max_draws: int = MAX_DRAWS) -> np.ndarray:
    """Pointwise log posterior predictive density of the observed choices."""
    groups, grouped_choices, inverse, _ = prepare(contexts, choices)
    post = fitted.idata.posterior
    flat = {k: np.asarray(post[k]).reshape(-1) for k in fitted.param_names}
    n = len(next(iter(flat.values())))
    take = np.linspace(0, n - 1, min(max_draws, n)).astype(int)
    out, start = [], 0
    for group in groups:
        rows = np.arange(len(group.indices))
        chosen = grouped_choices[start:start + len(rows)]
        start += len(rows)
        acc = np.zeros(len(rows))
        for i in take:
            p = np.asarray(class_probs(model.group_probs({k: v[i] for k, v in flat.items()}, group),
                                       group.classes))
            acc += p[rows, chosen]
        mean = acc / len(take)
        if np.any(mean <= 0):
            raise ValueError(f"{model.name} gives held-out choices probability 0")
        out.append(np.log(mean))
    return np.concatenate(out)[inverse]


@dataclass
class Args:
    loop_dir: Path
    """A finished RSA loop run (responses.csv, models/, models/pruned/, .fit_cache/)."""
    test: Path
    """Held-out trials (src.rsa.split's test.csv)."""
    out_dir: Path
    seed_models: Path = SEED_DIR
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    include_pruned: bool = True
    extra_models: List[Path] = field(default_factory=list)
    """More model files to score (e.g. a recovery run's ground truth)."""


def collect_models(args: Args) -> Dict[str, Path]:
    loop = Path(args.loop_dir)
    models: Dict[str, Path] = {}
    for name in read_manifest_names(args.seed_models):
        models[f"seed:{name}"] = Path(args.seed_models) / f"{name}.py"
    for name in read_manifest_names(loop / "models"):
        models[name] = loop / "models" / f"{name}.py"
    pruned = loop / "models" / "pruned"
    if args.include_pruned and pruned.exists():
        for f in sorted(pruned.glob("*.py")):
            models.setdefault(f"pruned:{f.stem}", f)
    for f in args.extra_models:
        models[f"extra:{Path(f).stem}"] = Path(f)
    return models


def main(args: Args) -> pd.DataFrame:
    loop = Path(args.loop_dir)
    train_path = loop / "responses.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found: is {loop} an RSA loop run?")
    test = load_forced_choice(args.test)
    if not test.contexts:
        raise ValueError(f"{args.test} has no included forced-choice trials")
    units = pd.factorize(unit_keys(test.frame))[0]
    source = test.frame["source"] if "source" in test.frame else pd.Series("pragmods", index=test.frame.index)
    settings = FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains)
    lpd: Dict[str, np.ndarray] = {}
    for label, path in collect_models(args).items():
        name = path.stem
        fitted = fit_cached(path, name, train_path, settings, loop / ".fit_cache")
        lpd[label] = heldout_lpd(RSAModel(path, name=name), fitted, test.contexts, test.choices)
        print(f"  {label}: held-out lpd {lpd[label].sum():.1f}")
    seeds = [k for k in lpd if k.startswith("seed:")]
    best_seed = max(seeds, key=lambda k: lpd[k].sum())
    rows = []
    for label, v in lpd.items():
        row = dict(model=label, lpd=float(v.sum()), lpd_per_trial=float(v.mean()),
                   diff_vs_best_seed=float(v.sum() - lpd[best_seed].sum()),
                   se_diff_clustered=float(cluster_dse(v, lpd[best_seed], units)) if label != best_seed else 0.0)
        for src in sorted(source.unique()):
            row[f"lpd[{src}]"] = float(v[(source == src).to_numpy()].sum())
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("lpd", ascending=False)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "heldout.csv", index=False)
    pd.DataFrame(lpd).to_csv(out / "heldout_pointwise.csv", index=False)
    (out / "heldout.json").write_text(json.dumps(dict(
        loop_dir=str(loop), test=str(args.test), n_test_trials=len(test.contexts),
        n_test_units=int(units.max() + 1), best_seed=best_seed, settings=vars(settings),
        table=table.to_dict(orient="records")), indent=1))
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    main(tyro.cli(Args))
