"""Reference games: posterior summaries of the three finals' parameters.

The finals' pooled fits (``data/rsa/live/finals_pool_fit.py``) were made in a
work root that no longer exists, and ``finals_pool_preds_final.json`` records
only their predictions. This refits each final the same way: on the
pre-registration's ``pool/pooled.csv`` (checked against the sha256 that
``bar_pool_preds.json`` records), through the loop's ``loop_fit`` with the
frozen settings. It then checks the refit against the frozen fit by comparing
its posterior-mean pool predictions with the frozen ones, and writes:

- ``parameters.csv``: one row per scalar parameter (and per element of a
  vector one): posterior mean, sd, 95% HDI, bulk ESS, R-hat;
- ``fits.json``: per final, its convergence verdict and the largest absolute
  difference between the refit's pool predictions and the frozen ones.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tyro
from pyprojroot import here

from src.rsa.cpus import pin_main_thread

LIVE_ROOT = here() / "data" / "rsa" / "live"


@dataclass
class Args:
    pooled: Path
    """The pre-registration's pooled data (``$SCRATCH/auto-psych/rsa_prereg/pool/pooled.csv``)."""
    work_dir: Path
    """The fit cache goes here."""
    out_dir: Path = here() / "data" / "paper" / "parameter_summaries" / "rsa_finals"


def finals_settings():
    """The settings ``finals_pool_fit.py`` fitted with."""
    from src.rsa.fit import FitSettings

    return FitSettings(num_warmup=1000, num_samples=1000, num_chains=4, seed=0, dense_mass=True)


def main(args: Args) -> None:
    pin_main_thread()  # before JAX starts its thread pools (src.rsa.cpus)
    import arviz as az

    from src.rsa.loop.fitting import NO_TIME_LIMIT, describe, loop_fit
    from src.rsa.loop.novelty import plain_pool, pool_digest, posterior_mean_class_probs
    from src.rsa.model_file import RSAModel

    frozen = json.loads((LIVE_ROOT / "finals_pool_preds_final.json").read_text())
    bar = json.loads((LIVE_ROOT / "bar_pool_preds.json").read_text())
    pooled_sha = hashlib.sha256(args.pooled.read_bytes()).hexdigest()
    if pooled_sha != bar["data"]["sha256"]:
        raise RuntimeError(f"{args.pooled} has sha256 {pooled_sha}, the frozen fits used {bar['data']['sha256']}")
    pool = plain_pool()
    if pool_digest(pool) != frozen["pool"]["digest"]:
        raise RuntimeError("plain_pool() is not the pool the frozen predictions were made on")

    finals = {}
    for name, record in frozen["models"].items():
        path = LIVE_ROOT / "finals" / f"{name}.py"
        sha = hashlib.sha256(path.read_bytes()).hexdigest()[:16]
        if sha != record["model_sha256"]:
            raise RuntimeError(f"{path} has sha256 {sha}, the frozen fit used {record['model_sha256']}")
        finals[name] = path

    def fit_one(name):
        fitted = loop_fit(finals[name], name, args.pooled, finals_settings(), args.work_dir / ".fit_cache",
                          time_limit_sec=NO_TIME_LIMIT)
        probs = posterior_mean_class_probs(RSAModel(finals[name], name=name), fitted, pool)
        print("done", name, describe(fitted), flush=True)
        return name, fitted, probs

    with ThreadPoolExecutor(max_workers=len(finals)) as ex:
        results = list(ex.map(fit_one, finals))

    tables, fits = [], {}
    for name, fitted, probs in results:
        frozen_probs = np.asarray(frozen["models"][name]["probs"])
        fits[name] = dict(chain=frozen["models"][name]["chain"], **describe(fitted),
                          max_abs_diff_from_frozen_preds=float(np.max(np.abs(probs - frozen_probs))))
        summary = az.summary(fitted.idata.posterior, var_names=fitted.param_names, kind="all", hdi_prob=0.95)
        summary = summary[["mean", "sd", "hdi_2.5%", "hdi_97.5%", "ess_bulk", "r_hat"]]
        summary.insert(0, "parameter", summary.index)
        summary.insert(0, "chain", fits[name]["chain"])
        summary.insert(0, "model", name)
        tables.append(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(tables).to_csv(args.out_dir / "parameters.csv", index=False)
    (args.out_dir / "fits.json").write_text(json.dumps(fits, indent=1) + "\n")
    print(json.dumps(fits, indent=1), flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
