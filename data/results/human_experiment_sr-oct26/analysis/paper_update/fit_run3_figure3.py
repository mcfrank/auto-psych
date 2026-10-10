"""Fit the seeds and run 3's experiment-3 winner on all of run 3's data (MCMC).

The Figure 3 fits for run 3, made the way the generalization analysis made
runs 1 and 2's (fit_transfer_models.py): draws 3000, tune 2000, 4 chains,
target_accept 0.8; each model at the first sampler seed of SEEDS whose fit
passes the convergence gate. Fits go to the analysis's shared mcmc_cache; the
converged seed of each model is recorded in
analysis/fits/run3_experiment3/fit_seeds.json. A model that converges at no
seed raises.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

# The pipeline sizes its concurrent fits by the scheduler's affinity mask
# (``allocated_cpus``), which only Linux has; on a laptop, every CPU.
if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid: set(range(os.cpu_count()))

ANALYSIS_ROOT = here() / "data/results/human_experiment_sr-oct26/analysis"
RUN_ROOT = here() / "data/results/human_experiment_sr-oct26/collected/run3/subjective_randomness"
OUT = ANALYSIS_ROOT / "fits/run3_experiment3"
SEEDS_MODELS = ("falk_konold_dp", "finite_experience_occurrence", "local_representativeness", "motif_stack")
WINNER = "mild_symmetric_trick_coin_streak_weary"
SAMPLER = dict(draws=3000, tune=2000, chains=4, target_accept=0.8)
# fit_transfer_models.py's seed order.
SEEDS = (20261003, 6, 2, 1, 5, 3, 4, 7, 8, 9, 10, 11, 12)


@dataclass
class Args:
    """Fit the seeds and run 3's winner on run 3's experiment-3 data."""


def assemble_models_dir() -> list[str]:
    models_dir = OUT / "models"
    if models_dir.exists():
        shutil.rmtree(models_dir)
    models_dir.mkdir(parents=True)
    sources = [(s, RUN_ROOT / "experiment1/cognitive_models_input" / f"{s}.py") for s in SEEDS_MODELS]
    sources.append((WINNER, RUN_ROOT / f"experiment3/model_loop/models/{WINNER}.py"))
    for name, path in sources:
        if not path.exists():
            raise FileNotFoundError(f"model source missing: {path}")
        shutil.copyfile(path, models_dir / f"{name}.py")
    (models_dir / "models_manifest.yaml").write_text(
        "models:\n" + "".join(f"  - name: {n}\n" for n, _ in sources), encoding="utf-8"
    )
    return [n for n, _ in sources]


def main(args: Args) -> None:
    from src.models.pymc_inference import (
        clear_fit_cache,
        convergence_problems_of,
        fit_model,
        fit_models_cached,
    )

    names = assemble_models_dir()
    models_dir = OUT / "models"
    responses = RUN_ROOT / "experiment3/model_loop/responses.csv"
    cache = ANALYSIS_ROOT / "mcmc_cache"

    # Every model at the first seed, concurrently.
    first = fit_models_cached(names, models_dir, responses, cache_dir=cache, random_seed=SEEDS[0], **SAMPLER)
    records = {}
    pending = []
    for name in names:
        problems = convergence_problems_of(first[name])
        print(f"[fit] {name} seed {SEEDS[0]}: {problems or 'converged'}", flush=True)
        records[name] = {"seed": SEEDS[0] if not problems else None, "tried": [{"seed": SEEDS[0], "problems": problems}]}
        if problems:
            pending.append(name)
    del first
    clear_fit_cache()

    # The rest, one seed at a time until one converges.
    for name in pending:
        for seed in SEEDS[1:]:
            fitted = fit_model(name, models_dir, responses, cache_dir=cache, random_seed=seed, **SAMPLER)
            problems = convergence_problems_of(fitted)
            print(f"[fit] {name} seed {seed}: {problems or 'converged'}", flush=True)
            records[name]["tried"].append({"seed": seed, "problems": problems})
            del fitted
            clear_fit_cache()
            if not problems:
                records[name]["seed"] = seed
                break
        else:
            raise RuntimeError(f"{name} converged at no seed of {SEEDS}")

    (OUT / "fit_seeds.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[fit] wrote {OUT / 'fit_seeds.json'}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
