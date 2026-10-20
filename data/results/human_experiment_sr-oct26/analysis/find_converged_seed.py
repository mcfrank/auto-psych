"""Find a sampler seed at which a model's fit passes the convergence gate.

``motif_stack`` fails at the analysis seed: some chains start where every
step diverges and never move, and others settle in a second, much worse mode
(sensitivity ``beta`` ~0.25 rather than ~0.9). The loop's own fits of it were
converged. This tries seeds in order and stops at the first whose fit has no
convergence problems (``convergence_problems_of``: divergences, R-hat, bulk
ESS), fitting into the analysis cache so the fit it finds is the one the
analysis loads. Every seed tried is printed with its per-chain divergences
and per-chain mean beta, the record of the search.

Choosing the converged fit over a stuck one is conservative for the paper's
question: it can only improve the seed model's score.

Usage:
    uv run python data/results/human_experiment_sr-oct26/analysis/find_converged_seed.py \\
        --model motif_stack --experiment 1
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid: set(range(os.cpu_count()))

ANALYSIS_ROOT = here() / "data/results/human_experiment_sr-oct26/analysis"


@dataclass
class Args:
    """Try sampler seeds until a model's fit converges."""

    model: str
    experiment: int
    """Fit on this experiment's cumulative model_loop/responses.csv."""
    run: int = 1
    models_dir: Path = ANALYSIS_ROOT / "ablation_models"
    """Where ``<model>.py`` is: the ablation models, or a seed's source dir."""
    cache_dir: Path = ANALYSIS_ROOT / "mcmc_cache"
    seeds: tuple[int, ...] = tuple(range(1, 13))
    # The live run's settings, as fit_run_models.py.
    draws: int = 3000
    tune: int = 2000
    chains: int = 4
    target_accept: float = 0.8


def main(args: Args) -> None:
    from src.models.pymc_inference import clear_fit_cache, convergence_problems_of, fit_model

    responses = (
        here() / f"data/results/human_experiment_sr-oct26/collected/run{args.run}/subjective_randomness"
        / f"experiment{args.experiment}/model_loop/responses.csv"
    )
    for seed in args.seeds:
        fitted = fit_model(
            args.model, args.models_dir, responses, cache_dir=args.cache_dir,
            draws=args.draws, tune=args.tune, chains=args.chains,
            target_accept=args.target_accept, random_seed=seed,
        )
        problems = convergence_problems_of(fitted)
        divergent = fitted.idata.sample_stats.diverging.sum("draw").values.tolist()
        beta = fitted.idata.posterior.get("beta")
        beta_by_chain = beta.mean("draw").values.round(3).tolist() if beta is not None else None
        print(
            f"SEED {seed} {args.model}@experiment{args.experiment} "
            f"elpd={fitted.loo_diagnostics().elpd_loo:.1f} divergent/chain={divergent} "
            f"beta/chain={beta_by_chain} problems={problems}",
            flush=True,
        )
        clear_fit_cache()
        if not problems:
            print(f"CONVERGED {args.model}@experiment{args.experiment}={seed}", flush=True)
            return
    raise RuntimeError(f"{args.model}@experiment{args.experiment}: no converged fit at seeds {args.seeds}")


if __name__ == "__main__":
    main(tyro.cli(Args))
