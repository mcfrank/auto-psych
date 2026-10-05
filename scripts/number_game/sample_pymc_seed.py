"""NUTS for the PyMC number-game seed on the whole dataset, for the port check.

    OMP_NUM_THREADS=1 uv run python -m scripts.number_game.sample_pymc_seed

Saves the draws of p_mix and the rule weights as `pymc_draws.npz` beside
`check_pymc_port.py sample`'s `numpyro_draws.npz`, for `check_pymc_port.py
compare`. Kept apart from that script so this process never imports jax:
PyMC forks its chains, and forking after jax has started its threads can hang.
The seed's rule weights are already normalised (Dirichlet), so they are saved
under the reference's name `p_rules`, which `compare` normalises (a no-op
here). Only the parameters are stored; `p_yes` for every response and draw would be
~9 GB.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pymc as pm
import tyro

from src.models.data_binding import make_stim_data
from src.models.model_loading import load_pymc_model
from src.number_game import data as ng_data
from src.runtime.config import REPO_ROOT

DATA_CSV = REPO_ROOT / "data" / "number-game" / "human experiment" / "numbergame_data.csv"
SEEDS_DIR = REPO_ROOT / "src" / "pipelines" / "outer_loop" / "projects" / "number_game" / "seed_models"


@dataclass
class Args:
    seed_model: str = "bigelow_piantadosi_independent"
    draws: int = 1000
    tune: int = 1000
    chains: int = 4
    seed: int = 0
    out_dir: Path = REPO_ROOT / "data" / "number-game" / "port_check"


def main(args: Args) -> None:
    responses = ng_data.load_responses(DATA_CSV)
    rows = [
        {"number_set": ",".join(map(str, s)), "target": t, "rating": r}
        for s, t, r in zip(responses["set"], responses["target"], responses["rating"])
    ]
    model = load_pymc_model(args.seed_model, SEEDS_DIR)
    with model:
        pm.set_data(make_stim_data(model, rows))
        trace = pm.sample(
            draws=args.draws, tune=args.tune, chains=args.chains, cores=args.chains,
            random_seed=args.seed, var_names=["p_mix", "rule_weights"],
        )
    post = trace.posterior
    divergent = int(trace.sample_stats["diverging"].sum())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "pymc_draws.npz"
    np.savez(out, p_mix=post["p_mix"].to_numpy(), p_rules=post["rule_weights"].to_numpy(), divergent=divergent)
    print(f"wrote {out}: {divergent} divergent")


if __name__ == "__main__":
    main(tyro.cli(Args))
