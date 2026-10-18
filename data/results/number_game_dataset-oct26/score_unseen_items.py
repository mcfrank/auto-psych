"""Score models on the number-game items a dataset run never drew.

The README's table B. Each model is fitted (or loaded from ``--cache``) on the
run's cumulative responses (``--train``, e.g. experiment 3's
``model_loop/responses.csv``) and predicts every (set, target) item of the
Bigelow & Piantadosi bank that the run did not draw. Reports, per model, the
log-likelihood per held-out response (from the posterior-mean P(yes)) and the
item R² (squared correlation of predicted P(yes) with each item's yes rate),
also as a share of the noise ceiling 1 - mean[r(1-r)/(n-1)] / var(r).

    uv run python data/results/number_game_dataset-oct26/score_unseen_items.py \\
        --models-dir <dir with the models' .py files> --train <responses.csv> \\
        --cache <fit cache dir> --models even_rule_digit_size_principle_counting ...
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import tyro

from src.models.data_binding import make_stim_data
from src.models.pymc_inference import fit_model
from src.number_game.dataset import _cells


@dataclass
class Args:
    models_dir: Path
    train: Path
    cache: Path
    models: List[str]


def main(args: Args) -> None:
    drawn = {(r["number_set"], r["target"]) for r in csv.DictReader(args.train.open())}
    cells = {k: v for k, v in _cells().items() if k not in drawn}
    keys = list(cells)
    n = np.array([len(cells[k]) for k in keys])
    yes = np.array([sum(r["rating"] for r in cells[k]) for k in keys])
    obs = yes / n
    ceiling = 1 - np.mean(obs * (1 - obs) / (n - 1)) / obs.var()
    print(f"{len(keys)} unseen items, {n.sum()} responses, yes rate {yes.sum() / n.sum():.3f}, R² ceiling {ceiling:.3f}")
    rows = [{"number_set": s, "target": t, "rating": 0} for s, t in keys]
    fit = dict(method="warm_start_nuts", tune=200, draws=500, chains=4)  # the run's fit settings
    for name in args.models:
        f = fit_model(name, args.models_dir, args.train, cache_dir=args.cache, **fit)
        p = np.concatenate([
            f.predict_p_left_draws(make_stim_data(f.model, rows[i:i + 2000]), var_name="p_yes", max_draws=200).mean(0)
            for i in range(0, len(rows), 2000)
        ])
        p = np.clip(p, 1e-9, 1 - 1e-9)
        ll = (yes * np.log(p) + (n - yes) * np.log(1 - p)).sum() / n.sum()
        r2 = np.corrcoef(p, obs)[0, 1] ** 2
        print(f"  {name}: log-lik/resp {ll:.3f}, item R² {r2:.3f} ({r2 / ceiling:.0%} of ceiling)")


if __name__ == "__main__":
    main(tyro.cli(Args))
