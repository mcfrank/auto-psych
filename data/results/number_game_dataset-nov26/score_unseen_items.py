"""Score models on the number-game items a dataset run never drew.

``../number_game_dataset-oct26/score_unseen_items.py`` with models named by
file (each experiment's winner sits in its own experiment's
``cognitive_models/``) and the results written to ``--out-dir``. Each model is
fitted (or loaded from ``--cache``) on the run's cumulative responses
(``--train``: experiment 3's ``model_loop/responses.csv``) and predicts every
(set, target) item of the Bigelow & Piantadosi bank that the run did not draw.
Per model: the log-likelihood per held-out response (from the posterior-mean
P(yes)) and the item R² (squared correlation of predicted P(yes) with each
item's yes rate), also as a share of the noise ceiling
1 - mean[r(1-r)/(n-1)] / var(r). Two constant baselines: the training yes rate,
and the unseen items' own yes rate.

    python data/results/number_game_dataset-nov26/score_unseen_items.py \\
        --train <responses.csv> --cache <fit cache dir> --out-dir <dir> \\
        --models <path/to/model.py> ...
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import tyro

from src.models.data_binding import make_stim_data
from src.models.pymc_inference import fit_model
from src.number_game.dataset import _cells

FIT = dict(method="warm_start_nuts", tune=200, draws=500, chains=4)  # dataset_run.yaml's fit settings


@dataclass
class Args:
    train: Path
    cache: Path
    out_dir: Path
    models: List[Path]


def _loglik(p: np.ndarray, yes: np.ndarray, n: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float((yes * np.log(p) + (n - yes) * np.log(1 - p)).sum() / n.sum())


def main(args: Args) -> None:
    train = list(csv.DictReader(args.train.open()))
    drawn = {(r["number_set"], r["target"]) for r in train}
    cells = {k: v for k, v in _cells().items() if k not in drawn}
    keys = list(cells)
    n = np.array([len(cells[k]) for k in keys])
    yes = np.array([sum(r["rating"] for r in cells[k]) for k in keys])
    obs = yes / n
    ceiling = 1 - np.mean(obs * (1 - obs) / (n - 1)) / obs.var()
    meta = {"train": str(args.train), "n_train_responses": len(train), "n_unseen_items": len(keys),
            "n_unseen_responses": int(n.sum()), "unseen_yes_rate": float(yes.sum() / n.sum()),
            "r2_ceiling": float(ceiling), "fit": FIT}
    print(json.dumps(meta))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "unseen_items.json").write_text(json.dumps(meta, indent=2) + "\n")

    train_rate = np.mean([int(r["rating"]) for r in train])
    out = [
        {"model": "constant: training yes rate", "file": "", "loglik_per_response": _loglik(np.full(len(keys), train_rate), yes, n),
         "item_r2": "", "share_of_ceiling": ""},
        {"model": "constant: unseen yes rate", "file": "", "loglik_per_response": _loglik(np.full(len(keys), meta["unseen_yes_rate"]), yes, n),
         "item_r2": "", "share_of_ceiling": ""},
    ]
    rows = [{"number_set": s, "target": t, "rating": 0} for s, t in keys]
    for path in args.models:
        f = fit_model(path.stem, path.parent, args.train, cache_dir=args.cache, **FIT)
        p = np.concatenate([
            f.predict_p_left_draws(make_stim_data(f.model, rows[i:i + 2000]), var_name="p_yes", max_draws=200).mean(0)
            for i in range(0, len(rows), 2000)
        ])
        r2 = float(np.corrcoef(p, obs)[0, 1] ** 2)
        out.append({"model": path.stem, "file": str(path), "loglik_per_response": _loglik(p, yes, n),
                    "item_r2": r2, "share_of_ceiling": r2 / ceiling})
        print(json.dumps(out[-1]), flush=True)
    with (args.out_dir / "unseen_scores.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)


if __name__ == "__main__":
    main(tyro.cli(Args))
