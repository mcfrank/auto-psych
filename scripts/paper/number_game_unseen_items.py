"""Score the pooled number-game models on the dataset's items no run drew.

The models are each run's winner and the seeds as fitted on all runs' data
pooled at the pooled settings (``number_game_figure_data.py``); their fits are
read from its fit cache, never sampled here. The test set is every response of
the Bigelow & Piantadosi dataset whose (set, target) item no run drew: no agent
saw these responses and no pruning decision used them. A second, stricter test
set keeps only the items whose example set no run drew.

Each response is scored by its log posterior predictive density, the log of
P(its rating) averaged over posterior draws, with its own ``trial_index``.
Reported per model and test set: the summed log predictive density, its
difference from the best seed's with the SE clustered by example set, and the
item R² (squared correlation of each item's mean predicted P(yes) with its yes
rate) as a share of the noise ceiling 1 - mean[r(1-r)/(n-1)] / var(r).

    uv run python -m scripts.paper.number_game_unseen_items --work-dir <figure data work dir>
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import tyro
from pyprojroot import here

from scripts.paper import number_game_figure_data as ng

# The columns the pooled models were fitted on (the pooled responses.csv's).
ROW_COLUMNS = ("number_set", "target", "trial_index", "rating")


@dataclass
class Args:
    work_dir: Path
    """The figure data's work directory: ``pooled/`` holds the models, the
    pooled responses and the fit cache."""
    harness_root: Path = ng.DEFAULT_HARNESS_ROOT
    runs: List[str] = field(default_factory=lambda: ["run1", "run2", "run3"])
    figure_data: Path = ng.DEFAULT_OUT
    """The figure data, for which models are the pooled winners and seeds."""
    out: Path = here() / "data" / "paper" / "number_game_unseen_items.json"
    prediction_draws: int = 1000
    """Posterior draws (thinned) behind each response's predictive density."""
    chunk_rows: int = 5000
    """Responses predicted per posterior-predictive pass (bounds memory)."""


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def unseen_responses(all_rows: Sequence[Mapping[str, Any]], drawn_rows: Sequence[Mapping[str, Any]]):
    """The responses whose (set, target) item no run drew, and for each
    whether no run drew its example set either."""
    drawn_items = {(r["number_set"], r["target"]) for r in drawn_rows}
    drawn_sets = {r["number_set"] for r in drawn_rows}
    rows = [r for r in all_rows if (r["number_set"], r["target"]) not in drawn_items]
    set_unseen = np.array([r["number_set"] not in drawn_sets for r in rows], dtype=bool)
    return rows, set_unseen


def pointwise_lpd(p_yes_draws: np.ndarray, ratings: np.ndarray) -> np.ndarray:
    """Each response's log of P(its rating) averaged over draws; ``p_yes_draws``
    is (draws, responses)."""
    p = np.clip(p_yes_draws, 1e-12, 1 - 1e-12)
    p_rating = np.where(ratings == 1, p, 1 - p)
    return np.log(p_rating.mean(axis=0))


def example_set_ids(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    ids: Dict[str, int] = {}
    return np.array([ids.setdefault(r["number_set"], len(ids)) for r in rows], dtype=np.int64)


def item_r2(rows: Sequence[Mapping[str, Any]], p_yes: np.ndarray) -> Dict[str, float]:
    """Squared correlation of each item's mean predicted P(yes) with its yes
    rate, and that as a share of the noise ceiling."""
    items: Dict[Any, List[int]] = {}
    for i, r in enumerate(rows):
        items.setdefault((r["number_set"], r["target"]), []).append(i)
    index = list(items.values())
    n = np.array([len(ix) for ix in index])
    ratings = np.array([int(r["rating"]) for r in rows])
    observed = np.array([ratings[ix].mean() for ix in index])
    predicted = np.array([p_yes[ix].mean() for ix in index])
    ceiling = 1 - np.mean(observed * (1 - observed) / (n - 1)) / observed.var()
    r2 = np.corrcoef(predicted, observed)[0, 1] ** 2
    return {"item_r2": float(r2), "ceiling": float(ceiling), "share_of_ceiling": float(r2 / ceiling)}


def score(rows: Sequence[Mapping[str, Any]], lpd: Dict[str, np.ndarray], p_yes: Dict[str, np.ndarray],
          seeds: Sequence[str]) -> Dict[str, Any]:
    """Every model's summed lpd, its difference from the best seed's (the seed
    with the highest lpd on these rows) with the SE clustered by example set,
    and its item R²."""
    from src.models.clustered_se import cluster_dse

    groups = example_set_ids(rows)
    best_seed = max(seeds, key=lambda s: lpd[s].sum())
    return {
        "n_responses": len(rows),
        "n_items": len({(r["number_set"], r["target"]) for r in rows}),
        "n_example_sets": int(groups.max()) + 1,
        "best_seed": best_seed,
        "models": {
            name: {
                "lpd": float(lpd[name].sum()),
                "lpd_per_response": float(lpd[name].mean()),
                "vs_best_seed": {"diff": float((lpd[name] - lpd[best_seed]).sum()),
                                 "se": cluster_dse(lpd[name], lpd[best_seed], groups)},
                **item_r2(rows, p_yes[name]),
            }
            for name in lpd
        },
    }


def predict(fitted: Any, rows: Sequence[Mapping[str, Any]], *, draws: int, chunk_rows: int) -> np.ndarray:
    """(draws, responses) posterior-predictive P(yes), in chunks of rows."""
    from src.models.data_binding import make_stim_data
    from src.tasks.number_game import NUMBER_GAME

    stim_rows = [{c: r[c] for c in ROW_COLUMNS} for r in rows]
    return np.concatenate([
        np.asarray(fitted.predict_p_left_draws(
            make_stim_data(fitted.model, stim_rows[i:i + chunk_rows]),
            var_name=NUMBER_GAME.prediction_var, seed=0, max_draws=draws), dtype="float64")
        for i in range(0, len(stim_rows), chunk_rows)
    ], axis=1)


def main(args: Args) -> None:
    from src.models.pymc_inference import load_cached_fit
    from src.number_game.dataset import _cells

    settings = ng.shared_run_settings(args.harness_root, args.runs)
    fit_settings = ng.pooled_fit_settings(ng.Args(), settings["fit_kwargs"])
    pooled = args.work_dir / ng.POOLED_DIR
    responses = pooled / "responses.csv"
    figure_data = json.loads(args.figure_data.read_text())["pooled"]
    winners = list(figure_data["winner_runs"])
    seeds = [name for name in figure_data["vs_best_seed"] if name not in winners]

    all_rows = [r for cell in _cells().values() for r in cell]
    rows, set_unseen = unseen_responses(all_rows, read_rows(responses))
    print(f"[unseen] {len(rows):,} responses on unseen items, "
          f"{set_unseen.sum():,} of them on unseen example sets", flush=True)
    ratings = np.array([r["rating"] for r in rows])

    lpd: Dict[str, np.ndarray] = {}
    p_yes: Dict[str, np.ndarray] = {}
    for name in [*winners, *seeds]:
        fitted = load_cached_fit(name, pooled / "models", responses, cache_dir=pooled / "fit_cache",
                                 **fit_settings)
        problems = fitted.convergence_problems()
        if problems:
            raise RuntimeError(f"{name}'s pooled fit did not converge: {problems}")
        draws = predict(fitted, rows, draws=args.prediction_draws, chunk_rows=args.chunk_rows)
        lpd[name], p_yes[name] = pointwise_lpd(draws, ratings), draws.mean(axis=0)
        print(f"[unseen] {name}: lpd/response {lpd[name].mean():.4f}", flush=True)

    unseen_set_rows = [r for r, keep in zip(rows, set_unseen) if keep]
    record = {
        "winner_runs": figure_data["winner_runs"],
        "seeds": seeds,
        "fit_settings": fit_settings,
        "prediction_draws": args.prediction_draws,
        "unseen_items": score(rows, lpd, p_yes, seeds),
        "unseen_example_sets": score(unseen_set_rows, {n: v[set_unseen] for n, v in lpd.items()},
                                     {n: v[set_unseen] for n, v in p_yes.items()}, seeds),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
