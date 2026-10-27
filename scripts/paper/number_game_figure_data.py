"""Number-game data behind the paper's Figures 2 and 3.

The number game has one run, the October 2026 dataset run (``data/results/
number_game_dataset-oct26/``): three experiments whose responses were drawn
from Bigelow & Piantadosi's bank. This fits, on the run's final cumulative
data (experiment 3's ``model_loop/responses.csv``, 7,560 responses), every
model live at the run's end and the run's starting models (the three
literature seeds), with the run's own sampler settings, and compares them by
the run's own criterion, grouped cross-validation over example sets
(``src.model_comparison.posterior.compare_table`` with the run's
``selection``; settings from the harness's ``holdout.json``):

- Figure 2: the run's winner, its other live models and the seeds as ELPD-CV
  differences from the best seed; the winner's with the stimulus-clustered
  standard error of that paired difference. Unlike subjective randomness,
  the other live models are refitted too (their CV folds are cheap), so every
  number on the figure comes from these fits; the loop's own ELPD-CV of each
  is recorded beside it. So is every model the run admitted and later
  removed (pruned, retired by the cap or dropped as unfittable; the ledger's
  ``admitted`` names that are neither live at the end nor seeds), scored on
  the same final data: ``pruned_during_run``. Of the 84, the 32 pruned at
  the end of experiment 3 have their ELPD-CV in its ``history.json``; the
  other 52 are fitted first, in shards (``--removed-shard``;
  ``scripts/paper/removed_models.py``). ``pruned_source`` says which. A
  removed model whose fit fails, or does not converge, is listed in
  ``pruned_unscored`` with the reason instead (its ELPD-CV is no
  measurement).
- Figure 3: one run, so "all runs pooled" is the run's final data, and the
  best model and best seed are Figure 2's. Each (example set, target) cell's
  observed proportion of "yes" beside both models' posterior-mean P(yes),
  from their fits on all of those data.

The seeds were pruned in experiment 1, so no seed has an ELPD-CV on the
final data without these fits. Fits are cached under ``--work-dir`` and the
result is saved there as ``scored.json``; a rerun reuses both. Any compared
model whose fit did not converge raises: which model is best is then not
known.

Usage (on Sherlock, both steps: ``bash scripts/paper/submit_figure_data.sh
number_game``):
    uv run python -m scripts.paper.number_game_figure_data --removed-shard 0 --removed-shards 1
    uv run python -m scripts.paper.number_game_figure_data
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import tyro
from pyprojroot import here

from scripts.paper import removed_models as rm
from scripts.paper import sr_figure_data as sr

RUN_ROOT = here() / "data" / "results" / "number_game_dataset-oct26"
DEFAULT_RUN_DIR = RUN_ROOT / "collected" / "run1" / "number_game"
DEFAULT_HOLDOUT = RUN_ROOT / "harness" / "run1" / "holdout.json"
DEFAULT_OUT = here() / "data" / "paper" / "figure_data" / "number_game.json"
DEFAULT_WORK_DIR = here() / "data" / "paper" / "work" / "number_game"


@dataclass
class Args:
    run_dir: Path = DEFAULT_RUN_DIR
    """The run's tree: ``experiment<N>/`` and ``starting_models.json``."""
    run: str = "run1"
    holdout: Path = DEFAULT_HOLDOUT
    """The harness's record of the run: its sampler settings (``fit_kwargs``)
    and model selection (``inner_loop.selection``)."""
    out: Path = DEFAULT_OUT
    work_dir: Path = DEFAULT_WORK_DIR
    """The comparison model set, a copy of the responses (the CV folds are
    written beside it) and the fit cache."""
    prediction_draws: int = 2000
    """Posterior draws (thinned) behind each posterior-mean prediction."""
    removed_shard: Optional[int] = None
    """Score only this shard of the removed models (of ``--removed-shards``)
    into ``<work-dir>/removed/`` and stop; without it the script assembles
    the figure data, which needs every removed model's record."""
    removed_shards: int = 1


def run_settings(holdout: Dict[str, Any]) -> Dict[str, Any]:
    """The run's sampler settings and model selection, as recorded."""
    from src.model_comparison.cv import Selection

    selection = Selection.from_config(holdout["inner_loop"]["selection"])
    if selection.criterion != "grouped_cv":
        raise RuntimeError(f"the run selected models by {selection.criterion!r}, not grouped CV")
    return {"fit_kwargs": dict(holdout["fit_kwargs"]), "selection": selection}


def live_models(final_dir: Path) -> Dict[str, Any]:
    """The run's winner and its other models live at the end, with the loop's
    own ELPD-CV of each (``model_posterior.json``)."""
    posterior = json.loads((final_dir / "model_loop" / "model_posterior.json").read_text())
    if posterior["criterion"] != "grouped_cv":
        raise RuntimeError(f"{final_dir}: the loop scored by {posterior['criterion']!r}, not grouped CV")
    winner = posterior["best_model"]
    names = list(posterior["elpd_loo"])
    if winner not in names:
        raise RuntimeError(f"{final_dir}: the loop's record has no ELPD for its winner {winner!r}")
    return {
        "winner": winner,
        "others": [n for n in names if n != winner],
        "recorded_elpd": {n: float(v) for n, v in posterior["elpd_loo"].items()},
        "n_trials": int(posterior["n_trials"]),
    }


def score_run(
    responses: Path,
    models_dir: Path,
    *,
    winner: str,
    others: Sequence[str],
    seeds: Sequence[str],
    compare: Callable[..., Dict[str, Dict[str, Any]]],
    **compare_kwargs: Any,
) -> Dict[str, Any]:
    """The winner and seeds against the best seed (``sr.score_against_best_seed``:
    paired differences and clustered SEs), and each other live model's
    difference from that seed. Raises if any of them did not converge."""
    scored = sr.score_against_best_seed(
        responses, models_dir, winner=winner, seeds=seeds, compare=compare, **compare_kwargs
    )
    best_seed = scored["best_seed"]
    others_at_end: Dict[str, float] = {}
    if others:
        table = compare(responses, models_dir, names=[*others, best_seed], **compare_kwargs)
        sr.require_trusted_fits(table, list(others))
        others_at_end = {
            n: float(table[n]["elpd_loo"]) - scored["elpd"][best_seed] for n in others
        }
        scored["elpd"].update({n: float(table[n]["elpd_loo"]) for n in others})
    return {**scored, "others_at_end": others_at_end}


def score_removed_model(name: str, models_dir: Path, responses: Path, *, selection: Any,
                        cache_dir: Path, **fit_kwargs: Any) -> Dict[str, Any]:
    """One removed model's ELPD-CV from its cached fits, or why it has none:
    its full-data fit (which decides convergence, as in ``compare_table``)
    did not converge."""
    from src.model_comparison.cv import cv_pointwise
    from src.models.pymc_inference import convergence_problems_of, fit_models_cached

    fitted = fit_models_cached([name], models_dir=models_dir, responses_path=responses,
                               cache_dir=cache_dir, **fit_kwargs)[name]
    problems = convergence_problems_of(fitted)
    if problems:
        return {"unscored": f"not converged ({'; '.join(problems)})"}
    lpd, _ = cv_pointwise([name], models_dir, responses, selection, cache_dir=cache_dir, **fit_kwargs)
    return {"elpd": float(lpd[name].sum())}


def condition_table(
    rows: Sequence[Dict[str, Any]], p_yes_by_model: Dict[str, np.ndarray]
) -> List[Dict[str, Any]]:
    """Per (example set, target) cell: ``n`` responses, the observed
    proportion of "yes" and each model's mean predicted P(yes)."""
    for name, p_yes in p_yes_by_model.items():
        if len(p_yes) != len(rows):
            raise ValueError(f"{name!r} has {len(p_yes)} predictions for {len(rows)} responses")
    grouped: Dict[tuple, Dict[str, List[float]]] = {}
    for i, row in enumerate(rows):
        rating = int(row["rating"])
        if rating not in (0, 1):
            raise ValueError(f"rating must be 0 or 1, not {row['rating']!r}")
        cell = grouped.setdefault(
            (row["number_set"], int(row["target"])),
            {"observed": [], **{m: [] for m in p_yes_by_model}},
        )
        cell["observed"].append(rating)
        for name, p_yes in p_yes_by_model.items():
            cell[name].append(float(p_yes[i]))
    return [
        {"number_set": number_set, "target": target, "n": len(cell["observed"]),
         **{k: float(np.mean(v)) for k, v in cell.items()}}
        for (number_set, target), cell in sorted(grouped.items())
    ]


def posterior_mean_p_yes(
    name: str, models_dir: Path, responses: Path, *, cache_dir: Path, prediction_draws: int,
    **fit_kwargs: Any,
) -> np.ndarray:
    """Each response's posterior-mean P(yes) under the model fitted on
    ``responses`` (a cache hit after the comparison)."""
    from src.models.data_binding import make_stim_data
    from src.models.pymc_inference import fit_model
    from src.tasks.number_game import NUMBER_GAME

    fitted = fit_model(name, models_dir, responses, cache_dir=cache_dir, **fit_kwargs)
    stim_data = make_stim_data(fitted.model, sr._read_rows(responses))
    draws = fitted.predict_p_left_draws(
        stim_data, var_name=NUMBER_GAME.prediction_var, seed=0, max_draws=prediction_draws
    )
    return np.asarray(draws, dtype="float64").mean(axis=0)


def main(args: Args) -> None:
    from src.model_comparison.posterior import compare_table

    settings = run_settings(json.loads(args.holdout.read_text()))
    fit_kwargs, selection = settings["fit_kwargs"], settings["selection"]
    final_dir = sr.final_experiment_dir(args.run_dir)
    live = live_models(final_dir)
    seeds = json.loads((args.run_dir / "starting_models.json").read_text())["starting_models"]
    if live["winner"] in seeds:
        raise RuntimeError(f"the winner {live['winner']!r} is a seed; nothing to compare")
    loop = final_dir / "model_loop"
    files = {n: loop / "models" / f"{n}.py" for n in [live["winner"], *live["others"]]}
    files.update({s: sr.seed_file(args.run_dir, final_dir, s) for s in seeds})

    removed = rm.removed_models(args.run_dir, final_dir, live=[live["winner"], *live["others"]],
                                seeds=seeds)
    recorded = rm.recorded_scores(final_dir, removed)
    to_fit = sorted(set(removed) - set(recorded))
    records_dir = args.work_dir / "removed"
    cache_dir = args.work_dir / "fit_cache"

    def removed_inputs(name: str) -> Dict[str, Any]:
        return sr.run_inputs(loop / "responses.csv", {name: removed[name]["file"]}, fit_kwargs,
                             selection=selection.as_record())

    if args.removed_shard is not None:
        # Each shard compares on a copy of its own: the CV folds are written
        # beside the responses, and identical copies share the fit cache.
        shard_dir = args.work_dir / "shards" / f"shard_{args.removed_shard}"
        responses = shard_dir / "responses.csv"
        rm.copy_atomically(loop / "responses.csv", responses)
        from src.model_comparison.cv import fold_training_files

        _, _, fold_files = fold_training_files(responses, selection)
        names = rm.shard_of(to_fit, args.removed_shard, args.removed_shards)
        rm.score_shard(
            {n: removed[n]["file"] for n in names},
            models_dir=shard_dir / "models", records_dir=records_dir, inputs=removed_inputs,
            training_files=[("all data", responses)]
            + [(f"CV fold {f}", p) for f, p in enumerate(fold_files)],
            score=lambda n: score_removed_model(n, shard_dir / "models", responses,
                                                selection=selection, cache_dir=cache_dir,
                                                **fit_kwargs),
            cache_dir=cache_dir, fit_kwargs=fit_kwargs,
        )
        return
    removed_scored = rm.collect(to_fit, records_dir=records_dir, inputs=removed_inputs)
    removed_scored["elpd"].update(recorded)

    # Compared on a copy: the CV folds are written beside the responses.
    responses = args.work_dir / "responses.csv"
    rm.copy_atomically(loop / "responses.csv", responses)
    rows = sr._read_rows(responses)
    if len(rows) != live["n_trials"]:
        raise RuntimeError(
            f"the loop scored its models on {live['n_trials']} responses, the final data "
            f"have {len(rows)}"
        )

    inputs = sr.run_inputs(responses, files, fit_kwargs, selection=selection.as_record(),
                           prediction_draws=args.prediction_draws)
    scored_path = args.work_dir / "scored.json"
    saved = sr.load_scored(scored_path, inputs)
    if saved is not None:
        print(f"reusing the scored record {scored_path}", flush=True)
        record = saved["record"]
    else:
        models_dir = args.work_dir / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)
        sr.write_model_set(models_dir, files)
        print(f"fitting {list(files)} on {responses} (grouped CV: {selection.as_record()})",
              flush=True)
        scored = score_run(
            responses, models_dir, winner=live["winner"], others=live["others"], seeds=seeds,
            compare=compare_table, selection=selection, cache_dir=cache_dir, **fit_kwargs,
        )
        predictions = {
            role: posterior_mean_p_yes(
                name, models_dir, responses, cache_dir=cache_dir,
                prediction_draws=args.prediction_draws, **fit_kwargs,
            )
            for role, name in (("best_model", live["winner"]), ("best_seed", scored["best_seed"]))
        }
        record = {
            "run": {
                "run": args.run,
                "final_experiment": final_dir.name,
                "n_trials": len(rows),
                "best_model": live["winner"],
                "recorded_elpd": live["recorded_elpd"],
                **scored,
            },
            "conditions": condition_table(rows, predictions),
        }
        sr.save_scored(scored_path, inputs, record)

    run = record["run"]
    best_seed_elpd = run["elpd"][run["best_seed"]]
    run = {
        **run,
        "pruned_during_run": {n: e - best_seed_elpd for n, e in removed_scored["elpd"].items()},
        "pruned_elpd": removed_scored["elpd"],
        "pruned_unscored": removed_scored["unscored"],
        "pruned_outcome": {n: r["outcome"] for n, r in removed.items()},
        "pruned_source": {n: "loop" if n in recorded else "refit" for n in removed},
        "pruned_order": rm.pruned_order(final_dir, sorted(removed_scored["elpd"])),
        "others_order": rm.pruned_order(final_dir, sorted(run["others_at_end"])),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "domain": "number_game",
                "scoring": (
                    f"ELPD from grouped cross-validation ({selection.cv_folds} folds of whole "
                    f"example sets) on the run's final data ({run['n_trials']:,} responses)"
                ),
                "fit_settings": fit_kwargs,
                "selection": selection.as_record(),
                "runs": [run],
                "pooled": {
                    "runs": [run["run"]],
                    "n_trials": run["n_trials"],
                    "best_model": run["best_model"],
                    "best_seed": run["best_seed"],
                    "winner_runs": {run["best_model"]: run["run"]},
                    "selected_by": "one run: its winner, and the seed with the highest ELPD-CV",
                },
                "conditions": record["conditions"],
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
