"""Number-game data behind the paper's Figures 2 and 3.

The number game's runs are the October 2026 dataset runs (``data/results/
number_game_dataset-oct26/``, seeds 1-3): three experiments each, whose
responses were drawn from Bigelow & Piantadosi's bank, each run drawing its
own items. Every model is compared by the runs' own criterion, grouped
cross-validation over example sets (``src.model_comparison.posterior.
compare_table`` with the runs' ``selection``), at the runs' own sampler
settings (both from each run's ``holdout.json``; they must agree).

- Each run (``runs``): on its final cumulative data (experiment 3's
  ``model_loop/responses.csv``, 7,560 responses), its winner, its other
  live models and the starting models (the three literature seeds), as
  ELPD-CV differences from the run's best seed; the winner's with the
  stimulus-clustered standard error of that paired difference. The loop's
  own ELPD-CV of each live model is recorded beside it. So is every model
  the run admitted and later removed (pruned, retired by the cap or dropped
  as unfittable; the ledger's ``admitted`` names that are neither live at
  the end nor seeds), scored on the same final data: ``pruned_during_run``.
  Those pruned at the end of experiment 3 have their ELPD-CV in its
  ``history.json``; the others are fitted first, in shards
  (``--removed-shard``; ``scripts/paper/removed_models.py``).
  ``pruned_source`` says which. A removed model whose fit fails, or does not
  converge, is listed in ``pruned_unscored`` with the reason instead (its
  ELPD-CV is no measurement).
- All runs pooled (``pooled``, what Figures 2 and 3 draw: each run is scored
  on its own items, so the runs' differences are not on one scale): every
  run's final data together (``pool_responses``; 22,680 responses), every
  run's winner and the seeds fitted on them, each against the best seed with
  its clustered SE (``vs_best_seed``), and every run's other live and
  removed models, labelled ``<run>__<name>`` (names repeat across runs with
  different code): ``others_at_end``, ``pruned_during_run``,
  ``extras_unscored``, fitted in the same shards. Figure 3: each (example
  set, target) cell's observed proportion of "yes" beside the posterior-mean
  P(yes) of the best winner and the best seed on the pooled data.

The seeds were pruned in experiment 1, so no seed has an ELPD-CV on the
final data without these fits. Fits are cached under ``--work-dir`` and each
run's (and the pooled) result is saved there as ``scored.json``; a rerun
reuses both. Any compared winner or seed whose fit did not converge raises:
which model is best is then not known.

Usage (on Sherlock, both steps: ``bash scripts/paper/submit_figure_data.sh
number_game``):
    uv run python -m scripts.paper.number_game_figure_data --removed-shard 0 --removed-shards 1
    uv run python -m scripts.paper.number_game_figure_data
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import tyro
from pyprojroot import here

from scripts.paper import removed_models as rm
from scripts.paper import sr_figure_data as sr

RUN_ROOT = here() / "data" / "results" / "number_game_dataset-oct26"
DEFAULT_RUNS_ROOT = RUN_ROOT / "collected"
DEFAULT_HARNESS_ROOT = RUN_ROOT / "harness"
DEFAULT_OUT = here() / "data" / "paper" / "figure_data" / "number_game.json"
DEFAULT_WORK_DIR = here() / "data" / "paper" / "work" / "number_game"
POOLED_DIR = sr.POOLED_DIR
POOLED_LABEL_SEP = sr.POOLED_LABEL_SEP


@dataclass
class Args:
    runs_root: Path = DEFAULT_RUNS_ROOT
    """Each run's tree: ``<run>/number_game/experiment<N>/`` and
    ``starting_models.json``."""
    harness_root: Path = DEFAULT_HARNESS_ROOT
    """Each run's harness record, ``<run>/holdout.json``: its sampler settings
    (``fit_kwargs``) and model selection (``inner_loop.selection``)."""
    runs: List[str] = field(default_factory=lambda: ["run1", "run2", "run3"])
    out: Path = DEFAULT_OUT
    work_dir: Path = DEFAULT_WORK_DIR
    """Per run and for all runs pooled (``<run>/``, ``pooled/``): the
    comparison model set, a copy of the responses (the CV folds are written
    beside it), the fit cache and the removed models' records."""
    prediction_draws: int = 2000
    """Posterior draws (thinned) behind each posterior-mean prediction."""
    removed_shard: Optional[int] = None
    """Score only this shard of the removed models and the pooled extras (of
    ``--removed-shards``) and stop; without it the script assembles the
    figure data, which needs every one's record."""
    removed_shards: int = 1


def run_dir(runs_root: Path, run: str) -> Path:
    return runs_root / run / "number_game"


def run_settings(holdout: Dict[str, Any]) -> Dict[str, Any]:
    """The run's sampler settings and model selection, as recorded."""
    from src.model_comparison.cv import Selection

    selection = Selection.from_config(holdout["inner_loop"]["selection"])
    if selection.criterion != "grouped_cv":
        raise RuntimeError(f"the run selected models by {selection.criterion!r}, not grouped CV")
    return {"fit_kwargs": dict(holdout["fit_kwargs"]), "selection": selection}


def shared_run_settings(harness_root: Path, runs: Sequence[str]) -> Dict[str, Any]:
    """The runs' sampler settings and model selection, which must be the same
    for every run: the pooled fits use them for all runs' models alike."""
    by_run = {run: run_settings(json.loads((harness_root / run / "holdout.json").read_text()))
              for run in runs}
    first = by_run[runs[0]]
    for run, settings in by_run.items():
        if (settings["fit_kwargs"] != first["fit_kwargs"]
                or settings["selection"].as_record() != first["selection"].as_record()):
            raise RuntimeError(
                f"{run} and {runs[0]}'s sampler settings or model selection differ: "
                f"{settings['fit_kwargs']} {settings['selection'].as_record()} vs "
                f"{first['fit_kwargs']} {first['selection'].as_record()}"
            )
    return first


def pool_responses(responses_by_run: Dict[str, Path], out: Path) -> None:
    """Write every run's responses, in run order, to one CSV. The runs hide
    participant ids (each run would number its own from 0), so a run that
    records them is refused rather than merged with another's people."""
    from src.runtime.atomic_files import write_text_atomically

    columns: Optional[List[str]] = None
    rows: List[Dict[str, str]] = []
    for run, responses in responses_by_run.items():
        with open(responses, newline="") as f:
            reader = csv.DictReader(f)
            if columns is None:
                columns = list(reader.fieldnames or [])
            if reader.fieldnames != columns:
                raise RuntimeError(f"{run}: {responses} has columns {reader.fieldnames}, not {columns}")
            if "participant_id" in columns:
                raise RuntimeError(f"{run}: {responses} records participant_id; pooling would merge "
                                   "different runs' participants")
            rows.extend(reader)
    text = io.StringIO()
    writer = csv.DictWriter(text, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    write_text_atomically(out, text.getvalue())


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


def run_models(args: Args, run: str, fit_kwargs: Dict[str, Any], selection: Any) -> Dict[str, Any]:
    """A run's live models, seeds, removed models (``rm.removed_models``) and
    final responses, and how a removed model's record is keyed."""
    tree = run_dir(args.runs_root, run)
    final_dir = sr.final_experiment_dir(tree)
    live = live_models(final_dir)
    seeds = json.loads((tree / "starting_models.json").read_text())["starting_models"]
    if live["winner"] in seeds:
        raise RuntimeError(f"{run}: the winner {live['winner']!r} is a seed; nothing to compare")
    loop = final_dir / "model_loop"
    files = {n: loop / "models" / f"{n}.py" for n in [live["winner"], *live["others"]]}
    files.update({s: sr.seed_file(tree, final_dir, s) for s in seeds})
    removed = rm.removed_models(tree, final_dir, live=[live["winner"], *live["others"]], seeds=seeds)
    recorded = rm.recorded_scores(final_dir, removed)
    responses = loop / "responses.csv"
    return {
        "run": run, "final_dir": final_dir, "live": live, "seeds": seeds, "files": files,
        "removed": removed, "recorded": recorded, "to_fit": sorted(set(removed) - set(recorded)),
        "responses": responses, "records_dir": args.work_dir / run / "removed",
        "inputs": lambda name: sr.run_inputs(responses, {name: removed[name]["file"]}, fit_kwargs,
                                             selection=selection.as_record()),
    }


def pooled_extras(infos: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Every run's other models live at the end and its removed models, for
    Figure 2's pooled view: ``<run>__<name>`` -> ``file``, ``run``, ``name``,
    ``role`` (``live`` / ``removed``), its run's admission ``order`` and a
    removed model's ``outcome`` (``sr.pooled_extras_fields`` reads them)."""
    extras: Dict[str, Dict[str, Any]] = {}
    for info in infos:
        run = info["run"]
        order = rm.admission_order(info["final_dir"])
        for name in info["live"]["others"]:
            extras[f"{run}{POOLED_LABEL_SEP}{name}"] = {
                "file": info["files"][name], "run": run, "name": name, "role": "live",
                "order": order[name]}
        for name, entry in info["removed"].items():
            extras[f"{run}{POOLED_LABEL_SEP}{name}"] = {
                "file": entry["file"], "run": run, "name": name, "role": "removed",
                "outcome": entry["outcome"], "order": order[name]}
    return extras


def write_pooled_responses(args: Args, infos: Sequence[Dict[str, Any]]) -> Path:
    responses = args.work_dir / POOLED_DIR / "responses.csv"
    responses.parent.mkdir(parents=True, exist_ok=True)
    pool_responses({info["run"]: info["responses"] for info in infos}, responses)
    return responses


def score_shard_models(
    files: Dict[str, Path], responses: Path, *, shard_dir: Path, records_dir: Path,
    cache_dir: Path, inputs: Callable[[str], Dict[str, Any]], selection: Any,
    fit_kwargs: Dict[str, Any],
) -> None:
    """Score ``files`` (name -> source) by grouped CV on ``responses``. Each
    shard compares on a copy of its own: the CV folds are written beside the
    responses, and identical copies share the fit cache."""
    from src.model_comparison.cv import fold_training_files

    copy = shard_dir / "responses.csv"
    rm.copy_atomically(responses, copy)
    _, _, fold_files = fold_training_files(copy, selection)
    rm.score_shard(
        files, models_dir=shard_dir / "models", records_dir=records_dir, inputs=inputs,
        training_files=[("all data", copy)] + [(f"CV fold {f}", p) for f, p in enumerate(fold_files)],
        score=lambda n: score_removed_model(n, shard_dir / "models", copy, selection=selection,
                                            cache_dir=cache_dir, **fit_kwargs),
        cache_dir=cache_dir, fit_kwargs=fit_kwargs,
    )


def score_one_run(args: Args, info: Dict[str, Any], removed_scored: Dict[str, Any],
                  fit_kwargs: Dict[str, Any], selection: Any) -> Dict[str, Any]:
    """Figure data's record of one run, on its own final data."""
    from src.model_comparison.posterior import compare_table

    run, live = info["run"], info["live"]
    work = args.work_dir / run
    # Compared on a copy: the CV folds are written beside the responses.
    responses = work / "responses.csv"
    rm.copy_atomically(info["responses"], responses)
    n_trials = len(sr._read_rows(responses))
    if n_trials != live["n_trials"]:
        raise RuntimeError(
            f"{run}: the loop scored its models on {live['n_trials']} responses, the final data "
            f"have {n_trials}"
        )
    inputs = sr.run_inputs(responses, info["files"], fit_kwargs, selection=selection.as_record())
    scored_path = work / "scored.json"
    saved = sr.load_scored(scored_path, inputs)
    if saved is not None:
        print(f"[{run}] reusing its scored record {scored_path}", flush=True)
        scored = saved["record"]
    else:
        models_dir = work / "models"
        cache_dir = work / "fit_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        sr.write_model_set(models_dir, info["files"])
        print(f"[{run}] fitting {list(info['files'])} on {responses} (grouped CV: "
              f"{selection.as_record()})", flush=True)
        scored = score_run(
            responses, models_dir, winner=live["winner"], others=live["others"],
            seeds=info["seeds"], compare=compare_table, selection=selection,
            cache_dir=cache_dir, **fit_kwargs,
        )
        sr.save_scored(scored_path, inputs, scored)

    best_seed_elpd = scored["elpd"][scored["best_seed"]]
    pruned_elpd = {**removed_scored["elpd"], **info["recorded"]}
    return {
        "run": run,
        "final_experiment": info["final_dir"].name,
        "n_trials": n_trials,
        "best_model": live["winner"],
        "recorded_elpd": live["recorded_elpd"],
        **scored,
        "pruned_during_run": {n: e - best_seed_elpd for n, e in pruned_elpd.items()},
        "pruned_elpd": pruned_elpd,
        "pruned_unscored": removed_scored["unscored"],
        "pruned_outcome": {n: r["outcome"] for n, r in info["removed"].items()},
        "pruned_source": {n: "loop" if n in info["recorded"] else "refit" for n in info["removed"]},
        "pruned_order": rm.pruned_order(info["final_dir"], sorted(pruned_elpd)),
        "others_order": rm.pruned_order(info["final_dir"], sorted(scored["others_at_end"])),
    }


def score_all_runs_pooled(args: Args, infos: Sequence[Dict[str, Any]], responses: Path,
                          fit_kwargs: Dict[str, Any], selection: Any) -> Dict[str, Any]:
    """Every run's winner and the seeds fitted on all runs' data, each against
    the best seed, and per condition the best winner's and best seed's
    predictions."""
    from src.model_comparison.posterior import compare_table

    files = sr.pooled_model_files(
        {i["run"]: {n: i["files"][n] for n in [i["live"]["winner"], *i["seeds"]]} for i in infos})
    winners = {i["live"]["winner"]: i["run"] for i in infos}
    if len(winners) != len(infos):
        raise RuntimeError(f"two runs share a winner's name: {[i['live']['winner'] for i in infos]}")
    seeds = infos[0]["seeds"]
    if any(i["seeds"] != seeds for i in infos):
        raise RuntimeError(f"the runs started from different seeds: {[i['seeds'] for i in infos]}")

    work = args.work_dir / POOLED_DIR
    inputs = sr.run_inputs(responses, files, fit_kwargs, selection=selection.as_record(),
                           prediction_draws=args.prediction_draws)
    scored_path = work / "scored.json"
    saved = sr.load_scored(scored_path, inputs)
    if saved is not None:
        print(f"[pooled] reusing its scored record {scored_path}", flush=True)
        return saved["record"]

    models_dir = work / "models"
    cache_dir = work / "fit_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    sr.write_model_set(models_dir, files)
    print(f"[pooled] fitting {list(winners)} and seeds {seeds} on {responses}", flush=True)
    scored = sr.score_pooled(responses, models_dir, winners=winners, seeds=seeds,
                             compare=compare_table, selection=selection, cache_dir=cache_dir,
                             **fit_kwargs)
    rows = sr._read_rows(responses)
    predictions = {
        role: posterior_mean_p_yes(scored[role], models_dir, responses, cache_dir=cache_dir,
                                   prediction_draws=args.prediction_draws, **fit_kwargs)
        for role in ("best_model", "best_seed")
    }
    record = {
        "pooled": {"runs": [i["run"] for i in infos], "n_trials": len(rows), **scored},
        "conditions": condition_table(rows, predictions),
    }
    sr.save_scored(scored_path, inputs, record)
    return record


def main(args: Args) -> None:
    settings = shared_run_settings(args.harness_root, args.runs)
    fit_kwargs, selection = settings["fit_kwargs"], settings["selection"]
    infos = [run_models(args, run, fit_kwargs, selection) for run in args.runs]
    extras = pooled_extras(infos)
    pooled_responses = write_pooled_responses(args, infos)
    pooled_records = args.work_dir / POOLED_DIR / "removed"

    def extra_inputs(label: str) -> Dict[str, Any]:
        return sr.run_inputs(pooled_responses, {label: extras[label]["file"]}, fit_kwargs,
                             selection=selection.as_record())

    if args.removed_shard is not None:
        items = [(i["run"], n) for i in infos for n in i["to_fit"]]
        items += [(POOLED_DIR, label) for label in sorted(extras)]
        mine = rm.shard_of(items, args.removed_shard, args.removed_shards)
        shard = f"shard_{args.removed_shard}"
        for info in infos:
            names = [n for group, n in mine if group == info["run"]]
            if names:
                print(f"[{info['run']}] scoring {len(names)} removed models", flush=True)
                score_shard_models(
                    {n: info["removed"][n]["file"] for n in names}, info["responses"],
                    shard_dir=args.work_dir / info["run"] / "shards" / shard,
                    records_dir=info["records_dir"],
                    cache_dir=args.work_dir / info["run"] / "fit_cache",
                    inputs=info["inputs"], selection=selection, fit_kwargs=fit_kwargs)
        labels = [label for group, label in mine if group == POOLED_DIR]
        if labels:
            print(f"[pooled] scoring {len(labels)} models on all runs pooled", flush=True)
            score_shard_models(
                {label: extras[label]["file"] for label in labels}, pooled_responses,
                shard_dir=args.work_dir / POOLED_DIR / "shards" / shard,
                records_dir=pooled_records, cache_dir=args.work_dir / POOLED_DIR / "fit_cache",
                inputs=extra_inputs, selection=selection, fit_kwargs=fit_kwargs)
        return

    # Every removed model's and extra's record first: a missing one stops the
    # script before any fit.
    removed_scored = {i["run"]: rm.collect(i["to_fit"], records_dir=i["records_dir"],
                                           inputs=i["inputs"]) for i in infos}
    extras_scored = rm.collect(sorted(extras), records_dir=pooled_records, inputs=extra_inputs)
    runs = [score_one_run(args, i, removed_scored[i["run"]], fit_kwargs, selection) for i in infos]
    pooled = score_all_runs_pooled(args, infos, pooled_responses, fit_kwargs, selection)
    pooled["pooled"].update(sr.pooled_extras_fields(extras, extras_scored, pooled["pooled"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "domain": "number_game",
                "scoring": (
                    f"ELPD from grouped cross-validation ({selection.cv_folds} folds of whole "
                    f"example sets) on each run's final data and on all runs' final data pooled "
                    f"({pooled['pooled']['n_trials']:,} responses)"
                ),
                "fit_settings": fit_kwargs,
                "selection": selection.as_record(),
                "runs": runs,
                **pooled,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
