"""Subjective-randomness data behind the paper's Figures 2 and 3.

For every run of the October 2026 live series (``data/results/
human_experiment_sr-oct26/collected/run<i>/``), this fits the final
experiment's exported winner and the run's starting models (the four
literature seeds) on the run's final pooled data, with the live run's own
sampler settings, and compares them by PSIS-LOO
(``src.model_comparison.posterior.compare_table``, the loop's own table):

- Figure 2: each model's ELPD-LOO difference from the run's best seed, with
  the stimulus-clustered standard error of that paired difference.
- Figure 3: every run's winner and the seeds fitted on all runs' data
  together (each run's participants kept apart, ``pool_responses``); the
  winner and the seed with the highest ELPD-LOO there are the best model and
  the best seed, and each condition's observed choice proportion is drawn
  beside their posterior-mean predictions. A condition is an unordered
  stimulus pair: both presentation orders are pooled onto the pair's
  alphabetically first sequence, so ``observed`` is P(choose ``pair[0]``).

Figure 2 also draws the models each run removed (pruned, retired or
dropped; ``scripts/paper/removed_models.py``), each fitted on the run's final
data at the same settings and scored by its own PSIS-LOO:
``pruned_during_run``, as a difference from the run's best seed. Those
pruned at the end of the final experiment have their ELPD-LOO in its
``history.json`` (18 to 21 per run); the other 50 to 60 per run are fitted
first, in shards (``--removed-shard``); ``pruned_source`` says which. One
that did not converge or whose PSIS-LOO is unreliable is listed in
``pruned_unscored`` with the reason instead.

Figure 2 draws subjective randomness from all runs pooled (each run is
scored on its own data, so the runs' differences are not on one scale): the
pooled record also has each winner's and seed's paired difference from the
best seed with its stimulus-clustered SE (``vs_best_seed``), and every run's
other live and removed models fitted on the pooled data at the pooled
settings, labelled ``<run>__<name>`` (names repeat across runs with different
code): ``others_at_end``, ``pruned_during_run``, ``extras_unscored``. The 254
of them are fitted in the same shards.

The seeds were pruned in experiment 1, so no seed has an ELPD on the final
data without these fits. Fits are cached under ``--work-dir``, and each run's
(and the pooled) result is saved there as ``scored.json``; a rerun reuses
both. A winner or best seed whose fit did not converge, or whose PSIS-LOO is
unreliable, raises: both are drawn in the figures.

Usage (long: one person-level winner takes tens of minutes at 7,680 trials;
on Sherlock, both steps: ``bash scripts/paper/submit_figure_data.sh sr``):
    uv run python -m scripts.paper.sr_figure_data --removed-shard 0 --removed-shards 1
    uv run python -m scripts.paper.sr_figure_data
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import tyro
import yaml
from pyprojroot import here

DEFAULT_RUNS_ROOT = here() / "data" / "results" / "human_experiment_sr-oct26" / "collected"
DEFAULT_OUT = here() / "data" / "paper" / "figure_data" / "subjective_randomness.json"
DEFAULT_WORK_DIR = here() / "data" / "paper" / "work" / "subjective_randomness"

# The live series' sampler settings (full_run_sr-oct26.yaml, ``modeling``).
LIVE_DRAWS = 3000
LIVE_TUNE = 2000
LIVE_CHAINS = 4
LIVE_TARGET_ACCEPT = 0.8


POOLED_DIR = "pooled"  # the work directory's subdirectory for all runs pooled


@dataclass
class Args:
    runs_root: Path = DEFAULT_RUNS_ROOT
    """Directory holding ``run<i>/subjective_randomness/`` run trees."""
    runs: List[str] = field(default_factory=lambda: ["run1", "run2", "run3"])
    out: Path = DEFAULT_OUT
    """Where the figure data (JSON) are written."""
    work_dir: Path = DEFAULT_WORK_DIR
    """Comparison model sets and the fit cache, one subdirectory per run."""
    draws: int = LIVE_DRAWS
    tune: int = LIVE_TUNE
    chains: int = LIVE_CHAINS
    target_accept: float = LIVE_TARGET_ACCEPT
    pooled_draws: int = 1500
    """Draws per chain of the fits on all runs pooled (Figure 3); chains are
    the runs'. Fewer than the runs' because each pooled fit holds ~3x the
    trials: at 3000 draws one person-level fit's log-likelihood is ~2 GB in
    memory and on disk, and seven are compared at once. At 1000, run 2's
    winner's bulk ESS was 82 (< 100) on one person-level parameter
    (2026-10-18)."""
    pooled_tune: int = 4000
    pooled_target_accept: float = 0.95
    """Warm-up and target acceptance of every pooled fit, all models alike.
    Longer and smaller-stepped than the runs': at the runs' settings run 2's
    winner did not converge on the pooled data (R-hat 1.66, 1,003 divergent
    transitions of 4,000; 2026-10-17)."""
    pooled_random_seed: int = 20261018
    """Random seed of every pooled fit (the starting points of its chains).
    Not the runs' 42: at 42 one of run 2's winner's four chains started where
    its step size fell to zero in warm-up, at both settings tried, so that
    chain never moved (2026-10-18); its other three chains agreed."""
    pooled_accepted_fit_problems: Dict[str, str] = field(default_factory=lambda: {
        "edge_streak_primacy_recency_pattern_rule": (
            "max R-hat 1.063 (> 1.05) and bulk ESS 48 (< 100) on one participant's "
            "z_sw and z_kappa (slot 115 of 400), whose posterior is two-humped: two "
            "chains settle on each; every population-level parameter has R-hat <= "
            "1.006 and no transition diverged (pooled fit of 2026-10-18)"
        ),
    })
    """Pooled fits whose failed convergence gate is let through, model ->
    reason; recorded in the figure data. An unreliable PSIS-LOO is never let
    through, and an entry for a model that converged raises (user decision
    2026-10-18)."""
    pooled_fit_method: str = "nuts_no_jitter"
    """``fit.method`` of every pooled fit: every chain starts at the model's
    initial point. With PyMC's jittered starts, at both seeds tried, some
    chain started so far out (log density -274,608 where the posterior's is
    about -13,930) that warm-up shrank its step size to zero and it never
    moved: one of run 2's winner's at seed 42, two of motif_stack's at
    20261018 (2026-10-18)."""
    prediction_draws: int = 2000
    """Posterior draws (thinned) behind each posterior-mean prediction."""
    discard_fits_after_run: bool = False
    """Delete each run's fits once it is scored and predicted (each run's fits
    take ~3 GB at 7,680 trials); a rerun then refits everything."""
    removed_shard: Optional[int] = None
    """Score only this shard of every run's removed models (of
    ``--removed-shards``) into ``<work-dir>/<run>/removed/`` and stop; without
    it the script assembles the figure data, which needs every removed
    model's record."""
    removed_shards: int = 1


# --- locating a run's models and data --------------------------------------


def _experiment_number(path: Path) -> int:
    match = re.fullmatch(r"experiment(\d+)", path.name)
    if match is None:
        raise ValueError(f"not an experiment directory: {path}")
    return int(match.group(1))


def final_experiment_dir(sr_dir: Path) -> Path:
    """The run's last experiment, which must have finished its model loop."""
    experiments = sorted(sr_dir.glob("experiment*"), key=_experiment_number)
    if not experiments:
        raise FileNotFoundError(f"no experiment directories in {sr_dir}")
    last = experiments[-1]
    if not (last / "model_loop" / "export_complete.json").exists():
        raise RuntimeError(f"{last} has no model_loop/export_complete.json: the run is unfinished")
    return last


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_file(sr_dir: Path, final_dir: Path, seed: str) -> Path:
    """The seed's code as the run ran it: in the final model set if it
    survived, else in the ``pruned/`` directory of the experiment that pruned
    it. Several copies must be identical."""
    copies = [final_dir / "model_loop" / "models" / f"{seed}.py"]
    copies += sorted(sr_dir.glob(f"experiment*/model_loop/models/pruned/{seed}.py"))
    found = [p for p in copies if p.exists()]
    if not found:
        raise FileNotFoundError(f"seed {seed!r} not found in {sr_dir}")
    if len({_sha256(p) for p in found}) > 1:
        raise RuntimeError(f"seed {seed!r} has differing copies in {sr_dir}: {found}")
    return found[0]


def write_model_set(models_dir: Path, files: Dict[str, Path]) -> None:
    """A model directory with exactly ``files`` (name -> source) and a manifest."""
    if models_dir.exists():
        shutil.rmtree(models_dir)
    models_dir.mkdir(parents=True)
    for name, source in files.items():
        shutil.copy(source, models_dir / f"{name}.py")
    manifest = {"models": [{"name": n, "rationale": "Paper figure comparison."} for n in files]}
    (models_dir / "models_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))


# --- comparison --------------------------------------------------------------


def _problems(row: Dict[str, Any]) -> List[str]:
    problems = []
    if row.get("not_converged"):
        problems.append("not converged")
    if row.get("loo_unreliable"):
        problems.append("unreliable PSIS-LOO")
    return problems


def require_trusted_fits(
    table: Dict[str, Dict[str, Any]], names: Sequence[str], accepted: Optional[Dict[str, str]] = None
) -> Dict[str, Dict[str, Any]]:
    """Raise if any of ``names`` did not converge or has an unreliable
    PSIS-LOO: its ELPD-LOO is then no measurement, so which model is best is
    not known, even when that model is not the one that comes out on top.

    ``accepted`` (model -> reason) lets a named model's failed convergence
    gate through, never an unreliable PSIS-LOO; it must name a compared model
    that did fail the gate, else it is stale and raises. Returns what it let
    through, with the reason, for the record."""
    accepted = accepted or {}
    stale = [n for n in accepted if n not in names or not table[n].get("not_converged")]
    if stale:
        raise RuntimeError(
            f"accepted convergence failures {stale!r} are stale: not compared, or converged"
        )
    let_through = {n: {"problems": ["not converged"], "reason": reason}
                   for n, reason in accepted.items()}
    problems = {}
    for n in names:
        left = [p for p in _problems(table[n]) if not (n in accepted and p == "not converged")]
        if left:
            problems[n] = left
    if problems:
        listed = "; ".join(f"{n}: {', '.join(p)}" for n, p in problems.items())
        raise RuntimeError(f"cannot pick the best model: {listed}")
    return let_through


def score_against_best_seed(
    responses: Path,
    models_dir: Path,
    *,
    winner: str,
    seeds: Sequence[str],
    compare: Callable[..., Dict[str, Dict[str, Any]]],
    **compare_kwargs: Any,
) -> Dict[str, Any]:
    """ELPD of the winner and every seed, and each one's paired difference
    from the best seed (``diff`` > 0: better than it) with the
    stimulus-clustered SE of that difference. Raises if any of them did not
    converge or has an unreliable PSIS-LOO (see ``require_trusted_fits``)."""
    names = [winner, *seeds]
    table = compare(responses, models_dir, names=names, **compare_kwargs)
    elpd = {n: float(table[n]["elpd_loo"]) for n in names}
    require_trusted_fits(table, names)
    best_seed = max(seeds, key=elpd.get)

    vs_best_seed = {best_seed: {"diff": 0.0, "se": 0.0}}
    for name in names:
        if name == best_seed:
            continue
        pair = compare(responses, models_dir, names=[name, best_seed], **compare_kwargs)
        lower = min((name, best_seed), key=elpd.get)
        vs_best_seed[name] = {
            "diff": elpd[name] - elpd[best_seed],
            "se": float(pair[lower]["dse_clustered"]),
        }
    return {
        "elpd": elpd,
        "best_seed": best_seed,
        "vs_best_seed": {n: vs_best_seed[n] for n in names},
    }


# --- all runs pooled (Figure 3) ------------------------------------------------


def pool_responses(responses_by_run: Dict[str, Path], out: Path) -> None:
    """Write every run's responses, in run order, to one CSV. Each run numbers
    its participants from 0, so each (run, participant) gets a new id,
    numbered in order of first appearance. The collected runs hold no
    Prolific IDs, so a person who took part in two runs counts as two."""
    from src.runtime.atomic_files import write_text_atomically
    from src.tasks.subjective_randomness import SUBJECTIVE_RANDOMNESS

    columns = list(SUBJECTIVE_RANDOMNESS.raw_columns)
    new_ids: Dict[tuple, int] = {}
    lines = [",".join(columns)]
    for run, responses in responses_by_run.items():
        with open(responses, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != columns:
                raise ValueError(f"{run}: {responses} has columns {reader.fieldnames}, not {columns}")
            for row in reader:
                key = (run, row["participant_id"])
                new_ids.setdefault(key, len(new_ids))
                row = {**row, "participant_id": str(new_ids[key])}
                lines.append(",".join(row[c] for c in columns))
    write_text_atomically(out, "\n".join(lines) + "\n")


def pooled_model_files(files_by_run: Dict[str, Dict[str, Path]]) -> Dict[str, Path]:
    """Every run's models (name -> source) in one set. A name in several runs
    must have identical code (the seeds, a winner two runs share)."""
    merged: Dict[str, Path] = {}
    for run, files in files_by_run.items():
        for name, path in files.items():
            if name in merged and _sha256(merged[name]) != _sha256(path):
                raise RuntimeError(
                    f"model {name!r} has different code in {run} ({path}) and in an earlier "
                    f"run ({merged[name]})"
                )
            merged.setdefault(name, path)
    return merged


def score_pooled(
    responses: Path,
    models_dir: Path,
    *,
    winners: Dict[str, str],
    seeds: Sequence[str],
    compare: Callable[..., Dict[str, Dict[str, Any]]],
    accepted: Optional[Dict[str, str]] = None,
    **compare_kwargs: Any,
) -> Dict[str, Any]:
    """ELPD-LOO of every run's winner (``winners``: name -> its run) and every
    seed on all runs pooled; the best of each by ELPD-LOO; and each one's
    paired difference from the best seed with its stimulus-clustered SE. Raises if any of
    them did not converge or has an unreliable PSIS-LOO, except a failed
    convergence gate ``accepted`` names (see ``require_trusted_fits``)."""
    names = [*winners, *seeds]
    table = compare(responses, models_dir, names=names, **compare_kwargs)
    elpd = {n: float(table[n]["elpd_loo"]) for n in names}
    accepted_fit_problems = require_trusted_fits(table, names, accepted)
    best_model = max(winners, key=elpd.get)
    best_seed = max(seeds, key=elpd.get)
    vs_best_seed = {best_seed: {"diff": 0.0, "se": 0.0}}
    for name in names:
        if name == best_seed:
            continue
        pair = compare(responses, models_dir, names=[name, best_seed], **compare_kwargs)
        lower = min((name, best_seed), key=elpd.get)
        vs_best_seed[name] = {"diff": elpd[name] - elpd[best_seed],
                              "se": float(pair[lower]["dse_clustered"])}
    return {
        "best_model": best_model,
        "best_seed": best_seed,
        "winner_runs": dict(winners),
        "elpd": elpd,
        "vs_best_seed": {n: vs_best_seed[n] for n in names},
        "accepted_fit_problems": accepted_fit_problems,
    }


# --- conditions --------------------------------------------------------------


def condition_table(
    rows: Sequence[Dict[str, Any]], p_left_by_model: Dict[str, np.ndarray]
) -> List[Dict[str, Any]]:
    """Per unordered pair: ``n`` trials, the observed proportion choosing the
    pair's alphabetically first sequence, and each model's mean predicted
    probability of that choice (from its per-trial ``p_left``)."""
    for name, p_left in p_left_by_model.items():
        if len(p_left) != len(rows):
            raise ValueError(
                f"{name!r} has {len(p_left)} predictions for {len(rows)} trials"
            )
    grouped: Dict[tuple, Dict[str, List[float]]] = {}
    for i, row in enumerate(rows):
        a, b = row["sequence_a"], row["sequence_b"]
        first_on_left = a <= b
        pair = (a, b) if first_on_left else (b, a)
        chose_left = int(row["chose_left"])
        if chose_left not in (0, 1):
            raise ValueError(f"chose_left must be 0 or 1, not {row['chose_left']!r}")
        cell = grouped.setdefault(pair, {"observed": [], **{m: [] for m in p_left_by_model}})
        cell["observed"].append(chose_left if first_on_left else 1 - chose_left)
        for name, p_left in p_left_by_model.items():
            p = float(p_left[i])
            cell[name].append(p if first_on_left else 1.0 - p)
    return [
        {"pair": list(pair), "n": len(cell["observed"]), **{k: float(np.mean(v)) for k, v in cell.items()}}
        for pair, cell in sorted(grouped.items())
    ]


def run_inputs(responses: Path, files: Dict[str, Path], fit_kwargs: Dict[str, Any],
               **extra: Any) -> Dict[str, Any]:
    """Everything a scored record depends on; a record whose inputs differ is
    scored again."""
    return {
        "responses_sha256": _sha256(responses),
        "model_sha256": {name: _sha256(path) for name, path in files.items()},
        "fit_settings": dict(fit_kwargs),
        **extra,
    }


def load_scored(path: Path, inputs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The run's saved record if it was made from ``inputs``, else ``None``."""
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    return saved if saved["inputs"] == inputs else None


def save_scored(path: Path, inputs: Dict[str, Any], record: Dict[str, Any]) -> None:
    from src.runtime.atomic_files import write_text_atomically

    write_text_atomically(path, json.dumps({"inputs": inputs, "record": record}))


def _read_rows(responses: Path) -> List[Dict[str, Any]]:
    with open(responses, newline="") as f:
        return list(csv.DictReader(f))


def posterior_mean_p_left(
    name: str, models_dir: Path, responses: Path, *, cache_dir: Path, prediction_draws: int, **fit_kwargs: Any
) -> np.ndarray:
    """Each trial's posterior-mean ``p_left`` under the model fitted on
    ``responses`` (a cache hit after the comparison), for the trial's own
    participant."""
    from src.models.data_binding import make_stim_data
    from src.models.pymc_inference import fit_model

    fitted = fit_model(name, models_dir, responses, cache_dir=cache_dir, **fit_kwargs)
    stim_data = make_stim_data(fitted.model, _read_rows(responses))
    draws = fitted.predict_p_left_draws(stim_data, seed=0, max_draws=prediction_draws)
    return np.asarray(draws, dtype="float64").mean(axis=0)


# --- entry point -------------------------------------------------------------


def _discard_fits(cache_dir: Path) -> None:
    for fit_file in cache_dir.glob("*.nc"):
        fit_file.unlink()


def others_at_end(posterior: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, float]:
    """Each model live at the run's end other than its winner, as its ELPD-LOO
    difference from the run's best seed. The model's ELPD-LOO is the loop's
    own, from its fit on the same final data at the same sampler settings
    (the winner's agrees with the refit here within ~1 nat of Monte Carlo
    error; the differences are ~1,000), so these need no MCMC. The seeds
    were pruned in experiment 1, so their ELPDs are this script's."""
    if posterior["n_trials"] != record["n_trials"]:
        raise RuntimeError(
            f"{record['run']}: the loop scored its models on {posterior['n_trials']} trials, "
            f"the final data have {record['n_trials']}"
        )
    if posterior["excluded_unreliable"]:
        raise RuntimeError(
            f"{record['run']}: the loop could not score {posterior['excluded_unreliable']} "
            "(unreliable PSIS-LOO)"
        )
    if record["best_model"] not in posterior["elpd_loo"]:
        raise RuntimeError(f"{record['run']}: the loop's record has no ELPD for its winner "
                           f"{record['best_model']!r}")
    best_seed = record["elpd"][record["best_seed"]]
    return {name: float(elpd) - best_seed for name, elpd in posterior["elpd_loo"].items()
            if name != record["best_model"]}


# --- the models each run removed ----------------------------------------------


def score_removed_model(name: str, models_dir: Path, responses: Path, *, cache_dir: Path,
                        **fit_kwargs: Any) -> Dict[str, Any]:
    """One removed model's ELPD-LOO from its cached fit, or why it has none."""
    from src.models.pymc_inference import convergence_problems_of, fit_models_cached

    fitted = fit_models_cached([name], models_dir=models_dir, responses_path=responses,
                               cache_dir=cache_dir, **fit_kwargs)[name]
    problems = convergence_problems_of(fitted)
    if problems:
        return {"unscored": f"not converged ({'; '.join(problems)})"}
    diag = fitted.loo_diagnostics()
    if diag.unreliable:
        return {"unscored": f"unreliable PSIS-LOO ({diag.n_bad_k} of {diag.n_points} trials "
                            f"with a high Pareto k)"}
    return {"elpd": float(diag.elpd_loo)}


def run_removed(args: Args, run: str, fit_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """A run's removed models, its final responses and how a record of one
    is keyed."""
    from scripts.paper import removed_models as rm

    sr_dir = args.runs_root / run / "subjective_randomness"
    final_dir = final_experiment_dir(sr_dir)
    loop = final_dir / "model_loop"
    live = list(json.loads((loop / "model_posterior.json").read_text())["elpd_loo"])
    seeds = json.loads((sr_dir / "starting_models.json").read_text())["starting_models"]
    removed = rm.removed_models(sr_dir, final_dir, live=live, seeds=seeds)
    recorded = rm.recorded_scores(final_dir, removed)
    responses = loop / "responses.csv"
    return {
        "removed": removed,
        "recorded": recorded,
        "to_fit": sorted(set(removed) - set(recorded)),
        "responses": responses,
        "inputs": lambda name: run_inputs(responses, {name: removed[name]["file"]}, fit_kwargs),
        "records_dir": args.work_dir / run / "removed",
        "final_dir": final_dir,
    }


def score_removed_shard(args: Args, fit_kwargs: Dict[str, Any]) -> None:
    """Score shard ``--removed-shard`` of every run's removed models, taken
    together (in run order, then by name)."""
    from scripts.paper import removed_models as rm

    by_run = {run: run_removed(args, run, fit_kwargs) for run in args.runs}
    extras = pooled_extras(args, fit_kwargs)
    items = [(run, name) for run in args.runs for name in by_run[run]["to_fit"]]
    items += [(POOLED_DIR, label) for label in sorted(extras)]
    mine = rm.shard_of(items, args.removed_shard, args.removed_shards)
    pooled_labels = [label for group, label in mine if group == POOLED_DIR]
    if pooled_labels:
        responses = write_pooled_responses(args)
        pooled_fit = pooled_fit_settings(args, fit_kwargs)
        models_dir = args.work_dir / POOLED_DIR / "shards" / f"shard_{args.removed_shard}" / "models"
        cache_dir = args.work_dir / POOLED_DIR / "fit_cache"
        print(f"[pooled] scoring {len(pooled_labels)} models on all runs pooled", flush=True)
        rm.score_shard(
            {label: extras[label]["file"] for label in pooled_labels},
            models_dir=models_dir, records_dir=args.work_dir / POOLED_DIR / "removed",
            inputs=lambda label: run_inputs(responses, {label: extras[label]["file"]}, pooled_fit),
            training_files=[("all runs pooled", responses)],
            score=lambda label: score_removed_model(label, models_dir, responses,
                                                    cache_dir=cache_dir, **pooled_fit),
            cache_dir=cache_dir, fit_kwargs=pooled_fit,
        )
    for run in args.runs:
        names = [name for r, name in mine if r == run]
        if not names:
            continue
        info = by_run[run]
        models_dir = args.work_dir / run / "shards" / f"shard_{args.removed_shard}" / "models"
        cache_dir = args.work_dir / run / "fit_cache"
        print(f"[{run}] scoring {len(names)} removed models", flush=True)
        rm.score_shard(
            {n: info["removed"][n]["file"] for n in names},
            models_dir=models_dir, records_dir=info["records_dir"], inputs=info["inputs"],
            training_files=[("all data", info["responses"])],
            score=lambda n: score_removed_model(n, models_dir, info["responses"],
                                                cache_dir=cache_dir, **fit_kwargs),
            cache_dir=cache_dir, fit_kwargs=fit_kwargs,
        )


def pruned_fields(info: Dict[str, Any], scored: Dict[str, Any], record: Dict[str, Any]) -> Dict[str, Any]:
    """A run's removed models in its figure record: each scored one as its
    ELPD-LOO difference from the run's best seed, and its place in the order
    the run admitted models (``pruned_order``)."""
    from scripts.paper import removed_models as rm

    best_seed = record["elpd"][record["best_seed"]]
    elpd = {**scored["elpd"], **info["recorded"]}
    return {
        "pruned_during_run": {n: e - best_seed for n, e in elpd.items()},
        "pruned_elpd": elpd,
        "pruned_unscored": scored["unscored"],
        "pruned_outcome": {n: r["outcome"] for n, r in info["removed"].items()},
        "pruned_source": {n: "loop" if n in info["recorded"] else "refit" for n in info["removed"]},
        "pruned_order": rm.pruned_order(info["final_dir"], sorted(elpd)),
    }


def score_run(args: Args, run: str, fit_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Figure 2's record of one run: its winner and seeds on its final data.
    Also returns the run's model files and responses for the pooled fits."""
    from src.model_comparison.posterior import compare_table
    from src.models.pymc_inference import clear_fit_cache

    sr_dir = args.runs_root / run / "subjective_randomness"
    final_dir = final_experiment_dir(sr_dir)
    loop = final_dir / "model_loop"
    posterior = json.loads((loop / "model_posterior.json").read_text())
    winner = posterior["best_model"]
    seeds = json.loads((sr_dir / "starting_models.json").read_text())["starting_models"]
    if winner in seeds:
        raise RuntimeError(f"{run}: the winner {winner!r} is a seed; nothing to compare")
    files = {winner: loop / "models" / f"{winner}.py"}
    files.update({s: seed_file(sr_dir, final_dir, s) for s in seeds})
    responses = loop / "responses.csv"
    sources = {"files": files, "responses": responses, "winner": winner, "seeds": seeds}

    inputs = run_inputs(responses, files, fit_kwargs)
    scored_path = args.work_dir / run / "scored.json"
    saved = load_scored(scored_path, inputs)
    if saved is not None:
        print(f"[{run}] reusing its scored record {scored_path}", flush=True)
        record = saved["record"]
        return {"record": {**record, "others_at_end": others_at_end(posterior, record)}, **sources}

    models_dir = args.work_dir / run / "models"
    cache_dir = args.work_dir / run / "fit_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_model_set(models_dir, files)
    print(f"[{run}] fitting {winner} and seeds {seeds} on {responses}", flush=True)
    scored = score_against_best_seed(
        responses, models_dir, winner=winner, seeds=seeds,
        compare=compare_table, cache_dir=cache_dir, **fit_kwargs,
    )
    record = {
        "run": run,
        "final_experiment": final_dir.name,
        "n_trials": len(_read_rows(responses)),
        "best_model": winner,
        "recorded_winner_elpd": posterior.get("elpd_loo", {}).get(winner),
        **scored,
    }
    save_scored(scored_path, inputs, record)
    # A loaded fit holds its pointwise log-likelihood (~0.7 GB at 7,680
    # trials and 12,000 draws).
    clear_fit_cache()
    if args.discard_fits_after_run:
        _discard_fits(cache_dir)
        print(f"[{run}] scored; its fits deleted", flush=True)
    return {"record": {**record, "others_at_end": others_at_end(posterior, record)}, **sources}


POOLED_RECORD_VERSION = 2  # 2: the pooled record has vs_best_seed (2026-10-20)
POOLED_LABEL_SEP = "__"  # a pooled extra model is <run>__<name>: names repeat across runs


def write_pooled_responses(args: Args) -> Path:
    """Every run's final responses in one CSV (``pool_responses``), written
    atomically under the work directory; its path."""
    responses = args.work_dir / POOLED_DIR / "responses.csv"
    responses.parent.mkdir(parents=True, exist_ok=True)
    pool_responses({run: final_experiment_dir(args.runs_root / run / "subjective_randomness")
                    / "model_loop" / "responses.csv" for run in args.runs}, responses)
    return responses


def pooled_fit_settings(args: Args, fit_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    return {**fit_kwargs, "draws": args.pooled_draws, "tune": args.pooled_tune,
            "target_accept": args.pooled_target_accept, "random_seed": args.pooled_random_seed,
            "method": args.pooled_fit_method}


def pooled_extras(args: Args, fit_kwargs: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Every run's other models live at the end and its removed models, for
    Figure 2's pooled view: ``<run>__<name>`` -> ``file``, ``run``, ``name``,
    ``role`` (``live`` / ``removed``) and a removed model's ``outcome``."""
    from scripts.paper import removed_models as rm

    extras: Dict[str, Dict[str, Any]] = {}
    for run in args.runs:
        info = run_removed(args, run, fit_kwargs)
        order = rm.admission_order(info["final_dir"])
        loop = info["final_dir"] / "model_loop"
        posterior = json.loads((loop / "model_posterior.json").read_text())
        for name in posterior["elpd_loo"]:
            if name != posterior["best_model"]:
                extras[f"{run}{POOLED_LABEL_SEP}{name}"] = {
                    "file": loop / "models" / f"{name}.py", "run": run, "name": name, "role": "live"}
        for name, entry in info["removed"].items():
            extras[f"{run}{POOLED_LABEL_SEP}{name}"] = {
                "file": entry["file"], "run": run, "name": name, "role": "removed",
                "outcome": entry["outcome"], "order": order[name]}
    return extras


def pooled_extras_fields(extras: Dict[str, Dict[str, Any]], scored: Dict[str, Any],
                         pooled: Dict[str, Any]) -> Dict[str, Any]:
    """The pooled extras in the pooled record, as ELPD-LOO differences from
    the best seed on the pooled data."""
    best_seed = pooled["elpd"][pooled["best_seed"]]

    def diffs(role: str) -> Dict[str, float]:
        return {label: e - best_seed for label, e in scored["elpd"].items() if extras[label]["role"] == role}

    return {
        "others_at_end": diffs("live"),
        "pruned_during_run": diffs("removed"),
        "extras_unscored": scored["unscored"],
        "pruned_outcome": {label: x["outcome"] for label, x in extras.items() if x["role"] == "removed"},
        # Each removed model's place in the order its own run admitted models.
        "pruned_order": {label: extras[label]["order"] for label in scored["elpd"]
                         if extras[label]["role"] == "removed"},
    }


def score_all_runs_pooled(args: Args, scored_runs: Sequence[Dict[str, Any]],
                          fit_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Figure 3's data: every run's winner and the seeds fitted on all runs'
    data, and per condition the best winner's and best seed's predictions."""
    from src.model_comparison.posterior import compare_table
    from src.models.pymc_inference import clear_fit_cache

    work = args.work_dir / POOLED_DIR
    responses = write_pooled_responses(args)
    files = pooled_model_files({r["record"]["run"]: r["files"] for r in scored_runs})
    winners = {r["winner"]: r["record"]["run"] for r in scored_runs}
    if len(winners) != len(scored_runs):
        raise RuntimeError(f"two runs share a winner's name: {[r['winner'] for r in scored_runs]}")
    seeds = scored_runs[0]["seeds"]
    if any(r["seeds"] != seeds for r in scored_runs):
        raise RuntimeError(f"the runs started from different seeds: {[r['seeds'] for r in scored_runs]}")

    pooled_fit = pooled_fit_settings(args, fit_kwargs)
    inputs = run_inputs(responses, files, pooled_fit, prediction_draws=args.prediction_draws,
                        accepted_fit_problems=args.pooled_accepted_fit_problems,
                        record_version=POOLED_RECORD_VERSION)
    scored_path = work / "scored.json"
    saved = load_scored(scored_path, inputs)
    if saved is not None:
        print(f"[pooled] reusing its scored record {scored_path}", flush=True)
        return saved["record"]

    models_dir = work / "models"
    cache_dir = work / "fit_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    write_model_set(models_dir, files)
    print(f"[pooled] fitting {list(winners)} and seeds {seeds} on {responses}", flush=True)
    scored = score_pooled(responses, models_dir, winners=winners, seeds=seeds,
                          compare=compare_table, accepted=args.pooled_accepted_fit_problems,
                          cache_dir=cache_dir, **pooled_fit)
    rows = _read_rows(responses)
    predictions = {
        role: posterior_mean_p_left(
            scored[role], models_dir, responses, cache_dir=cache_dir,
            prediction_draws=args.prediction_draws, **pooled_fit,
        )
        for role in ("best_model", "best_seed")
    }
    record = {
        "pooled": {
            "runs": [r["record"]["run"] for r in scored_runs],
            "n_trials": len(rows),
            "n_participants": len({row["participant_id"] for row in rows}),
            "fit_settings": pooled_fit,
            **scored,
        },
        "conditions": condition_table(rows, predictions),
    }
    save_scored(scored_path, inputs, record)
    clear_fit_cache()
    if args.discard_fits_after_run:
        _discard_fits(cache_dir)
        print("[pooled] scored and predicted; its fits deleted", flush=True)
    return record


def main(args: Args) -> None:
    fit_kwargs = {
        "draws": args.draws,
        "tune": args.tune,
        "chains": args.chains,
        "target_accept": args.target_accept,
    }
    if args.removed_shard is not None:
        score_removed_shard(args, fit_kwargs)
        return
    from scripts.paper import removed_models as rm

    # Every removed model's record first: a missing one stops the script
    # before any fit.
    removed = {run: run_removed(args, run, fit_kwargs) for run in args.runs}
    removed_scored = {
        run: rm.collect(info["to_fit"], records_dir=info["records_dir"], inputs=info["inputs"])
        for run, info in removed.items()
    }
    extras = pooled_extras(args, fit_kwargs)
    pooled_responses = write_pooled_responses(args)
    pooled_fit = pooled_fit_settings(args, fit_kwargs)
    extras_scored = rm.collect(
        sorted(extras), records_dir=args.work_dir / POOLED_DIR / "removed",
        inputs=lambda label: run_inputs(pooled_responses, {label: extras[label]["file"]}, pooled_fit))
    scored_runs = [score_run(args, run, fit_kwargs) for run in args.runs]
    for r in scored_runs:
        run = r["record"]["run"]
        r["record"].update(pruned_fields(removed[run], removed_scored[run], r["record"]))
    pooled = score_all_runs_pooled(args, scored_runs, fit_kwargs)
    pooled["pooled"].update(pooled_extras_fields(extras, extras_scored, pooled["pooled"]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "domain": "subjective_randomness",
                "scoring": "ELPD-LOO (PSIS) on each run's final pooled data",
                "fit_settings": fit_kwargs,
                "runs": [r["record"] for r in scored_runs],
                **pooled,
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
