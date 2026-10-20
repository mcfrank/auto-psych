"""Generalization of each run's models to new people and to the other run, with two-way bootstrap.

Per source run S and target run T, for the seeds and each experiment's winner
of S (full, and with every person-level parameter pooled):

* ``within`` (T = S): leave-one-participant-out ELPD on S's data, by PSIS over
  people with the person-level parameters integrated out
  (``lopo_analysis.py``).
* ``refit`` (T != S): the same on T's data, every model refit there
  (``fit_transfer_models.py``): does the functional form generalize?
* ``fixed`` (T != S): S's fits predict every person of T as a new participant,
  ``sum_i log E_{theta ~ p(theta | S), z ~ N(0, 1)} [prod_t p(y_it | theta, z)]``:
  does the fitted theory generalize? The integral over z is importance-sampled
  as in ``lopo_analysis.py``, the proposal for each of T's people centred on
  their posterior z in the T refit (any proposal is valid; that one is close
  to where the integrand is).

**Two-way bootstrap.** The people and the stimulus pairs of T are both
samples, crossed (everyone in an experiment answers the same pairs). Each
replicate draws people with replacement within each experiment (weight
w_p = times drawn) and stimulus pairs with replacement (v_s, pairs grouped as
``stimulus_clusters``) and rescores every model with trial (p, s) weighted
w_p v_s: a person's trial weights enter their likelihood inside the integral
over their z, and the LOPO importance weights are recomputed. The weights are
drawn once per target run (saved, the same for every model and analysis), so
model differences are paired. The fits are not redone per replicate: this is
the uncertainty from T's people and pairs, not from S's. The replicates use a
subsample (``keep_every``-th posterior draw, the first ``keep_k`` z draws);
the point estimates use everything, and intervals are the point estimate
plus the replicates' deviation from the unit-weight replicate.

Writes ``results/generalization/<analysis>_run<S>_on_run<T>.csv`` (point
estimates), ``..._people.csv``, ``..._boot.npz`` (per model: replicate
totals and the unit-weight total) and ``boot_weights_run<T>.npz``. On the
cluster each task scores some of the models (``--models``) into
``parts/<analysis>_run<S>_on_run<T>__<part>.*`` and ``--merge`` assembles
them, refusing a missing or duplicated model.

Usage:
    uv run python data/results/human_experiment_sr-oct26/analysis/generalization_analysis.py \\
        --analysis fixed --source 1 --target 2
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tyro
from pyprojroot import here
from scipy.special import logsumexp

ANALYSIS_ROOT = here() / "data/results/human_experiment_sr-oct26/analysis"
sys.path.insert(0, str(here()))
sys.path.insert(0, str(ANALYSIS_ROOT))

# The pipeline sizes its concurrent fits by the scheduler's affinity mask
# (``allocated_cpus``), which only Linux has; on a laptop, every CPU.
if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid: set(range(os.cpu_count()))

from fit_transfer_models import (  # noqa: E402
    GENERALIZATION,
    responses,
    run_root,
    seed_record,
    transfer_model_names,
    transfer_models_dir,
)
from lopo_analysis import (  # noqa: E402
    free_posterior,
    loglik,
    marginal_loglik,
    predict,
    proposal_moments,
    psis_lopo,
    self_check,
    slot_vectors,
)



@dataclass
class Args:
    """Within-run LOPO, refit and fixed-parameter transfer, with a two-way bootstrap."""

    analysis: Literal["within", "refit", "fixed"]
    source: int
    target: int
    models: tuple[str, ...] = ()
    """Score only these models (default: all of the source run's), into ``parts/`` as ``--part``."""
    part: str | None = None
    merge: bool = False
    """Assemble the analysis' parts into its tables and stop."""
    fit_seeds: Path = GENERALIZATION / "fit_seeds"
    """The converged sampler seed of every fit (``fit_transfer_models.py fit``)."""
    draws: int = 1000
    k: int = 400
    alpha: float = 0.1
    inflate: float = 1.5
    seed: int = 20261005
    keep_every: int = 5
    """Bootstrap: every n-th of the ``draws`` posterior draws."""
    keep_k: int = 100
    """Bootstrap: the first n of the ``k`` z draws."""
    n_boot: int = 1000
    boot_seed: int = 20261006
    weights_only: bool = False
    """Draw and save the target run's bootstrap weights, then stop (the batch
    setup does this once, so concurrent tasks only read them)."""
    out_dir: Path = GENERALIZATION
    cache_dir: Path = ANALYSIS_ROOT / "mcmc_cache"
    # The fits' settings (fit_transfer_models.py).
    fit_draws: int = 3000
    tune: int = 2000
    chains: int = 4
    target_accept: float = 0.8


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def boot_weights(target: int, rows, people, args: Args) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(people weights (B, P), stimulus weights (B, n_stimuli), each person's
    trials' stimulus index (P, most trials of anyone)), drawn once per target
    run and saved. A person seen in two experiments has twice the trials; the
    others' rows are padded with index n_stimuli, a stimulus of weight 0. That
    person is resampled within their first experiment."""
    from src.models.clustered_se import stimulus_clusters

    path = args.out_dir / f"boot_weights_run{target}.npz"
    clusters = np.asarray(stimulus_clusters(rows))
    _, stim_index = np.unique(clusters, return_inverse=True)
    by_person = {p: [] for p in people}
    for r, s in zip(rows, stim_index):
        by_person[r["participant_id"]].append(s)
    n_stim = stim_index.max() + 1
    most = max(len(v) for v in by_person.values())
    stim_of = np.array([by_person[p] + [n_stim] * (most - len(by_person[p])) for p in people])
    if path.exists():
        saved = np.load(path)
        if (saved["people"] != np.array(people, dtype=int)).any() or (saved["stim_of"] != stim_of).any():
            raise ValueError(f"{path} was drawn for other people or stimuli")
        if len(saved["people_w"]) != args.n_boot:
            raise ValueError(f"{path} holds {len(saved['people_w'])} replicates, not --n-boot {args.n_boot}")
        return saved["people_w"], saved["stim_w"], stim_of
    experiment_of = {}
    for e in (1, 2, 3):
        for r in read_rows(run_root(target) / f"experiment{e}/data/responses.csv"):
            experiment_of.setdefault(r["participant_id"], e)
    rng = np.random.default_rng(args.boot_seed + target)
    people_w = np.zeros((args.n_boot, len(people)))
    for e in (1, 2, 3):
        members = np.array([i for i, p in enumerate(people) if experiment_of[p] == e])
        picks = rng.integers(0, len(members), size=(args.n_boot, len(members)))
        for b in range(args.n_boot):
            np.add.at(people_w[b], members[picks[b]], 1)
    stim_w = np.zeros((args.n_boot, n_stim))
    picks = rng.integers(0, n_stim, size=(args.n_boot, n_stim))
    for b in range(args.n_boot):
        np.add.at(stim_w[b], picks[b], 1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(path, people=np.array(people, dtype=int), stim_of=stim_of, people_w=people_w, stim_w=stim_w)
    return people_w, stim_w, stim_of


def replicate_totals(trial_ll, log_ratio, people_w, stim_w, stim_of, lopo: bool, n_chains: int) -> tuple[np.ndarray, float]:
    """Bootstrap totals (B,) and the unit-weight total.

    ``trial_ll`` (P, draws, K, trials), ``log_ratio`` (P, K) or None.
    """
    P, S, K, T = trial_ll.shape
    flat = trial_ll.reshape(P, S * K, T)
    ratio = 0.0 if log_ratio is None else log_ratio[:, None, :K]

    def total(v: np.ndarray, w: np.ndarray) -> float:
        v = np.append(v, 0.0)  # the padding stimulus
        a = np.matmul(flat, v[stim_of][:, :, None].astype(np.float32))[..., 0].reshape(P, S, K) + ratio
        per_draw = logsumexp(a, axis=2) - np.log(K)  # (P, S)
        if lopo:
            elpd, _ = psis_lopo(per_draw.T, n_chains)
        else:
            elpd = logsumexp(per_draw, axis=1) - np.log(S)
        return float(w @ elpd)

    unit = total(np.ones(stim_w.shape[1]), np.ones(P))
    totals = np.array([total(stim_w[b], people_w[b]) for b in range(len(people_w))])
    if not (np.isfinite(totals).all() and np.isfinite(unit)):
        raise FloatingPointError("non-finite bootstrap totals")
    return totals, unit


def load(name, models_dir, run: int, args: Args):
    """``name``'s fit on run ``run``'s all-data responses; it must be converged."""
    from src.models.pymc_inference import convergence_problems_of, fit_model

    data = responses(run)
    record = seed_record(name, run, args.fit_seeds)
    if not record.exists():
        raise FileNotFoundError(f"no converged fit recorded for {name} on run {run}: {record}")
    seed = json.loads(record.read_text())["seed"]
    fitted = fit_model(
        name, models_dir, data, cache_dir=args.cache_dir, draws=args.fit_draws, tune=args.tune,
        chains=args.chains, target_accept=args.target_accept, random_seed=seed,
    )
    problems = convergence_problems_of(fitted)
    if problems:
        raise RuntimeError(f"{name} on run {run}: fit at seed {seed} has convergence problems {problems}")
    return fitted, seed


def main(args: Args) -> None:
    from src.models.pymc_inference import clear_fit_cache

    if (args.analysis == "within") != (args.source == args.target):
        raise ValueError("'within' scores a run on its own data; 'refit' and 'fixed' on the other run's")
    tag = f"{args.analysis}_run{args.source}_on_run{args.target}"
    names = transfer_model_names(args.source)
    if args.merge:
        merge(args.out_dir, tag, names)
        return
    if bool(args.models) != bool(args.part):
        raise ValueError("--models and --part go together (a subset is written as a part)")
    if args.models:
        unknown = sorted(set(args.models) - set(names))
        if unknown:
            raise ValueError(f"--models names unknown models: {unknown}")
        names = [n for n in names if n in args.models]
    target_rows = read_rows(responses(args.target))
    people = sorted({r["participant_id"] for r in target_rows}, key=int)
    rows_by_person = {p: [r for r in target_rows if r["participant_id"] == p] for p in people}
    ordered = [r for p in people for r in rows_by_person[p]]
    people_w, stim_w, stim_of = boot_weights(args.target, ordered, people, args)
    if args.weights_only:
        print(f"[weights] run {args.target}: {len(people_w)} replicates in {args.out_dir}")
        return
    keep_draws = np.arange(0, args.draws, args.keep_every)
    # Where each model is fitted for scoring (theta) and, for 'fixed', the
    # target refit that gives the proposal.
    target_dir = transfer_models_dir(args.source, args.target)
    theta_dir, theta_run = (
        (transfer_models_dir(args.source, args.source), args.source)
        if args.analysis == "fixed"
        else (target_dir, args.target)
    )
    lopo = args.analysis != "fixed"

    records, person_rows, boot = [], [], {}
    for name in names:
        started = time.time()
        fitted, seed = load(name, theta_dir, theta_run, args)
        idata = free_posterior(fitted, args.draws)
        n_chains = idata.posterior.sizes["chain"]
        vectors = slot_vectors(fitted.model)
        rng = np.random.default_rng(args.seed)
        info: dict = {}
        if vectors:
            if lopo:
                self_check(fitted, idata, vectors, rows_by_person, people)
                moments = None
            else:
                refit, _ = load(name, target_dir, args.target, args)
                if slot_vectors(refit.model) != vectors:
                    raise ValueError(f"{name}: the target refit has other person-level vectors")
                moments = proposal_moments(free_posterior(refit, args.draws), vectors, people)
                del refit
            terms, info = marginal_loglik(
                fitted, idata, vectors, rows_by_person, people, args, rng,
                moments=moments, keep_draws=keep_draws, keep_k=args.keep_k,
            )
            trial_ll, log_ratio = info.pop("trial_ll"), info.pop("log_ratio")
            per_draw = logsumexp(terms, axis=2) - np.log(args.k)  # (draws, P)
        else:
            ll = loglik(predict(fitted, idata, ordered), np.array([int(r["chose_left"]) for r in ordered]))
            counts = [len(rows_by_person[p]) for p in people]
            trial_ll = np.zeros((len(people), len(keep_draws), 1, max(counts)), dtype=np.float32)
            starts = np.cumsum([0, *counts])
            for i in range(len(people)):
                trial_ll[i, :, 0, : counts[i]] = ll[keep_draws, starts[i] : starts[i + 1]]
            log_ratio = None
            per_draw = np.stack([ll[:, starts[i] : starts[i + 1]].sum(axis=1) for i in range(len(people))], axis=1)
        if lopo:
            elpd_i, k_i = psis_lopo(per_draw, n_chains)
        else:
            elpd_i, k_i = logsumexp(per_draw, axis=0) - np.log(per_draw.shape[0]), np.full(len(people), np.nan)
        totals, unit = replicate_totals(trial_ll, log_ratio, people_w, stim_w, stim_of, lopo, n_chains)
        boot[name] = totals
        boot[f"{name}__unit"] = np.array(unit)
        records.append(
            {
                "model": name,
                "elpd": float(elpd_i.sum()),
                "se_people": float(np.sqrt(len(elpd_i) * elpd_i.var())),
                "se_two_way": float(totals.std()),
                "elpd_subsample": unit,
                "max_pareto_k": float(np.nanmax(k_i)) if lopo else np.nan,
                "n_k_over_0.7": int((k_i > 0.7).sum()) if lopo else 0,
                "n_person_vectors": len(vectors),
                "n_people": len(people),
                "n_trials": len(ordered),
                **info,
                "random_seed": seed,
                "seconds": round(time.time() - started),
            }
        )
        person_rows += [{"model": name, "participant_id": p, "elpd": e} for p, e in zip(people, elpd_i)]
        print(
            f"[{args.analysis}] run {args.source} on run {args.target} {name}: {elpd_i.sum():.1f} "
            f"(two-way SE {totals.std():.1f}, subsample {unit:.1f}, {records[-1]['seconds']} s)",
            flush=True,
        )
        del fitted, idata, trial_ll
        clear_fit_cache()

    stem = args.out_dir / "parts" / f"{tag}__{args.part}" if args.part else args.out_dir / tag
    stem.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(records).sort_values("elpd", ascending=False)
    table.to_csv(f"{stem}.csv", index=False)
    pd.DataFrame(person_rows).to_csv(f"{stem}_people.csv", index=False)
    np.savez(f"{stem}_boot.npz", **boot)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(table.to_string(index=False, float_format="%.2f"))


def merge(out_dir: Path, tag: str, names: list[str]) -> None:
    """Assemble ``parts/<tag>__*`` into the analysis' tables: every model once."""
    parts = sorted((out_dir / "parts").glob(f"{tag}__*.csv"))
    parts = [p for p in parts if not p.name.endswith("_people.csv")]
    if not parts:
        raise FileNotFoundError(f"no parts for {tag} in {out_dir / 'parts'}")
    tables = [pd.read_csv(p) for p in parts]
    table = pd.concat(tables).sort_values("elpd", ascending=False)
    counts = table.model.value_counts()
    if set(counts.index) != set(names) or (counts > 1).any():
        missing = sorted(set(names) - set(counts.index))
        raise ValueError(f"{tag}: parts cover {sorted(counts.index)} (missing {missing}, duplicated "
                         f"{sorted(counts[counts > 1].index)})")
    people = pd.concat([pd.read_csv(str(p).replace(".csv", "_people.csv")) for p in parts])
    boot = {}
    for p in parts:
        with np.load(str(p).replace(".csv", "_boot.npz")) as part:
            boot.update({k: part[k] for k in part.files})
    table.to_csv(out_dir / f"{tag}.csv", index=False)
    people.to_csv(out_dir / f"{tag}_people.csv", index=False)
    np.savez(out_dir / f"{tag}_boot.npz", **boot)
    print(f"[merge] {tag}: {len(parts)} parts, {len(table)} models")


if __name__ == "__main__":
    main(tyro.cli(Args))
