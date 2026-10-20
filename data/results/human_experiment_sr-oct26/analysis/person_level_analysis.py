"""How much of the discovered models' gain comes from person-level parameters?

Two analyses over the same model set: the seeds and the run's winners as they
are, plus the ablations ``make_ablation_models.py`` writes (seeds given a
person-level sensitivity and lapse; winners with their person-level
parameters pooled, in full or keeping the lapse / lapse + sensitivity).

1. **Structure x personalisation (``scheme=all``)**: every model fit on all of
   run 1's data (7,680 trials), scored by trial-level PSIS-LOO (what the loop
   selects by) and by a **new-participant** score.
2. **Held-out people (``scheme=heldout``)**: every model fit on experiments
   1-2 (5,120 trials) and scored on experiment 3's 40 new people, on
   experiment 3's new stimuli. The experiment-3 winners and the four tied
   models were selected after the loop saw experiment 3's data, so for them
   this is not clean; for the seeds, the experiment 1-2 winners and their
   ablations it is.

The new-participant score of a person is ``log E[prod_t p(y_t | theta_new)]``
over the posterior and a fresh draw of the person's parameters from the
model's population distribution: each person's trials are predicted as a
participant slot no data reached (``MAX_PARTICIPANTS`` = 400 slots; slots
past the data hold prior draws of the person-level z's, i.e. population
draws), as the design stage predicts new participants
(``eig._new_participant_draws``). Several rounds of distinct slots add
Monte Carlo samples. A population-level model's score uses its posterior
draws alone. In ``scheme=all`` each person also helped fit the population (a
leak of 1 person in 118); ``scheme=heldout`` has none.

Also reported per scheme: per-stimulus RMSE / R² of the **population**
prediction (each trial's p averaged over draws and new-participant slots),
which no person-specific fit can inflate.

Writes ``<out_dir>/person_level_<scheme>.csv`` and ``..._people.csv`` (the
per-person scores); for ``--run 2``, ``person_level_<scheme>_run2.csv`` etc.

Usage:
    uv run python data/results/human_experiment_sr-oct26/analysis/person_level_analysis.py
"""

from __future__ import annotations

import csv
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tyro
from pyprojroot import here
from scipy.special import logsumexp

sys.path.insert(0, str(here()))

# The pipeline sizes its concurrent fits by the scheduler's affinity mask
# (``allocated_cpus``), which only Linux has; on a laptop, every CPU.
if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid: set(range(os.cpu_count()))

ANALYSIS_ROOT = here() / "data/results/human_experiment_sr-oct26/analysis"
SEEDS = ("falk_konold_dp", "finite_experience_occurrence", "local_representativeness", "motif_stack")
# Per run: name -> experiment whose model_loop/models/ holds it
WINNERS = {
    1: {
        "individual_streak_aversion_lapse": 1,
        "most_lopsided_stretch_aversion": 2,
        "personal_pattern_detection_gain": 3,
        "capacity_blur_pattern_gain": 3,
        "returns_to_balance_expectation": 3,
        "sensitivity_linked_consistency": 3,
    },
    2: {
        "graded_periodicity_personal_ideal": 1,
        "tally_span_switch_ideal_periodic_unit": 2,
        "edge_streak_primacy_recency_pattern_rule": 3,
    },
}
FIRST_FREE_SLOT = 120  # past every participant id of either run (run 1: 0-117, run 2: 0-119)


def run_root(run: int) -> Path:
    return here() / f"data/results/human_experiment_sr-oct26/collected/run{run}/subjective_randomness"


def ablation_dir(run: int) -> Path:
    return ANALYSIS_ROOT / ("ablation_models" if run == 1 else f"ablation_models_run{run}")
MAX_PARTICIPANTS = 400


@dataclass
class Args:
    """Structure x personalisation ablation and new-participant scores."""

    scheme: Literal["all", "heldout", "both"] = "both"
    run: int = 1
    out_dir: Path = ANALYSIS_ROOT / "results"
    work_dir: Path | None = None
    """Default: ``fits/person_level`` (run 1), ``fits/person_level_run<N>``."""
    cache_dir: Path = ANALYSIS_ROOT / "mcmc_cache"
    # The live run's settings (full_run_sr-oct26.yaml), as fit_run_models.py.
    draws: int = 3000
    tune: int = 2000
    chains: int = 4
    target_accept: float = 0.8
    random_seed: int = 20261003
    seed_overrides: dict[str, int] = field(default_factory=dict)
    """Sampler seeds keyed ``<model>@experiment<N>`` (the experiment whose
    cumulative data the fit uses: 3 for ``all``, 2 for ``heldout``) or
    ``<model>``, as fit_run_models.py; recorded per row."""
    max_draws: int = 3000
    """Posterior draws per new-participant round."""
    fit_batch: int = 3
    """Models sampled (and held in memory) at once."""
    only: tuple[str, ...] = ()
    """Fit and score only these models, written to ``..._only.csv``: their
    per-person scores are comparable with a full run's (same people, draws
    and seeds), and the full run's files are left alone."""


def assemble_models_dir(work: Path, run: int) -> list[str]:
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    root, abl = run_root(run), ablation_dir(run)
    sources = [(s, root / "experiment1/cognitive_models_input" / f"{s}.py") for s in SEEDS]
    sources += [(w, root / f"experiment{e}/model_loop/models/{w}.py") for w, e in WINNERS[run].items()]
    ablations = [
        line.split("name: ")[1].strip()
        for line in (abl / "models_manifest.yaml").read_text().splitlines()
        if "name:" in line
    ]
    sources += [(a, abl / f"{a}.py") for a in ablations]
    for name, path in sources:
        if not path.exists():
            raise FileNotFoundError(f"model source missing: {path}")
        shutil.copyfile(path, work / f"{name}.py")
    (work / "models_manifest.yaml").write_text(
        "models:\n" + "".join(f"  - name: {n}\n" for n, _ in sources), encoding="utf-8"
    )
    return [n for n, _ in sources]


def describe(name: str) -> dict:
    """The model's place in the 2x2: structure and person-level parameters."""
    base, _, variant = name.partition("__")
    structure = "seed" if base in SEEDS else "discovered"
    if structure == "seed":
        person = "sensitivity+lapse" if variant == "personal" else "none"
    else:
        person = {
            "": "full", "pooled": "none", "lapse_only": "lapse", "sens": "sensitivity", "sens_lapse": "sensitivity+lapse",
        }[variant]
    return {"base_model": base, "structure": structure, "person_level": person}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def new_participant_draws(fitted, rows, people, max_draws: int, seed: int) -> np.ndarray:
    """(samples, trials) p_left with each person predicted as a new participant.

    Each round gives every person a distinct slot no data reached; a model
    without a participant effect is predicted once (its draws are the samples).
    """
    from src.models.data_binding import make_stim_data
    from src.models.model_loading import pm_data_inputs

    if "participant_id" not in pm_data_inputs(fitted.model):
        return fitted.predict_p_left_draws(make_stim_data(fitted.model, rows), seed=seed, max_draws=max_draws)
    index = {p: i for i, p in enumerate(people)}
    rounds = (MAX_PARTICIPANTS - FIRST_FREE_SLOT) // len(people)
    if rounds < 1:
        raise ValueError(f"{len(people)} people do not fit in the free slots")
    out = []
    for r in range(rounds):
        base = FIRST_FREE_SLOT + r * len(people)
        slotted = [{**row, "participant_id": base + index[row["participant_id"]]} for row in rows]
        out.append(fitted.predict_p_left_draws(make_stim_data(fitted.model, slotted), seed=seed, max_draws=max_draws))
    return np.concatenate(out, axis=0)


def person_scores(p: np.ndarray, y: np.ndarray, person: np.ndarray, people: list[str]) -> np.ndarray:
    """log E_samples[prod_t p(y_t)] per person."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    ll = np.where(y[None, :] == 1, np.log(p), np.log1p(-p))  # (samples, trials)
    out = np.empty(len(people))
    for i, who in enumerate(people):
        per_sample = ll[:, person == who].sum(axis=1)
        out[i] = logsumexp(per_sample) - np.log(len(per_sample))
    return out


def stimulus_fit(rows, p_mean: np.ndarray) -> tuple[float, float]:
    df = pd.DataFrame(rows)
    first = np.minimum(df.sequence_a.values, df.sequence_b.values)
    a_first = df.sequence_a.values == first
    y = df.chose_left.astype(int).values
    agg = pd.DataFrame(
        {
            "stim": [" | ".join(sorted(p)) for p in zip(df.sequence_a, df.sequence_b)],
            "obs": np.where(a_first, y, 1 - y),
            "pred": np.where(a_first, p_mean, 1 - p_mean),
        }
    ).groupby("stim")[["obs", "pred"]].mean()
    resid = agg.obs - agg.pred
    return float(np.sqrt((resid**2).mean())), float(1 - (resid**2).sum() / ((agg.obs - agg.obs.mean()) ** 2).sum())


def run_scheme(scheme: str, args: Args) -> None:
    from src.models.clustered_se import cluster_dse, stimulus_clusters
    from src.models.pymc_inference import (
        clear_fit_cache,
        convergence_problems_of,
        fit_model,
        fit_models_to_cache,
    )

    root = run_root(args.run)
    fit_csv = root / ("experiment3" if scheme == "all" else "experiment2") / "model_loop/responses.csv"
    eval_csv = (
        root / "experiment3/model_loop/responses.csv"
        if scheme == "all"
        else root / "experiment3/data/responses.csv"
    )
    work_dir = args.work_dir or ANALYSIS_ROOT / ("fits/person_level" if args.run == 1 else f"fits/person_level_run{args.run}")
    models_dir = work_dir / scheme / "models"
    names = assemble_models_dir(models_dir, args.run)
    if args.only:
        unknown = sorted(set(args.only) - set(names))
        if unknown:
            raise ValueError(f"--only names unknown models: {unknown}")
        names = [n for n in names if n in args.only]
    suffix = ("_only" if args.only else "") + ("" if args.run == 1 else f"_run{args.run}")
    def fit_kwargs(name: str) -> dict:
        return dict(
            draws=args.draws, tune=args.tune, chains=args.chains, target_accept=args.target_accept,
            random_seed=args.seed_overrides.get(
                f"{name}@{fit_csv.parent.parent.name}", args.seed_overrides.get(name, args.random_seed)
            ),
        )
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Sample in small batches (each loaded fit is ~1.5 GB at 7,680 trials).
    for i in range(0, len(names), args.fit_batch):
        batch = names[i : i + args.fit_batch]
        print(f"[person-level] {scheme}: fitting {batch}", flush=True)
        by_seed: dict[int, list[str]] = {}
        for name in batch:
            by_seed.setdefault(fit_kwargs(name)["random_seed"], []).append(name)
        for group in by_seed.values():
            failures = fit_models_to_cache(group, models_dir, fit_csv, cache_dir=args.cache_dir, **fit_kwargs(group[0]))
            if failures:
                raise RuntimeError(f"{scheme}: fits failed: {failures}")
            clear_fit_cache()

    fit_rows = read_rows(fit_csv)
    eval_rows = read_rows(eval_csv)
    if scheme == "heldout":
        seen = {r["participant_id"] for r in fit_rows}
        if seen & {r["participant_id"] for r in eval_rows}:
            raise ValueError("held-out people overlap the fitting data")
    y = np.array([int(r["chose_left"]) for r in eval_rows])
    person = np.array([r["participant_id"] for r in eval_rows])
    people = sorted(set(person), key=int)
    if max(int(r["participant_id"]) for r in fit_rows + eval_rows) >= FIRST_FREE_SLOT:
        raise ValueError(f"participant ids reach the new-participant slots (from {FIRST_FREE_SLOT})")
    clusters = stimulus_clusters(fit_rows)

    records, loo_i, person_rows = [], {}, []
    for name in names:
        fitted = fit_model(name, models_dir, fit_csv, cache_dir=args.cache_dir, **fit_kwargs(name))
        diag = fitted.loo_diagnostics()
        loo_i[name] = np.asarray(diag.loo.loo_i)
        draws = new_participant_draws(fitted, eval_rows, people, args.max_draws, args.random_seed)
        scores = person_scores(draws, y, person, people)
        rmse, r2 = stimulus_fit(eval_rows, draws.mean(axis=0))
        records.append(
            {
                "model": name,
                **describe(name),
                "trial_loo_elpd": float(diag.elpd_loo) if scheme == "all" else None,
                "new_participant_score": float(scores.sum()),
                "n_people": len(people),
                "mc_samples_per_person": int(draws.shape[0]),
                "pop_rmse_stimulus": rmse,
                "pop_r2_stimulus": r2,
                "loo_unreliable": diag.unreliable,
                "convergence_problems": "; ".join(convergence_problems_of(fitted)),
                "random_seed": fit_kwargs(name)["random_seed"],
            }
        )
        person_rows += [{"model": name, "participant_id": p, "score": s} for p, s in zip(people, scores)]
        print(f"[person-level] {scheme}: {name} new-participant {scores.sum():.1f}", flush=True)
        del fitted, draws
        clear_fit_cache()

    table = pd.DataFrame(records)
    people_df = pd.DataFrame(person_rows)
    wide = people_df.pivot(index="participant_id", columns="model", values="score")
    best_new = table.loc[table.new_participant_score.idxmax(), "model"]
    diff = wide[best_new].values[:, None] - wide[table.model].values
    table["new_participant_diff"] = diff.sum(axis=0)
    table["new_participant_dse"] = np.sqrt(len(people) * diff.var(axis=0))
    if scheme == "all":
        best_loo = table.loc[table.trial_loo_elpd.idxmax(), "model"]
        table["trial_loo_diff"] = [float((loo_i[best_loo] - loo_i[m]).sum()) for m in table.model]
        table["trial_loo_dse_clustered"] = [
            0.0 if m == best_loo else cluster_dse(loo_i[best_loo], loo_i[m], clusters) for m in table.model
        ]
    table = table.sort_values("new_participant_score", ascending=False)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / f"person_level_{scheme}{suffix}.csv", index=False)
    people_df.to_csv(args.out_dir / f"person_level_{scheme}_people{suffix}.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(table.drop(columns=["convergence_problems"]).to_string(index=False, float_format="%.3f"))


def main(args: Args) -> None:
    for scheme in (("all", "heldout") if args.scheme == "both" else (args.scheme,)):
        run_scheme(scheme, args)


if __name__ == "__main__":
    main(tyro.cli(Args))
