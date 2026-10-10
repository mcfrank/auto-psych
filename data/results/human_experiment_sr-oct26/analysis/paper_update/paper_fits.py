"""MCMC fits for the paper's pooled tables: Table 1 (all runs) and Table S1 (each
winner on the two runs that did not produce it).

``prepare`` writes, per dataset, ``analysis/fits/<dataset>/responses.csv`` (the
runs' experiment-3 cumulative responses, concatenated, each run's participant
ids offset past the previous runs' so no two people share an id) and
``models/`` (the four seeds and the dataset's winners, with a manifest), and
``tasks.tsv`` (one fit per line: dataset, model).

``fit --dataset D --model M`` fits one model at the live run's sampler settings
(draws 3000, tune 2000, 4 chains, target_accept 0.8) at the first seed of SEEDS
(SEED_ORDER for motif_stack) whose fit passes the convergence gate within the
attempt's time limit (TIME_LIMIT_SEC), into the analysis's shared mcmc_cache,
and records it in ``analysis/fits/<dataset>/fit_seeds/<model>.json``. A model
that converges at no seed raises.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Union

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid: set(range(os.cpu_count()))

SERIES = here() / "data/results/human_experiment_sr-oct26"
ANALYSIS_ROOT = SERIES / "analysis"
SEED_MODELS = ("falk_konold_dp", "finite_experience_occurrence", "local_representativeness", "motif_stack")
WINNERS = {
    1: "personal_pattern_detection_gain",
    2: "edge_streak_primacy_recency_pattern_rule",
    3: "mild_symmetric_trick_coin_streak_weary",
}
# dataset -> (runs whose data it pools, runs whose winners it fits)
DATASETS = {
    "pooled_runs123": ((1, 2, 3), (1, 2, 3)),
    "without_run1": ((2, 3), (1,)),
    "without_run2": ((1, 3), (2,)),
    "without_run3": ((1, 2), (3,)),
}
SAMPLER = dict(draws=3000, tune=2000, chains=4, target_accept=0.8)
SEEDS = (20261003, 6, 2, 1, 5, 3, 4, 7, 8, 9, 10, 11, 12)
# motif_stack failed at 20261003 (stuck chains) and converged at 6 in all three
# single-run fits, and a stuck chain takes hours at 15-23k trials: start at 6.
SEED_ORDER = {"motif_stack": (6, 20261003, 2, 1, 5, 3, 4, 7, 8, 9, 10, 11, 12)}
# Likewise run 2's winner: 20261003 failed on run 2 alone (3000 divergent, R-hat
# 1.53) and on the pooled data the same way; 6 converged on run 2.
SEED_ORDER["edge_streak_primacy_recency_pattern_rule"] = SEED_ORDER["motif_stack"]
# Per attempt; a timed-out attempt moves on to the next seed like an unconverged one.
TIME_LIMIT_SEC = {"seed": 45 * 60, "winner": 4 * 60 * 60}


def run_root(run: int):
    return SERIES / f"collected/run{run}/subjective_randomness"


def dataset_dir(dataset: str):
    return ANALYSIS_ROOT / "fits" / dataset


@dataclass
class Prepare:
    """Write every dataset's responses, models and the task list."""


@dataclass
class Fit:
    """Fit one model on one dataset at the first converging sampler seed."""

    dataset: str
    model: str


def prepare() -> None:
    tasks = []
    for dataset, (data_runs, winner_runs) in DATASETS.items():
        out = dataset_dir(dataset)
        out.mkdir(parents=True, exist_ok=True)
        rows, offset = [], 0
        for run in data_runs:
            with (run_root(run) / "experiment3/model_loop/responses.csv").open(encoding="utf-8", newline="") as f:
                run_rows = list(csv.DictReader(f))
            ids = sorted({int(r["participant_id"]) for r in run_rows})
            if ids != list(range(len(ids))):
                raise ValueError(f"run {run}: participant ids are not 0..{len(ids) - 1}")
            for r in run_rows:
                rows.append({**r, "participant_id": str(int(r["participant_id"]) + offset)})
            offset += len(ids)
        if offset > 400:
            raise ValueError(f"{dataset}: {offset} participants exceed the winners' 400 slots")
        with (out / "responses.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        models_dir = out / "models"
        if models_dir.exists():
            shutil.rmtree(models_dir)
        models_dir.mkdir()
        sources = [(s, run_root(1) / "experiment1/cognitive_models_input" / f"{s}.py") for s in SEED_MODELS]
        sources += [(WINNERS[r], run_root(r) / f"experiment3/model_loop/models/{WINNERS[r]}.py") for r in winner_runs]
        for name, path in sources:
            shutil.copyfile(path, models_dir / f"{name}.py")
        (models_dir / "models_manifest.yaml").write_text(
            "models:\n" + "".join(f"  - name: {n}\n" for n, _ in sources), encoding="utf-8"
        )
        print(f"[prepare] {dataset}: {len(rows)} trials, {offset} people, models {[n for n, _ in sources]}")
        tasks += [(dataset, n) for n, _ in sources]
    # The slow person-level winners first, so they overlap.
    tasks.sort(key=lambda t: (t[1] in SEED_MODELS, t[0] != "pooled_runs123"))
    (ANALYSIS_ROOT / "fits/paper_fit_tasks.tsv").write_text("".join(f"{d}\t{m}\n" for d, m in tasks), encoding="utf-8")


def fit(args: Fit) -> None:
    from src.models.pymc_inference import (
        FitTimeLimitExceeded,
        clear_fit_cache,
        convergence_problems_of,
        fit_model,
    )

    out = dataset_dir(args.dataset)
    record_path = out / "fit_seeds" / f"{args.model}.json"
    record_path.parent.mkdir(exist_ok=True)
    limit = TIME_LIMIT_SEC["seed" if args.model in SEED_MODELS else "winner"]
    tried = []
    for seed in SEED_ORDER.get(args.model, SEEDS):
        try:
            fitted = fit_model(
                args.model, out / "models", out / "responses.csv",
                cache_dir=ANALYSIS_ROOT / "mcmc_cache", random_seed=seed,
                time_limit_sec=limit, **SAMPLER,
            )
        except FitTimeLimitExceeded:
            print(f"[fit] {args.dataset} {args.model} seed {seed}: timed out at {limit} s", flush=True)
            tried.append({"seed": seed, "problems": [f"timed out at {limit} s"]})
            continue
        problems = convergence_problems_of(fitted)
        print(f"[fit] {args.dataset} {args.model} seed {seed}: {problems or 'converged'}", flush=True)
        tried.append({"seed": seed, "problems": problems})
        del fitted
        clear_fit_cache()
        if not problems:
            record_path.write_text(json.dumps({"seed": seed, "tried": tried}, indent=2), encoding="utf-8")
            return
    raise RuntimeError(f"{args.dataset} {args.model} converged at no seed of {SEED_ORDER.get(args.model, SEEDS)}")


if __name__ == "__main__":
    cmd: Union[Prepare, Fit] = tyro.cli(Union[Prepare, Fit])
    if isinstance(cmd, Prepare):
        prepare()
    else:
        fit(cmd)
