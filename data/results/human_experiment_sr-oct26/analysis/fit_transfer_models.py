"""The fits the generalization analysis needs, one per (model, data run), each converged.

Model sets: for source run S, the seeds and each experiment's winner of S,
full and with every person-level parameter pooled
(``fits/transfer/run<S>_on_run<T>/models``). Every set is fitted on every
run's all-data responses: on its own run for within-run LOPO and as the source
of fixed-parameter transfer, on the other run for refit transfer (and the
fixed-transfer proposals). The seeds are the same files in both runs, so a
seed is fitted once per data run.

``assemble`` writes the model dirs, ``fits.tsv`` (one fit per line: model,
model-set source run, data run) and ``scores.tsv``, the scoring tasks of
``generalization_analysis.py`` (analysis, source, target, part, models): per
analysis, one task per person-level winner (the slow ones) and one for the
population-level models together. ``fit --task i`` fits line i at the first
sampler seed of ``seeds`` whose fit passes the convergence gate
(``convergence_problems_of``; ``fit_model`` itself refits a near miss at
target_accept 0.95) and records it in ``fit_seeds/<model>@run<T>.json``, which
``generalization_analysis.py`` reads. A fit that converges at no seed raises.

Usage:
    uv run python data/results/human_experiment_sr-oct26/analysis/fit_transfer_models.py assemble
    uv run python data/results/human_experiment_sr-oct26/analysis/fit_transfer_models.py fit --task 1
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import tyro
from pyprojroot import here

sys.path.insert(0, str(here()))

# The pipeline sizes its concurrent fits by the scheduler's affinity mask
# (``allocated_cpus``), which only Linux has; on a laptop, every CPU.
if not hasattr(os, "sched_getaffinity"):
    os.sched_getaffinity = lambda pid: set(range(os.cpu_count()))

ANALYSIS_ROOT = here() / "data/results/human_experiment_sr-oct26/analysis"
GENERALIZATION = ANALYSIS_ROOT / "results/generalization"
SEEDS = ("falk_konold_dp", "finite_experience_occurrence", "local_representativeness", "motif_stack")
RUNS = (1, 2)
# Per run: each experiment's winner -> the experiment whose model_loop/models/ holds it.
WINNERS = {
    1: {
        "individual_streak_aversion_lapse": 1,
        "most_lopsided_stretch_aversion": 2,
        "personal_pattern_detection_gain": 3,
    },
    2: {
        "graded_periodicity_personal_ideal": 1,
        "tally_span_switch_ideal_periodic_unit": 2,
        "edge_streak_primacy_recency_pattern_rule": 3,
    },
}


def run_root(run: int) -> Path:
    return here() / f"data/results/human_experiment_sr-oct26/collected/run{run}/subjective_randomness"


def responses(run: int) -> Path:
    return run_root(run) / "experiment3/model_loop/responses.csv"


def ablation_dir(run: int) -> Path:
    return ANALYSIS_ROOT / ("ablation_models" if run == 1 else f"ablation_models_run{run}")


def transfer_models_dir(source: int, target: int) -> Path:
    return ANALYSIS_ROOT / f"fits/transfer/run{source}_on_run{target}/models"


def transfer_model_names(source: int) -> list[str]:
    return [*SEEDS, *WINNERS[source], *(f"{w}__pooled" for w in WINNERS[source])]


def seed_record(model: str, data_run: int, records: Path = GENERALIZATION / "fit_seeds") -> Path:
    return records / f"{model}@run{data_run}.json"


def assemble_dir(source: int, target: int) -> None:
    work = transfer_models_dir(source, target)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    sources = [(s, run_root(source) / "experiment1/cognitive_models_input" / f"{s}.py") for s in SEEDS]
    sources += [(w, run_root(source) / f"experiment{e}/model_loop/models/{w}.py") for w, e in WINNERS[source].items()]
    sources += [(f"{w}__pooled", ablation_dir(source) / f"{w}__pooled.py") for w in WINNERS[source]]
    for name, path in sources:
        if not path.exists():
            raise FileNotFoundError(f"model source missing: {path}")
        shutil.copyfile(path, work / f"{name}.py")
    (work / "models_manifest.yaml").write_text(
        "models:\n" + "".join(f"  - name: {n}\n" for n, _ in sources), encoding="utf-8"
    )
    if [n for n, _ in sources] != transfer_model_names(source):
        raise AssertionError("model list differs from transfer_model_names")


# (analysis, source run, target run) of generalization_analysis.py.
ANALYSES = [("within", 1, 1), ("within", 2, 2), ("refit", 1, 2), ("refit", 2, 1), ("fixed", 1, 2), ("fixed", 2, 1)]


def score_tasks() -> list[tuple[str, int, int, str, str]]:
    tasks = []
    for analysis, source, target in ANALYSES:
        for winner in WINNERS[source]:
            tasks.append((analysis, source, target, winner, winner))
        population = [n for n in transfer_model_names(source) if n not in WINNERS[source]]
        tasks.append((analysis, source, target, "population", ",".join(population)))
    return tasks


@dataclass
class Assemble:
    """Write every model-set dir, the fit manifest and the scoring tasks."""

    manifest: Path = GENERALIZATION / "fits.tsv"
    scores: Path = GENERALIZATION / "scores.tsv"


@dataclass
class Fit:
    """Fit one manifest line at the first converging sampler seed."""

    task: int
    """1-based manifest line (the Slurm array index)."""
    manifest: Path = GENERALIZATION / "fits.tsv"
    records: Path = GENERALIZATION / "fit_seeds"
    cache_dir: Path = ANALYSIS_ROOT / "mcmc_cache"
    seeds: tuple[int, ...] = (20261003, 6, 2, 1, 5, 3, 4, 7, 8, 9, 10, 11, 12)
    """Tried in order; the analysis seed first, then those that unstuck other fits here."""
    # The fits' settings (person_level_analysis.py).
    draws: int = 3000
    tune: int = 2000
    chains: int = 4
    target_accept: float = 0.8


def assemble(args: Assemble) -> None:
    lines, seen = [], set()
    for source in RUNS:
        for target in RUNS:
            assemble_dir(source, target)
            for name in transfer_model_names(source):
                # A seed is the same file in every set: fitted once per data run,
                # from the data run's own set.
                set_run = target if name in SEEDS else source
                if (name, target) not in seen:
                    seen.add((name, target))
                    lines.append((name, set_run, target))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t", lineterminator="\n").writerows(lines)
    tasks = score_tasks()
    with args.scores.open("w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="\t", lineterminator="\n").writerows(tasks)
    print(f"[assemble] {len(lines)} fits -> {args.manifest}; {len(tasks)} scoring tasks -> {args.scores}")


def fit(args: Fit) -> None:
    from src.models.pymc_inference import clear_fit_cache, convergence_problems_of, fit_model

    with args.manifest.open(encoding="utf-8", newline="") as f:
        lines = list(csv.reader(f, delimiter="\t"))
    if not 1 <= args.task <= len(lines):
        raise IndexError(f"task {args.task}: the manifest has {len(lines)} lines")
    name, set_run, data_run = lines[args.task - 1][0], int(lines[args.task - 1][1]), int(lines[args.task - 1][2])
    models_dir = transfer_models_dir(set_run, data_run)
    tried = []
    for seed in args.seeds:
        fitted = fit_model(
            name, models_dir, responses(data_run), cache_dir=args.cache_dir, draws=args.draws, tune=args.tune,
            chains=args.chains, target_accept=args.target_accept, random_seed=seed,
        )
        problems = convergence_problems_of(fitted)
        print(f"[fit] {name} on run {data_run} seed {seed}: {problems or 'converged'}", flush=True)
        tried.append({"seed": seed, "problems": problems})
        if not problems:
            args.records.mkdir(parents=True, exist_ok=True)
            seed_record(name, data_run, args.records).write_text(
                json.dumps(
                    {
                        "model": name,
                        "data_run": data_run,
                        "models_dir": str(models_dir.relative_to(here())),
                        "seed": seed,
                        "elpd_loo": float(fitted.loo_diagnostics().elpd_loo),
                        "tried": tried,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return
        del fitted
        clear_fit_cache()
    raise RuntimeError(f"{name} on run {data_run}: no converged fit at seeds {args.seeds}")


if __name__ == "__main__":
    command = tyro.extras.subcommand_cli_from_dict({"assemble": Assemble, "fit": Fit})
    assemble(command) if isinstance(command, Assemble) else fit(command)
