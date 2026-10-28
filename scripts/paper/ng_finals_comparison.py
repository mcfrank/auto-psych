"""Which number-game dataset run's final fits best: the three finals on the three runs' data pooled.

    python scripts/paper/ng_finals_comparison.py --out <dir> --mode data
    python scripts/paper/ng_finals_comparison.py --out <dir> --mode fit --task <0..14>   (a Slurm array)
    python scripts/paper/ng_finals_comparison.py --out <dir> --mode report

The three runs (data/results/number_game_dataset-oct26/) ended on three different finals, each
scored only on its own run's items. This compares them on one data set and picks the one to
ablate (user decision, 2026-10-28):

- **Data** (``--mode data``): each run's experiment-3 cumulative responses, pooled. The runs drew
  without replacement from one response bank, so a row in two runs' data (same example set,
  target, session position and rating) is one response drawn twice: the pool takes each row as
  often as the run that has it most (a multiset union), not the sum. Written to
  ``<out>/pooled.csv`` with the runs' own columns.
- **CV**: 5 folds of whole example sets (`src.model_comparison.cv`, ``number_set``, seed 0), the
  number-game ablation's (`src.number_game.ablation.eliminate`).
- **Fits** (``--mode fit``): one (final, fold) per task, at the number-game ablation's sampler
  settings (4 chains, 4,000 tuning steps, 1,000 draws, target_accept 0.95, ``nuts_no_jitter``,
  seed 20261018); held-out log densities and the convergence gate on every scalar parameter.
- **Report** (``--mode report``): each final's ELPD-CV, every pairwise difference with its SE
  clustered by example set, each fit's convergence; ``report.json`` and ``report.md``.
"""

from __future__ import annotations

import collections
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tyro
from pyprojroot import here

COLLECTED = here() / "data/results/number_game_dataset-oct26/collected"
FINALS = {1: "even_rule_digit_size_principle_counting", 2: "relational_analogy_transfer",
          3: "outlier_tolerant_rule_concepts"}
FOLDS = 5
SAMPLER = dict(draws=1000, tune=4000, chains=4, target_accept=0.95, method="nuts_no_jitter", random_seed=20261018)


@dataclass
class Args:
    out: Path
    mode: Literal["data", "fit", "report"]
    task: int = -1
    """fit: (final, fold) task number, final-major: task = 5 * (run - 1) + fold."""


def _responses(run: int) -> Path:
    return COLLECTED / f"run{run}/number_game/experiment3/model_loop/responses.csv"


def _selection():
    from src.model_comparison.cv import Selection
    from src.number_game.ablation.eliminate import CV_GROUP

    return Selection(criterion="grouped_cv", cv_group=CV_GROUP, cv_folds=FOLDS, cv_seed=0)


def make_data(out: Path) -> Path:
    """The multiset union of the three runs' responses, in run order."""
    header, per_run = None, {}
    for run in FINALS:
        with _responses(run).open(newline="") as f:
            reader = csv.reader(f)
            h = next(reader)
            if header not in (None, h):
                raise ValueError(f"run {run}'s responses have columns {h}, not {header}")
            header, per_run[run] = h, [tuple(r) for r in reader]
    taken: collections.Counter = collections.Counter()
    pooled = []
    for run, rows in per_run.items():
        need = collections.Counter(rows)
        for row in rows:
            if taken[row] < need[row]:  # this run has more copies of the row than the pool so far
                taken[row] += 1
                pooled.append(row)
    path = out / "pooled.csv"
    out.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(".pooled.csv.tmp")
    with tmp.open("w", newline="") as f:
        csv.writer(f).writerows([header, *pooled])
    if path.exists() and path.read_bytes() != tmp.read_bytes():
        tmp.unlink()
        raise ValueError(f"{path} exists with other contents; use a fresh --out")
    tmp.replace(path)
    (out / "models").mkdir(exist_ok=True)
    for run, name in FINALS.items():
        src = COLLECTED / f"run{run}/number_game/experiment3/cognitive_models/{name}.py"
        (out / "models" / f"{name}.py").write_bytes(src.read_bytes())
    from src.model_comparison.cv import fold_training_files

    fold_training_files(path, _selection())  # written once here, before the fit tasks read them
    print(f"[data] {path}: {len(pooled)} responses ({sum(map(len, per_run.values()))} drawn by the runs)", flush=True)
    return path


def fit(out: Path, task: int) -> dict:
    from src.model_comparison.cv import fold_training_files
    from src.models.data_binding import make_stim_data
    from src.models.pymc_inference import convergence_problems, fit_model

    run, fold = task // FOLDS + 1, task % FOLDS
    name = FINALS[run]
    rows, folds, paths = fold_training_files(out / "pooled.csv", _selection())
    fitted = fit_model(name, out / "models", paths[fold], cores=SAMPLER["chains"], **SAMPLER)
    scalars = [rv.name for rv in fitted.model.free_RVs if fitted.idata.posterior[rv.name].ndim == 2]
    lpd = fitted.heldout_lpd(make_stim_data(fitted.model, [rows[i] for i in np.flatnonzero(folds == fold)]))
    (out / "cv").mkdir(exist_ok=True)
    np.save(out / "cv" / f"{name}.fold{fold}.npy", lpd)
    record = dict(model=name, run=run, fold=fold, elpd=float(lpd.sum()),
                  problems=convergence_problems(fitted.idata, scalars))
    (out / "cv" / f"{name}.fold{fold}.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"[fit] {name} fold {fold}: held-out lpd {lpd.sum():.1f}, problems {record['problems']}", flush=True)
    return record


def report(out: Path) -> dict:
    from src.model_comparison.cv import fold_training_files
    from src.models.clustered_se import cluster_dse
    from src.number_game.ablation.eliminate import example_set_clusters

    rows, folds, _ = fold_training_files(out / "pooled.csv", _selection())
    units = example_set_clusters(rows)
    pointwise, fits = {}, {}
    for name in FINALS.values():
        p = np.full(len(rows), np.nan)
        for f in range(FOLDS):
            p[folds == f] = np.load(out / "cv" / f"{name}.fold{f}.npy")
            fits[(name, f)] = json.loads((out / "cv" / f"{name}.fold{f}.json").read_text())
        pointwise[name] = p
    ranked = sorted(pointwise, key=lambda n: -pointwise[n].sum())
    best = ranked[0]
    doc = dict(n_responses=len(rows), folds=FOLDS, sampler=SAMPLER,
               models={n: dict(run=r, elpd_cv=float(pointwise[n].sum()),
                               converged=all(not fits[(n, f)]["problems"] for f in range(FOLDS)),
                               problems={f: fits[(n, f)]["problems"] for f in range(FOLDS) if fits[(n, f)]["problems"]})
                       for r, n in FINALS.items()},
               best=best,
               vs_best={n: dict(diff=float(pointwise[best].sum() - pointwise[n].sum()),
                                se=float(cluster_dse(pointwise[best], pointwise[n], units)))
                        for n in ranked[1:]})
    (out / "report.json").write_text(json.dumps(doc, indent=1) + "\n")
    lines = [f"# Number-game finals on the three runs' data pooled ({len(rows)} responses, {FOLDS}-fold CV by example set)",
             "", "| run | final | ELPD-CV | behind the best (SE) | converged |", "|---|---|---|---|---|"]
    for n in ranked:
        d = doc["vs_best"].get(n)
        lines.append(f"| {doc['models'][n]['run']} | `{n}` | {doc['models'][n]['elpd_cv']:.1f} | "
                     + (f"{d['diff']:.1f} ({d['se']:.1f})" if d else "—") + f" | {doc['models'][n]['converged']} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return doc


def main(args: Args):
    if args.mode == "data":
        return make_data(args.out)
    if args.mode == "fit":
        return fit(args.out, args.task)
    return report(args.out)


if __name__ == "__main__":
    main(tyro.cli(Args))
