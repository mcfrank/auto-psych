"""Reference games: each recovery cell's best model's distance to the ground
truth at every step, for a recovery figure like the other two domains'.

Sherlock run 2's recovery cells recorded the distance only for the exported
model (``recovery/recovery.json``). Here every step's best model
(``history.json``) and the ground truth are refitted on the cell's simulated
training data with the recovery's own fit rule and settings
(``src.rsa.recovery``: ``loop_fit``, 4 × (1,000 + 1,000), seed 0), and the
distance is the RMSE between their posterior-mean choice-class probabilities
over the novelty pool, as there.

The cells' data are Mike Frank's copy of run 2 (``--data-root``). They include
sources without a redistribution licence, so every trial-level file stays in
``--work-dir`` on scratch (the script refuses a work directory inside the
checkout); only the per-step distances are written to ``--out``.

    --mode setup     copy each cell's training data and model files into the
                     work directory, checking them against run 2's hashes, and
                     list the fit tasks
    --mode fit       one fit task (``--task``, a Slurm array index): fit the
                     model and save its pool predictions
    --mode assemble  the distances, after checking that the last step's
                     reproduces the cell's recorded end point
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import tyro
from pyprojroot import here

CELLS = ["recovery_literal_rep1", "recovery_literal_rep2", "recovery_salience_rep1", "recovery_salience_rep2"]
GROUND_TRUTHS = {"recovery_literal": "literal_listener", "recovery_salience": "rsa_l1_salience"}
SEED_MODELS = here() / "src" / "pipelines" / "outer_loop" / "projects" / "rsa_reference" / "seed_models"
RUN2_RECORDS = here() / "data" / "rsa" / "sherlock_run2"
# The end point recomputed here must match the recorded one this closely.
END_POINT_TOLERANCE = 1e-4


@dataclass
class Args:
    mode: Literal["setup", "fit", "assemble"]
    work_dir: Path
    """Trial-level data, fits and predictions; must be outside the checkout."""
    data_root: Path = Path("/scratch/groups/mcfrank/rsa_data/rsa_run2")
    """Run 2's cells and simulated data (``<cell>/``, ``data/``)."""
    task: Optional[int] = None
    out: Path = here() / "data" / "paper" / "rsa_recovery_trajectory.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_outside_checkout(work_dir: Path) -> None:
    if work_dir.resolve().is_relative_to(here().resolve()):
        raise ValueError(f"the work directory {work_dir} is inside the checkout; trial-level data stay out of git")


def steps(cell_dir: Path) -> List[Dict[str, Any]]:
    """Each step's number and best model, from the cell's history."""
    return [{"step": h["step"], "best_model": h["best_model"]}
            for h in json.loads((cell_dir / "history.json").read_text())]


def model_file(cell_dir: Path, name: str) -> Path:
    for path in [cell_dir / "models" / f"{name}.py", cell_dir / "models" / "pruned" / f"{name}.py"]:
        if path.exists():
            return path
    raise FileNotFoundError(f"no file for {name} under {cell_dir / 'models'}")


def setup(args: Args) -> None:
    tasks = []
    for cell in CELLS:
        source, work = args.data_root / cell, args.work_dir / cell
        cell_record = json.loads((RUN2_RECORDS / cell / "cell.json").read_text())
        condition = cell_record["condition"]

        # The loop's training data must be the simulated train.csv run 2 hashed.
        recorded = dict(line.split()[::-1] for line in (RUN2_RECORDS / cell / "data.sha256").read_text().splitlines())
        if sha256(source / "responses.csv") != recorded["train.csv"]:
            raise ValueError(f"{source / 'responses.csv'} is not the training data run 2 recorded for {cell}")
        (work / "models").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / "responses.csv", work / "responses.csv")

        gt = GROUND_TRUTHS[condition]
        provenance = json.loads((args.data_root / "data" / condition / "provenance.json").read_text())
        if sha256(SEED_MODELS / f"{gt}.py") != provenance["ground_truth_sha256"]:
            raise ValueError(f"{SEED_MODELS / gt}.py is not the ground truth {condition} was simulated from")
        best_models = sorted({s["best_model"] for s in steps(source)})
        if gt in best_models:
            raise ValueError(f"{cell}: the ground truth {gt} was one of the loop's models")
        shutil.copyfile(SEED_MODELS / f"{gt}.py", work / "models" / f"{gt}.py")
        for name in best_models:
            shutil.copyfile(model_file(source, name), work / "models" / f"{name}.py")
        tasks += [{"cell": cell, "model": name} for name in [gt, *best_models]]
        shutil.copyfile(source / "history.json", work / "history.json")

    (args.work_dir / "tasks.json").write_text(json.dumps(tasks, indent=1))
    print(f"[setup] {len(tasks)} fit tasks in {args.work_dir / 'tasks.json'}", flush=True)


def fit(args: Args) -> None:
    from src.rsa.fit import FitSettings
    from src.rsa.loop.fitting import loop_fit
    from src.rsa.loop.novelty import novelty_pool, posterior_mean_class_probs
    from src.rsa.model_file import RSAModel

    task = json.loads((args.work_dir / "tasks.json").read_text())[args.task]
    work = args.work_dir / task["cell"]
    out = work / "predictions" / f"{task['model']}.npy"
    if out.exists():
        print(f"[fit] {task['cell']} {task['model']}: already done", flush=True)
        return
    path = work / "models" / f"{task['model']}.py"
    model = RSAModel(path, name=task["model"])
    print(f"[fit] {task['cell']} {task['model']}", flush=True)
    fitted = loop_fit(path, task["model"], work / "responses.csv", FitSettings(), work / "fit_cache")
    predictions = posterior_mean_class_probs(model, fitted, novelty_pool())
    out.parent.mkdir(exist_ok=True)
    np.save(out.with_suffix(".partial.npy"), predictions)
    out.with_suffix(".partial.npy").rename(out)


def assemble(args: Args) -> None:
    from src.rsa.loop.novelty import closest

    record: Dict[str, Any] = {
        "distance": "RMSE between posterior-mean choice-class probabilities over the novelty pool",
        "fit_settings": "src.rsa.fit.FitSettings() via src.rsa.loop.fitting.loop_fit",
        "cells": {},
    }
    for cell in CELLS:
        work = args.work_dir / cell
        ground_truth = GROUND_TRUTHS[json.loads((RUN2_RECORDS / cell / "cell.json").read_text())["condition"]]
        gt = np.load(work / "predictions" / f"{ground_truth}.npy")
        trajectory = [
            {**s, "rmse_to_gt": closest(np.load(work / "predictions" / f"{s['best_model']}.npy"), {"gt": gt})[1]}
            for s in steps(work)
        ]
        recovery = json.loads((RUN2_RECORDS / cell / "recovery" / "recovery.json").read_text())
        end = trajectory[-1]
        if end["best_model"] != recovery["best_model"]:
            raise ValueError(f"{cell}: the last step's best is {end['best_model']}, "
                             f"the exported model {recovery['best_model']}")
        if abs(end["rmse_to_gt"] - recovery["best_pool_rmse_to_gt"]) > END_POINT_TOLERANCE:
            raise ValueError(f"{cell}: the refitted end point {end['rmse_to_gt']:.6f} does not reproduce "
                             f"the recorded {recovery['best_pool_rmse_to_gt']:.6f}")
        record["cells"][cell] = {
            "ground_truth": ground_truth,
            "recorded_end_point": recovery["best_pool_rmse_to_gt"],
            "best_seed": recovery["best_seed"],
            "best_seed_rmse_to_gt": recovery["best_seed_rmse_to_gt"],
            "rmse_threshold": recovery["rmse_threshold"],
            "trajectory": trajectory,
        }
        print(f"[assemble] {cell}: " + ", ".join(f"{t['rmse_to_gt']:.4f}" for t in trajectory), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    print(f"wrote {args.out}", flush=True)


def main(args: Args) -> None:
    check_outside_checkout(args.work_dir)
    if args.mode == "setup":
        setup(args)
    elif args.mode == "fit":
        if args.task is None:
            raise ValueError("--mode fit needs --task")
        fit(args)
    else:
        assemble(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
