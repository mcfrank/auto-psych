"""Held-out conditions, panel by panel: people against a cell's models.

    uv run python -m src.rsa.heldout_report --cell data/rsa/sherlock_run2/real_rep3 \
        --train <the cell's train.csv> --test <its test.csv> --work <scratch dir>

    # on Sherlock, from the loop's own fits (scripts/rsa/slurm/heldout_reports.sbatch)
    uv run python -m src.rsa.heldout_report --cell data/rsa/sherlock_run2/real_rep3 \
        --loop-dir <the cell's agent tree>/repo/_runs/loop --test <its test.csv> --work <scratch dir>

The condition-by-condition page of `src.rsa.report` (one panel per display
cell: people's choice proportions against model predictions) drawn on the
test conditions a brought-back cell never saw, for the models that matter to
the paper's claim 1: the best seed, the exported model and the models that
did best on the held-out table (``heldout/heldout.json``), plus any named
with ``--models``. Each is fitted to the cell's training trials (at the
held-out evaluation's settings, `src.rsa.loop.fitting.loop_fit`) and the
standing is the held-out lpd, its difference from the best of these models
and the SE of that difference clustered by held-out unit.

``--test`` (and ``--train``) must hash as the cell recorded them
(``data.sha256``). With ``--loop-dir`` the training trials are the loop's
own ``responses.csv`` and the fits come from its ``.fit_cache``: the held-out
evaluation's fits, so nothing is refitted. Otherwise fits go to
``<work>/cache`` (the same fits: same trials, settings and seed).

Also writes ``<cell>/heldout/unit_lpd.csv``: each shown model's lpd summed
over every unit (`src.rsa.split.unit_keys`) of the test set (held out) and
of the training set (in sample, the posterior predictive of the fitted
trials) and, with ``--loop-dir``, out of fold (the loop's grouped CV, from
its ``.cv`` folds and cached fold fits). It locates where models gain: per
source, grouped CV and the held-out test can disagree when a source's
training units differ in kind from its test units. The page and table carry only aggregates (choice counts
per display, lpd sums per unit), never trial rows.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tyro

from src.models.clustered_se import cluster_dse
from src.rsa.compare_seeds import cell_table
from src.rsa.dataset import load_forced_choice
from src.rsa.evaluate_heldout import SEED_DIR, heldout_lpd
from src.rsa.fit import FitSettings, posterior_mean_probs
from src.rsa.loop.cv import cv_pointwise, make_folds
from src.rsa.loop.fitting import loop_fit
from src.rsa.model_file import RSAModel
from src.rsa.report import build_bundle, render
from src.rsa.split import unit_keys

METRIC = dict(
    lede=("How well each model predicts which referent people pick in the held-out conditions: conditions of "
          "the same papers that the loop never saw. Every model was fitted to the loop's training trials only. "
          "The panels show people's choice proportions against the models' predictions for each held-out "
          "game, word and condition."),
    standing_note=("Held-out log predictive density behind the best of these models, in nats (whiskers ±2 SE "
                   "of the difference, clustered by held-out condition). Lower is better. r and RMSE compare "
                   "predicted and observed choice proportions over held-out display cells."),
    diff_label="Δlpd",
)


@dataclass
class Args:
    cell: Path
    """A brought-back cell (export.json, heldout/heldout.json, models/, data.sha256)."""
    test: Path
    """The cell's held-out trials (test.csv)."""
    work: Path
    """Scratch directory (the fit cache without --loop-dir)."""
    train: Optional[Path] = None
    """The cell's training trials (train.csv); or give --loop-dir."""
    loop_dir: Optional[Path] = None
    """The cell's loop directory (responses.csv, .fit_cache), on the machine that ran it."""
    top: int = 3
    """Also show this many of the best models on the held-out table."""
    models: List[str] = field(default_factory=list)
    """More models, as labelled in heldout.json (``seed:x``, ``pruned:x`` or a live name)."""
    out_html: Optional[Path] = None
    """Default: <cell>/heldout/report.html."""
    min_cell_n: int = 10


def _check_hash(cell: Path, path: Path) -> None:
    sha = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    record = cell / "data.sha256"
    if not record.exists():
        raise FileNotFoundError(f"{record} is missing: cannot check that {path} is this cell's data")
    if sha not in record.read_text():
        raise ValueError(f"{path} (sha256 {sha[:12]}) is not data {record} records")


def model_path(cell: Path, label: str, seeds: Optional[Path] = None) -> Path:
    kind, _, name = label.rpartition(":")
    path = {"seed": seeds or SEED_DIR, "pruned": cell / "models" / "pruned", "": cell / "models"}[kind] / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"{label}: no model file at {path}")
    return path


def chosen_models(cell: Path, heldout: dict, top: int, extra: List[str]) -> List[str]:
    """Best seed, exported model, the ``top`` best held-out models, then ``extra``."""
    exported = json.loads((cell / "export.json").read_text())["best_model"]
    labels = [r["model"] for r in sorted(heldout["table"], key=lambda r: -r["lpd"])]
    out = [heldout["best_seed"], exported]
    names = {l.rpartition(":")[2] for l in out}
    out += [l for l in labels if not l.startswith("seed:") and l.rpartition(":")[2] not in names][:top]
    for label in extra:
        if label not in labels:
            raise ValueError(f"{label} is not in {cell}/heldout/heldout.json")
        if label not in out:
            out.append(label)
    return out


def main(args: Args) -> Path:
    cell, work = Path(args.cell), Path(args.work)
    if (args.train is None) == (args.loop_dir is None):
        raise ValueError("give exactly one of --train and --loop-dir")
    _check_hash(cell, args.test)
    if args.loop_dir is not None:
        train, cache = Path(args.loop_dir) / "responses.csv", Path(args.loop_dir) / ".fit_cache"
        if not cache.is_dir():
            raise FileNotFoundError(f"{cache} is missing: is {args.loop_dir} the cell's loop directory?")
    else:
        _check_hash(cell, args.train)
        train, cache = Path(args.train), work / "cache"
    heldout = json.loads((cell / "heldout" / "heldout.json").read_text())
    recorded_seeds = Path(heldout["seed_models"])
    seeds = recorded_seeds if recorded_seeds.is_dir() else None  # the evaluation's own seed files, where they are
    settings = FitSettings(**heldout["settings"])
    labels = chosen_models(cell, heldout, args.top, args.models)
    exported = json.loads((cell / "export.json").read_text())["best_model"]

    test = load_forced_choice(args.test)
    if len(test.contexts) != heldout["n_test_trials"]:
        raise ValueError(f"{args.test} has {len(test.contexts)} held-out trials; {cell} recorded {heldout['n_test_trials']}")
    units = pd.factorize(unit_keys(test.frame))[0]
    recorded = {r["model"]: r["lpd"] for r in heldout["table"]}

    train_trials = load_forced_choice(train)
    folds = None
    if args.loop_dir is not None and (Path(args.loop_dir) / ".cv" / "folds.json").exists():
        record = json.loads((Path(args.loop_dir) / ".cv" / "folds.json").read_text())
        folds = make_folds(train, Path(args.loop_dir) / ".cv", record["k"], seed=record["seed"])
    units_of = {"test": unit_keys(test.frame).to_numpy(), "train": unit_keys(train_trials.frame).to_numpy()}
    lpd: Dict[str, np.ndarray] = {}
    preds, params, problems, unit_rows = {}, [], {}, []
    display = {}
    cached = set(Path(cache).glob("*.nc"))
    for label in labels:
        name = label.rpartition(":")[2]
        shown = f"{name} (seed)" if label.startswith("seed:") else (
            f"{name} (exported)" if name == exported else (f"{name} (pruned)" if label.startswith("pruned:") else name))
        display[label] = shown
        path = model_path(cell, label, seeds)
        model = RSAModel(path, name=name)
        fitted = loop_fit(path, name, train, settings, cache)
        lpd[shown] = heldout_lpd(model, fitted, test.contexts, test.choices)
        in_sample = heldout_lpd(model, fitted, train_trials.contexts, train_trials.choices)
        splits = [("test", test, lpd[shown]), ("train", train_trials, in_sample)]
        if folds is not None:
            splits.append(("cv", train_trials, cv_pointwise(path, name, train, folds, settings, cache).pointwise))
        for split, trials, values in splits:
            frame = trials.frame.assign(unit=units_of["test" if split == "test" else "train"], lpd=values)
            source = frame["source"] if "source" in frame else "pragmods"
            g = frame.assign(source=source).groupby(["source", "experiment", "unit"], sort=True)
            unit_rows.append(g["lpd"].agg(n="size", lpd="sum").reset_index().assign(split=split, model=shown))
        preds[shown] = posterior_mean_probs(model, fitted, test.contexts)
        problems[shown] = "; ".join(fitted.convergence_problems)
        for p in fitted.param_names:
            d = np.asarray(fitted.idata.posterior[p]).ravel()
            params.append(dict(model=shown, param=p, mean=d.mean(), lo=np.quantile(d, 0.03), hi=np.quantile(d, 0.97)))
        new = set(Path(cache).glob("*.nc")) - cached
        cached |= new
        fitted_now = f"; fitted {len(new)} (not in {cache})" if new else ""
        print(f"  {shown}: held-out lpd {lpd[shown].sum():.1f} (recorded {recorded[label]:.1f}){fitted_now}", flush=True)

    best = max(lpd, key=lambda n: lpd[n].sum())
    order = sorted(lpd, key=lambda n: -lpd[n].sum())
    comp = pd.DataFrame([
        dict(name=n, rank=i, elpd_loo=lpd[n].sum(), se=float(np.sqrt(len(lpd[n])) * lpd[n].std()),
             elpd_diff=lpd[best].sum() - lpd[n].sum(),
             dse=0.0 if n == best else float(cluster_dse(lpd[best], lpd[n], units)),
             p_loo=float("nan"), loo_reliable=True, convergence_problems=problems[n] or float("nan"))
        for i, n in enumerate(order)
    ]).set_index("name")

    out_dir = work / "heldout_report"
    out_dir.mkdir(parents=True, exist_ok=True)
    comp.to_csv(out_dir / "comparison.csv")
    pd.DataFrame(params).to_csv(out_dir / "params.csv", index=False)
    cell_table(test, preds).to_csv(out_dir / "cells.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(dict(n_trials=len(test.contexts), settings=vars(settings))))

    (cell / "heldout").mkdir(exist_ok=True)
    pd.concat(unit_rows)[["model", "split", "source", "experiment", "unit", "n", "lpd"]].to_csv(
        cell / "heldout" / "unit_lpd.csv", index=False)

    rationale_dir = out_dir / "manifest"
    rationale_dir.mkdir(exist_ok=True)
    (rationale_dir / "models_manifest.yaml").write_text(json.dumps({"models": [
        dict(name=display[l], file=model_path(cell, l, seeds).name, rationale=_docstring(model_path(cell, l, seeds)))
        for l in labels
    ]}))
    bundle = build_bundle(out_dir, rationale_dir, title=f"{cell.name}: held-out conditions",
                          dataset_label=f"{cell.name} · {heldout['n_test_units']} held-out conditions",
                          min_cell_n=args.min_cell_n)
    bundle["metric"] = METRIC
    # The panels open on the claim's comparison: the seed, the exported model, the best held out.
    bundle["default_models"] = list(dict.fromkeys(
        [display[heldout["best_seed"]], display[exported], order[0]]))
    for m in bundle["models"]:
        m["loo_reliable"] = None
        m["p_loo"] = None  # no PSIS-LOO on held-out data
        m["heldout_vs_seed"] = float(lpd[m["name"]].sum() - lpd[display[heldout["best_seed"]]].sum())
    out = Path(args.out_html or cell / "heldout" / "report.html")
    out.write_text(render(bundle), encoding="utf-8")
    out.with_suffix(".bundle.json").write_text(json.dumps(bundle, indent=1, allow_nan=False))
    print(f"wrote {out} ({len(bundle['panels'])} panels, {len(bundle['models'])} models)")
    return out


def _docstring(path: Path) -> str:
    import ast

    doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
    return " ".join(doc.split())


if __name__ == "__main__":
    main(tyro.cli(Args))
