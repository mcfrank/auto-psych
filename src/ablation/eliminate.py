"""Backward elimination over a switchable final's mechanisms, by grouped CV.

The rule (the RSA consolidation's, docs/auto_rsa/CONSOLIDATION.md §2, on one
final at a time; docs/sr_ablation.md, docs/ng_ablation.md):

1. Start from the final with every mechanism on.
2. Each round, score the current model and every model with one mechanism
   removed by ELPD from grouped CV (`src.model_comparison.cv`: whole units of
   the ablation's ``cv_group`` in one fold; each held-out trial scored by its
   log posterior predictive density). Only removals that leave a valid model
   are candidates (`Variants.removals`).
3. Of the converged candidates, take the one with the highest ELPD-CV. Its
   cost is the current model's ELPD-CV minus its own; its SE is clustered by
   the ablation's ``units``. If the cost is at most 0 or below
   ``threshold_dse`` (2) SEs, that mechanism goes, and the next round starts
   from the smaller model.
4. Stop when every removal costs at least ``threshold_dse`` SEs (or none
   converges, or nothing is left to remove): the current model is the lean model.

The starting model must converge (on every fold); the elimination raises
otherwise (the RSA run costed its first drop against an unconverged start).
Converged means the repository's gate on every population-level parameter
(``gate="population"``; person-level vectors are reported, not gated) or on
every free parameter (``gate="all"``).

The work is one Slurm task per (model, fold): ``plan`` replays the rule over
the saved fold results and writes ``next.json`` with the (model, fold) fits
the next round still needs, or, when none is missing, ``elimination.json``,
``summary.md`` and ``lean/``. ``worker`` fits task ``task`` of ``next.json``
at the first of ``seeds`` that converges within ``time_limit_hours`` and
saves its held-out log densities (``cv/<model>/fold<k>.npz``) and every
attempt (``fold<k>.json``). A fold that converged at no seed is unconverged;
adding seeds and planning again retries it at the new ones only.

Each domain's ``eliminate`` module (`src.subjective_randomness.ablation.eliminate`,
`src.number_game.ablation.eliminate`) names its finals, CV unit and clusters
(an ``Ablation``) and its command line (a subclass of ``Args``).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

from src.ablation.variants import Variants

THRESHOLD_DSE = 2.0


@dataclass(frozen=True)
class Ablation:
    """What differs between ablations: the finals, the CV unit, the SE's clusters."""

    variants: Variants
    cv_group: Union[str, Tuple[str, ...]]
    """The grouped CV's unit (`src.model_comparison.cv.Selection.cv_group`)."""
    units: Callable[[Sequence[Mapping[str, Any]]], np.ndarray]
    """One integer per response row: the clusters of the SE of a difference."""


@dataclass
class Scored:
    pointwise: np.ndarray
    converged: bool

    @property
    def elpd(self) -> float:
        return float(self.pointwise.sum())


class Missing(Exception):
    """A round needs (model, fold) fits that are not saved yet."""

    def __init__(self, sets: List[Tuple[str, ...]]):
        super().__init__(f"{len(sets)} models need their CV")
        self.sets = sets


def eliminate(V: Variants, final: str, start: Sequence[str],
              score: Callable[[List[Tuple[str, ...]]], Dict[Tuple[str, ...], Scored]],
              units: np.ndarray, threshold_dse: float = THRESHOLD_DSE) -> dict:
    """The rule above. ``score`` maps mechanism sets to their CV."""
    from src.models.clustered_se import cluster_dse

    current = V.canonical(final, start)
    reason = V.invalid_reason(final, current)
    if reason:
        raise ValueError(f"the starting set is not a model of {final}: {reason}")
    rounds = []
    while True:
        drops = [V.canonical(final, set(current) - {m}) for m in V.removals(final, current)]
        scored = score([current] + drops)
        cur = scored[current]
        if not rounds and not cur.converged:
            raise RuntimeError(f"the starting model {V.variant_name(final, current)} did not converge on every fold; "
                               "no removal can be costed against it (add seeds and plan again)")
        rows = []
        for d in drops:
            (m,) = set(current) - set(d)
            s = scored[d]
            cost = cur.elpd - s.elpd
            se = float(cluster_dse(cur.pointwise, s.pointwise, units))
            rows.append(dict(mechanism=m, model=V.variant_name(final, d), elpd_cv=s.elpd, cost=float(cost), se=se,
                             cost_in_se=float(cost / se) if se > 0 else (0.0 if cost == 0 else float(np.sign(cost)) * np.inf),
                             converged=s.converged))
        rows.sort(key=lambda r: (not np.isfinite(r["cost"]), r["cost"]))
        record = dict(model=V.variant_name(final, current), mechanisms=list(current), elpd_cv=cur.elpd,
                      converged=cur.converged, size=V.size(final, current), candidates=rows)
        best = next((r for r in rows if r["converged"]), None)
        if best is not None and (best["cost"] <= 0 or best["cost"] < threshold_dse * best["se"]):
            record["decision"] = f"drop {best['mechanism']}"
            rounds.append(record)
            current = V.canonical(final, set(current) - {best["mechanism"]})
            continue
        record["decision"] = ("stop: nothing left to remove" if not rows else
                              "stop: no candidate converged" if best is None else
                              f"stop: the cheapest removal ({best['mechanism']}) costs {best['cost']:.1f} nats, "
                              f"{best['cost_in_se']:.2f} SE >= {threshold_dse}")
        rounds.append(record)
        break
    first = V.canonical(final, start)
    ends = score([first, current])
    return dict(
        final=final, rule=dict(threshold_dse=threshold_dse, start=list(first)), rounds=rounds,
        lean=dict(model=V.variant_name(final, current), mechanisms=list(current),
                  removed=[m for m in first if m not in current], size=V.size(final, current),
                  elpd_cv=ends[current].elpd, cost_vs_start=ends[first].elpd - ends[current].elpd,
                  se_vs_start=float(cluster_dse(ends[first].pointwise, ends[current].pointwise, units))),
    )


@dataclass
class Args:
    """The command line every ablation shares; a domain narrows ``final`` and documents ``data``."""

    final: str
    data: Path
    """The responses the ablation is scored on."""
    out: Path
    """This final's elimination directory."""
    mode: Literal["plan", "worker"] = "plan"
    task: int = -1
    """worker: the index into ``next.json``'s tasks (the array task id)."""
    folds: int = 5
    fold_seed: int = 0
    threshold_dse: float = THRESHOLD_DSE
    start: Optional[List[str]] = None
    """The mechanisms the elimination starts from (default: every one, the final itself)."""
    draws: int = 1000
    tune: int = 4000
    chains: int = 4
    target_accept: float = 0.95
    method: str = "nuts_no_jitter"
    """Sampler settings: the paper's fits on all runs pooled (scripts/paper/sr_figure_data.py), at 1,000 draws."""
    gate: Literal["population", "all"] = "population"
    """Which free parameters the convergence gate covers."""
    seeds: List[int] = field(default_factory=lambda: [20261018, 6])
    """worker: sampler seeds, tried in order until a fit converges (not part of the config: may grow)."""
    time_limit_hours: float = 22.0
    """worker: per sampling attempt; a timed-out attempt moves on to the next seed."""


def _selection(ablation: Ablation, args: Args):
    from src.model_comparison.cv import Selection

    return Selection(criterion="grouped_cv", cv_group=ablation.cv_group, cv_folds=args.folds, cv_seed=args.fold_seed)


def _config(ablation: Ablation, args: Args) -> dict:
    V = ablation.variants
    return dict(final=args.final, data_sha256=hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                folds=args.folds, fold_seed=args.fold_seed, threshold_dse=args.threshold_dse,
                start=list(V.canonical(args.final, args.start or V.spec(args.final).MECHANISMS)),
                sampler=dict(draws=args.draws, tune=args.tune, chains=args.chains, target_accept=args.target_accept,
                             method=args.method), gate=args.gate)


def _check_config(ablation: Ablation, args: Args, write: bool) -> None:
    """Every fit under ``out`` was made with one configuration (``config.json``)."""
    from src.runtime.atomic_files import write_text_atomically

    path = Path(args.out) / "config.json"
    mine = _config(ablation, args)
    if not path.exists():
        if not write:
            raise FileNotFoundError(f"{path} is missing (the plan step writes it)")
        Path(args.out).mkdir(parents=True, exist_ok=True)
        write_text_atomically(path, json.dumps(mine, indent=1) + "\n")
        return
    recorded = json.loads(path.read_text())
    if recorded != mine:
        diff = {k: (recorded.get(k), mine.get(k)) for k in set(recorded) | set(mine) if recorded.get(k) != mine.get(k)}
        raise ValueError(f"{Path(args.out)} was made with another configuration (recorded, now): {diff}; use a fresh --out")


def _fold_record(out: Path, name: str, fold: int) -> Optional[dict]:
    path = out / "cv" / name / f"fold{fold}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _fold_needed(record: Optional[dict], seeds: Sequence[int]) -> bool:
    if record is None:
        return True
    tried = {a["seed"] for a in record["attempts"]}
    return not record["converged"] and bool(set(seeds) - tried)


def _scored(out: Path, name: str, folds: np.ndarray, k: int) -> Scored:
    pointwise = np.full(len(folds), np.nan)
    converged = True
    for f in range(k):
        record = _fold_record(out, name, f)
        converged &= record["converged"]
        if record["has_lpd"]:
            pointwise[folds == f] = np.load(out / "cv" / name / f"fold{f}.npz")["lpd"]
    return Scored(pointwise, bool(converged))


def plan(ablation: Ablation, args: Args) -> dict:
    """Replay the rule over the saved fold results: the next round's missing fits, or the result."""
    from src.model_comparison.cv import fold_training_files
    from src.runtime.atomic_files import write_text_atomically

    V = ablation.variants
    out = Path(args.out)
    _check_config(ablation, args, write=True)
    rows, folds, _ = fold_training_files(args.data, _selection(ablation, args))  # written once here, read by the workers
    units = ablation.units(rows)

    def score(sets):
        names = {s: V.variant_name(args.final, s) for s in sets}
        missing = [s for s in sets if any(_fold_needed(_fold_record(out, names[s], f), args.seeds)
                                          for f in range(args.folds))]
        if missing:
            raise Missing(missing)
        return {s: _scored(out, names[s], folds, args.folds) for s in sets}

    nxt = out / "next.json"
    before = json.loads(nxt.read_text()) if nxt.exists() else None
    try:
        record = eliminate(V, args.final, _config(ablation, args)["start"], score, units, args.threshold_dse)
    except Missing as m:
        tasks = []
        for s in m.sets:
            name, _ = V.write_variant(args.final, s, out / "models")  # written once here, read by the workers
            tasks += [dict(model=name, mechanisms=list(s), fold=f) for f in range(args.folds)
                      if _fold_needed(_fold_record(out, name, f), args.seeds)]
        same = before and before.get("tasks") == tasks and before.get("seeds") == list(args.seeds)
        attempt = before["attempt"] + 1 if same else 1  # new seeds start the count again
        doc = dict(status="need", tasks=tasks, seeds=list(args.seeds), attempt=attempt)
        write_text_atomically(nxt, json.dumps(doc, indent=1) + "\n")
        print(f"[plan {args.final}] {len(tasks)} fits to make for {len(m.sets)} models (attempt {attempt}): "
              f"{', '.join(V.variant_name(args.final, s) for s in m.sets)}", flush=True)
        return doc
    _finish(ablation, args, out, record, n_trials=len(rows))
    doc = dict(status="done", lean=record["lean"])
    write_text_atomically(nxt, json.dumps(doc, indent=1) + "\n")
    return doc


def _finish(ablation: Ablation, args: Args, out: Path, record: dict, n_trials: int) -> None:
    from src.runtime.atomic_files import write_text_atomically

    V = ablation.variants
    name, path = V.write_variant(args.final, record["lean"]["mechanisms"], out / "models")
    (out / "lean").mkdir(exist_ok=True)
    shutil.copyfile(path, out / "lean" / f"{name}.py")
    write_text_atomically(out / "lean" / "models_manifest.yaml", f"models:\n  - name: {name}\n")
    record.update(config=_config(ablation, args), n_trials=n_trials,
                  lean_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    write_text_atomically(out / "elimination.json", json.dumps(record, indent=1) + "\n")
    summary = summary_markdown(V, record)
    write_text_atomically(out / "summary.md", summary)
    print(summary, flush=True)


def summary_markdown(V: Variants, record: dict) -> str:
    lean = record["lean"]
    lines = [f"# {record['final']} ({V.spec(record['final']).FINAL}): backward elimination", "",
             "| round | model | params | ELPD-CV | cheapest removal | cost (nats) | cost / SE | decision |",
             "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(record["rounds"], 1):
        c = next((c for c in r["candidates"] if c["converged"]), None)
        size = f"{r['size']['population']} + {r['size']['person_vectors']}p"
        lines.append(f"| {i} | `{r['model']}` | {size} | {r['elpd_cv']:.1f} | "
                     + (f"{c['mechanism']} | {c['cost']:.1f} ± {c['se']:.1f} | {c['cost_in_se']:.2f} | " if c else "— | — | — | ")
                     + f"{r['decision']} |")
    lines += ["", f"**Lean model:** `{lean['model']}`, {lean['size']['population']} population parameters + "
              f"{lean['size']['person_vectors']} person-level vectors. Kept: {', '.join(lean['mechanisms']) or 'the core only'}. "
              f"Removed: {', '.join(lean['removed']) or 'nothing'}. Against the starting model"
              f"{' (the full final)' if len(record['rule']['start']) == len(V.spec(record['final']).MECHANISMS) else ''}: "
              f"{lean['cost_vs_start']:.1f} ± {lean['se_vs_start']:.1f} nats of ELPD-CV.", ""]
    return "\n".join(lines)


def worker(ablation: Ablation, args: Args) -> dict:
    """Fit task ``task`` of ``next.json`` and save its held-out log densities."""
    from src.model_comparison.cv import fold_training_files
    from src.models.data_binding import make_stim_data
    from src.models.pymc_inference import FitTimeLimitExceeded, convergence_problems, fit_model
    from src.runtime.atomic_files import write_text_atomically

    V = ablation.variants
    out = Path(args.out)
    _check_config(ablation, args, write=False)
    doc = json.loads((out / "next.json").read_text())
    if doc["status"] != "need" or not 0 <= args.task < len(doc["tasks"]):
        raise ValueError(f"task {args.task} is not one of next.json's {len(doc.get('tasks', []))} tasks")
    task = doc["tasks"][args.task]
    name, fold = task["model"], task["fold"]
    if V.variant_name(args.final, task["mechanisms"]) != name or not (out / "models" / f"{name}.py").exists():
        raise ValueError(f"next.json's task {args.task} ({name}) has no model file of that name under {out / 'models'}")
    rows, folds, paths = fold_training_files(args.data, _selection(ablation, args))
    if not all(p.exists() for p in paths):
        raise FileNotFoundError(f"fold files missing: {paths} (the plan step writes them)")
    cv_dir = out / "cv" / name
    cv_dir.mkdir(parents=True, exist_ok=True)
    record = _fold_record(out, name, fold) or dict(model=name, fold=fold, converged=False, has_lpd=False, attempts=[])
    held_rows = [rows[i] for i in np.flatnonzero(folds == fold)]
    for seed in args.seeds:
        if record["converged"] or seed in {a["seed"] for a in record["attempts"]}:
            continue
        t0 = time.time()
        try:
            fitted = fit_model(name, out / "models", paths[fold], draws=args.draws, tune=args.tune, chains=args.chains,
                               cores=args.chains, random_seed=seed, target_accept=args.target_accept,
                               method=args.method, time_limit_sec=args.time_limit_hours * 3600)
        except FitTimeLimitExceeded as e:
            record["attempts"].append(dict(seed=seed, problems=[str(e)], problems_all=[str(e)],
                                           seconds=round(time.time() - t0)))
            print(f"[worker] {name} fold {fold} seed {seed}: timed out", flush=True)
        else:
            free = [rv.name for rv in fitted.model.free_RVs]
            population = [n for n in free if fitted.idata.posterior[n].ndim == 2]  # (chain, draw): a scalar
            problems = convergence_problems(fitted.idata, population if args.gate == "population" else free)
            lpd = fitted.heldout_lpd(make_stim_data(fitted.model, held_rows))
            np.savez(cv_dir / f".fold{fold}.tmp.npz", lpd=lpd)
            (cv_dir / f".fold{fold}.tmp.npz").replace(cv_dir / f"fold{fold}.npz")
            record.update(converged=not problems, has_lpd=True, elpd=float(lpd.sum()), seed=seed)
            record["attempts"].append(dict(seed=seed, problems=problems,
                                           problems_all=convergence_problems(fitted.idata, free),
                                           seconds=round(time.time() - t0)))
            print(f"[worker] {name} fold {fold} seed {seed}: held-out lpd {lpd.sum():.1f}, "
                  f"{'converged' if not problems else problems}", flush=True)
            del fitted
        # After every attempt: a task killed later keeps what it did.
        write_text_atomically(cv_dir / f"fold{fold}.json", json.dumps(record, indent=1) + "\n")
    return record
