"""Promote run 2's models to seed the live experiments (PI decision 2026-10-08).

    uv run python -m src.rsa.promote --sweep data/rsa/sherlock_run2 \
        --train <train.csv> --test <test.csv> --work <scratch dir> --out data/rsa/live_seeds

The rule, fixed before any live data:

1. **Candidates**: every model admitted in the sweep's real-data cells, live
   at the end or pruned (the starting models are not candidates). Identical
   sources count once. The pruned models are included on purpose: the models
   that did best on held-out conditions in real_rep3 exist only in the
   pruned set, and the live sets alone would carry each cell's path
   dependence (one lineage per replicate) into the live phase.
2. **Refit** each on all existing data, training and held-out trials
   together (claim 1 is scored; the test split goes back in), with the
   loop's fit rule (`loop_fit`) and settings.
3. **Group by behaviour**: average-linkage clustering of the candidates'
   posterior-mean choice probabilities on the novelty pool (RMSE distance,
   the novelty gate's), cut into ``k`` groups.
4. **Keep one per group**: the member with the best grouped CV (5 folds of
   whole conditions over all data, as the loop selects); a member whose fit
   or CV failed to converge only when no member converged.
5. Add ``rsa_l2``, the starting model every claim is measured against.

It reads predictions and fit only: never hypotheses or source labels. The
reason is experimental design: the EIG design can only aim at disagreements
between the models it is given, and near-duplicates give it nothing.

Writes ``<out>/models/`` (the promoted model files and a manifest) and
``<out>/promotion.json`` (every candidate's fit, CV, group and distances; the
groups and the choice). The combined trial file and fits stay in ``--work``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tyro
import yaml

from src.rsa.fit import FitSettings
from src.rsa.loop.cv import DEFAULT_FOLDS, CVResult, cv_pointwise, make_folds
from src.rsa.loop.fitting import FIT_TIME_LIMIT_SEC, ModelFailure, loop_fit
from src.rsa.loop.novelty import novelty_pool, pool_digest, posterior_mean_class_probs
from src.rsa.model_file import RSAModel
from src.runtime.config import PROJECT_ASSETS_DIR

SEED_DIR = PROJECT_ASSETS_DIR / "rsa_reference" / "seed_models"
REFERENCE = "rsa_l2"


@dataclass
class Args:
    sweep: Path
    """The brought-back sweep (its real_rep*/ cells: models, ledger, history, cell.json)."""
    train: Path
    """The sweep's training trials (train.csv)."""
    test: Path
    """The sweep's held-out trials (test.csv)."""
    work: Path
    """Scratch: the combined trials, folds and fit cache."""
    out: Path
    """Where the promoted models and promotion.json go."""
    k: int = 10
    """Groups, and so promoted models (rsa_l2 comes on top)."""
    cells: List[str] = field(default_factory=lambda: ["real_rep1", "real_rep2", "real_rep3"])
    folds: int = DEFAULT_FOLDS
    fold_seed: int = 0
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    workers: int = 0
    """Fits at once (each in its own process); 0: one per CPU."""
    time_limit_sec: float = FIT_TIME_LIMIT_SEC


@dataclass
class Candidate:
    key: str  # <cell>/<name>
    name: str
    cell: str
    path: Path
    sha: str
    status: str  # live | pruned (at the end of its cell)


def candidates(sweep: Path, cells: List[str]) -> List[Candidate]:
    """Every model a cell admitted (live or pruned at the end), the starting
    models excluded, identical sources once."""
    out, seen = [], set()
    for cell in cells:
        cdir = Path(sweep) / cell
        history = json.loads((cdir / "history.json").read_text())
        seeds = {e["name"] for e in history[0]["events"] if e.get("outcome") == "seeded"}
        ledger = [json.loads(line) for line in (cdir / "attempted_hypotheses.jsonl").read_text().splitlines() if line.strip()]
        admitted = [e["name"] for e in ledger if e["outcome"] == "admitted" and e["name"] not in seeds]
        for name in dict.fromkeys(admitted):
            live, pruned = cdir / "models" / f"{name}.py", cdir / "models" / "pruned" / f"{name}.py"
            path = live if live.exists() else pruned
            if not path.exists():
                raise FileNotFoundError(f"{cell}: admitted model {name} has no file in models/ or models/pruned/")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            if sha in seen:
                continue
            seen.add(sha)
            out.append(Candidate(f"{cell}/{name}", name, cell, path, sha, "live" if path == live else "pruned"))
    return out


def combined_trials(train: Path, test: Path, out: Path) -> Path:
    """Training and held-out trials in one file (written once)."""
    out = Path(out)
    if not out.exists():
        frame = pd.concat([pd.read_csv(train), pd.read_csv(test)], ignore_index=True)
        tmp = out.with_suffix(".tmp.csv")
        frame.to_csv(tmp, index=False)
        tmp.replace(out)
    return out


def cluster(preds: Dict[str, np.ndarray], k: int) -> Dict[str, int]:
    """Average-linkage groups of the pool predictions (RMSE), cut into k."""
    from scipy.cluster.hierarchy import cut_tree, linkage
    from scipy.spatial.distance import squareform

    names = list(preds)
    if len(names) <= k:
        return {n: i for i, n in enumerate(names)}
    x = np.stack([preds[n] for n in names])
    d = np.sqrt(((x[:, None, :] - x[None, :, :]) ** 2).mean(-1))
    # cut_tree gives exactly k groups (fcluster's maxclust can return fewer).
    labels = cut_tree(linkage(squareform(d, checks=False), method="average"), n_clusters=k).ravel()
    order = {lab: i for i, lab in enumerate(dict.fromkeys(labels))}  # groups numbered by first member
    return {n: order[lab] for n, lab in zip(names, labels)}


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main(args: Args) -> dict:
    work, out = Path(args.work), Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    cands = candidates(args.sweep, args.cells)
    if len(cands) < args.k:
        raise ValueError(f"{len(cands)} candidates for {args.k} groups")
    data = combined_trials(args.train, args.test, work / "all_trials.csv")
    settings = FitSettings(num_warmup=args.num_warmup, num_samples=args.num_samples, num_chains=args.num_chains,
                           seed=args.fold_seed)
    cache = work / ".fit_cache"
    workers = args.workers or max(1, len(__import__("os").sched_getaffinity(0)))
    pool = novelty_pool()
    print(f"{len(cands)} candidates from {', '.join(args.cells)}; fitting on {data} ({workers} at once)", flush=True)

    # 2. refit on all data (each fit in its own time-limited process)
    def fit_one(c: Candidate):
        try:
            fitted = loop_fit(c.path, c.name, data, settings, cache, time_limit_sec=args.time_limit_sec)
            return c.key, fitted, posterior_mean_class_probs(RSAModel(c.path, name=c.name), fitted, pool), None
        except (ModelFailure, ValueError) as exc:  # ValueError: not finite on the pool
            return c.key, None, None, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        fitted = {key: (fit, pr, err) for key, fit, pr, err in ex.map(fit_one, cands)}
    usable = [c for c in cands if fitted[c.key][0] is not None]
    print(f"  {len(usable)} of {len(cands)} fitted; failures: "
          f"{ {k: v[2] for k, v in fitted.items() if v[2]} }", flush=True)

    # 3. group by behaviour
    preds = {c.key: fitted[c.key][1] for c in usable}
    group = cluster(preds, args.k)

    # 4. grouped CV on all data, best per group
    folds = make_folds(data, work / ".cv", args.folds, seed=args.fold_seed)

    def cv_one(c: Candidate):
        try:
            return c.key, cv_pointwise(c.path, c.name, data, folds, settings, cache,
                                       time_limit_sec=args.time_limit_sec, workers=args.folds)
        except ModelFailure as exc:
            print(f"  {c.key}: no grouped CV ({exc})", flush=True)
            return c.key, None

    with ThreadPoolExecutor(max_workers=max(1, workers // args.folds)) as ex:
        cv: Dict[str, Optional[CVResult]] = dict(ex.map(cv_one, usable))
    chosen = {}
    for g in sorted(set(group.values())):
        members = [c for c in usable if group[c.key] == g and cv[c.key] is not None]
        if not members:
            continue
        good = [c for c in members if cv[c.key].converged and fitted[c.key][0].converged] or members
        chosen[g] = max(good, key=lambda c: cv[c.key].elpd)

    # 5. write the promoted set (names made unique across cells), rsa_l2 on top
    models_dir = out / "models"
    if models_dir.exists():
        shutil.rmtree(models_dir)
    models_dir.mkdir(parents=True)
    entries, names = [], set()
    for g, c in sorted(chosen.items()):
        name = c.name if c.name not in names else f"{c.name}_{c.cell.replace('real_', '')}"
        names.add(name)
        shutil.copyfile(c.path, models_dir / f"{name}.py")
        entries.append(dict(name=name, rationale=_doc(c.path), source=f"{c.cell}/{c.status}/{c.name}", group=g))
    shutil.copyfile(SEED_DIR / f"{REFERENCE}.py", models_dir / f"{REFERENCE}.py")
    entries.append(dict(name=REFERENCE, rationale=_doc(SEED_DIR / f"{REFERENCE}.py"), source="starting model (reference)"))
    (models_dir / "models_manifest.yaml").write_text(yaml.safe_dump({"models": entries}, sort_keys=False))

    best_cv = max(r.elpd for r in cv.values() if r is not None)
    record = dict(
        rule=__doc__.split("The rule, fixed before any live data:", 1)[1].split("Writes ``", 1)[0].strip(),
        sweep=str(args.sweep), cells=args.cells, k=args.k, linkage="average", distance="RMSE of posterior-mean "
        "choice-class probabilities over the novelty pool", pool_digest=pool_digest(pool), n_pool=len(pool),
        settings=vars(settings), folds=args.folds, n_trials=int(len(folds.fold)),
        data_sha256={p.name: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (Path(args.train), Path(args.test))},
        candidates=[dict(
            key=c.key, cell=c.cell, name=c.name, status=c.status, sha256=c.sha,
            fit_error=fitted[c.key][2],
            converged=None if fitted[c.key][0] is None else fitted[c.key][0].converged,
            group=group.get(c.key),
            elpd_cv=None if cv.get(c.key) is None else cv[c.key].elpd,
            cv_behind_best=None if cv.get(c.key) is None else best_cv - cv[c.key].elpd,
            cv_converged=None if cv.get(c.key) is None else cv[c.key].converged,
            chosen=any(ch.key == c.key for ch in chosen.values()),
            rmse_to_chosen=None if c.key not in preds or group[c.key] not in chosen
            else rmse(preds[c.key], preds[chosen[group[c.key]].key]),
        ) for c in cands],
        groups=[dict(group=g, chosen=chosen[g].key if g in chosen else None,
                     members=[c.key for c in usable if group[c.key] == g]) for g in sorted(set(group.values()))],
        between_chosen_rmse={a.key: {b.key: rmse(preds[a.key], preds[b.key]) for b in chosen.values()}
                             for a in chosen.values()},
        promoted=[e["name"] for e in entries],
    )
    (out / "promotion.json").write_text(json.dumps(record, indent=1))
    print(f"promoted {len(entries)} models to {models_dir}: {[e['name'] for e in entries]}")
    return record


def _doc(path: Path) -> str:
    import ast

    doc = ast.get_docstring(ast.parse(Path(path).read_text(encoding="utf-8"))) or ""
    return " ".join(doc.split())


if __name__ == "__main__":
    main(tyro.cli(Args))
