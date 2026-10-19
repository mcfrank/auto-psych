"""The models a run admitted and later removed, scored on its final data, for
Figure 2's "pruned during the run" (``sr_figure_data.py``,
``number_game_figure_data.py``).

A removed model is one the run's ledger records as ``admitted`` that is
neither live at the end nor a seed: pruned, retired by the cap or dropped as
unfittable.

A model pruned at the end of the run's final experiment needs no fit: the
final experiment's ``history.json`` records its ELPD on exactly the final
data at the run's settings (``recorded_scores``; pruning acts only on a
trusted fit, converged and with a reliable PSIS-LOO). A refit reproduces it
(to four decimals, number game, 2026-10-19). Every other removed model --
pruned in an earlier experiment, so fitted only on less data, retired by the
live-set cap or dropped -- is fitted on the final data. There are about 50
per run, so they are scored in shards (a Slurm array,
``scripts/paper/submit_figure_data.sh``), each model on its own: one record per model, ``<records_dir>/<name>.json``, holding its score
or why it has none, with the inputs it was made from. A shard skips the
models whose record is current, so a resubmitted array scores only what is
left. The figure-data scripts then read every record and raise if one is
missing.

A removed model whose fit fails or does not converge (or, under PSIS-LOO,
whose estimate is unreliable) is recorded as ``unscored`` with the reason; it
does not stop the shard, since it is not one of the models the comparison is
about. An infrastructure failure (``FitInfrastructureFailure``) raises.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from scripts.paper import sr_figure_data as sr


def source_file(run_dir: Path, name: str) -> Path:
    """A model's file in the run: in any experiment's ``models/`` or
    ``models/pruned/``, where every copy must be identical."""
    copies = sorted(run_dir.glob(f"experiment*/model_loop/models/{name}.py"))
    copies += sorted(run_dir.glob(f"experiment*/model_loop/models/pruned/{name}.py"))
    if not copies:
        raise FileNotFoundError(f"no source file of the model {name!r} in {run_dir}")
    if len({sr._sha256(p) for p in copies}) > 1:
        raise RuntimeError(f"the model {name!r} has differing copies: {copies}")
    return copies[0]


def removed_models(run_dir: Path, final_dir: Path, *, live: Sequence[str],
                   seeds: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """Every model the run admitted and later removed: name -> its source
    file and the ledger's last entry for it (``outcome``: ``pruned`` or
    ``dropped``; its ``detail`` and ``context``). The final ledger continues
    every earlier experiment's. A model's copies (in
    any experiment's ``models/`` or ``models/pruned/``) must be identical."""
    from src.pipelines.inner_loop.hypothesis_ledger import HypothesisLedger

    entries = HypothesisLedger(final_dir / "model_loop" / "attempted_hypotheses.jsonl").entries()
    admitted = [e.name for e in entries if e.outcome == "admitted"]
    last = {e.name: e for e in entries}
    removed: Dict[str, Dict[str, Any]] = {}
    for name in dict.fromkeys(admitted):
        if name in live or name in seeds:
            continue
        if last[name].outcome == "admitted":
            raise RuntimeError(f"{name!r} was admitted, is not live at the end and was never removed")
        removed[name] = {"file": source_file(run_dir, name), "outcome": last[name].outcome,
                         "detail": last[name].detail, "context": last[name].context}
    return removed


RETIRED_MARK = "retired to keep the live set"  # in the ledger detail of a cap retirement


def recorded_scores(final_dir: Path, removed: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    """The removed models pruned (not retired) at the end of the final
    experiment, each with the ELPD the final experiment's ``history.json``
    records for it: on the final data, at the run's settings. Every step
    that lists a model must give it the same ELPD (the data do not change
    within an experiment)."""
    history = json.loads((final_dir / "model_loop" / "history.json").read_text())
    unreliable = {n for step in history for n in step.get("excluded_unreliable") or []}
    scores: Dict[str, float] = {}
    for name, entry in removed.items():
        if (entry["outcome"] != "pruned" or RETIRED_MARK in entry["detail"]
                or not entry["context"].startswith(f"{final_dir.name} end of experiment")):
            continue
        values = {float(step["elpd_loo"][name]) for step in history
                  if name in (step.get("elpd_loo") or {})}
        if len(values) != 1:
            raise RuntimeError(
                f"{name!r} was pruned at the end of {final_dir.name}, whose history records "
                f"{len(values)} distinct ELPDs for it ({sorted(values)})"
            )
        if name in unreliable:
            raise RuntimeError(f"{name!r} was pruned, but {final_dir.name}'s history marks its "
                               "PSIS-LOO unreliable")
        scores[name] = values.pop()
    return scores


def shard_of(items: Sequence[Any], shard: int, n_shards: int) -> List[Any]:
    """Shard ``shard`` of ``n_shards`` (every ``n_shards``-th item)."""
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard {shard} is not in 0..{n_shards - 1}")
    return list(items[shard::n_shards])


def copy_atomically(source: Path, dest: Path) -> None:
    """Copy ``source`` to ``dest`` through a temporary file and a rename, so
    that concurrent shards never read a half-written copy."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.")
    os.close(fd)
    shutil.copyfile(source, tmp)
    os.replace(tmp, dest)


def record_path(records_dir: Path, name: str) -> Path:
    return records_dir / f"{name}.json"


def score_shard(
    files: Dict[str, Path],
    *,
    models_dir: Path,
    records_dir: Path,
    inputs: Callable[[str], Dict[str, Any]],
    training_files: Sequence[Tuple[str, Path]],
    score: Callable[[str], Dict[str, Any]],
    cache_dir: Path,
    fit_kwargs: Dict[str, Any],
) -> None:
    """Score each model of ``files`` (name -> source) that has no current
    record. They are fitted first, concurrently and tolerantly, on every
    ``training_files`` entry (label, responses); ``score(name)`` then reads
    the cached fits and returns ``{"elpd": ...}`` or ``{"unscored": reason}``.
    The in-process fit cache is cleared after each file and each model (a
    loaded fit stays in memory)."""
    from src.models.pymc_inference import clear_fit_cache, fit_models_to_cache

    todo = [n for n in files if sr.load_scored(record_path(records_dir, n), inputs(n)) is None]
    print(f"[removed] {len(files) - len(todo)} of {len(files)} already scored", flush=True)
    if not todo:
        return
    sr.write_model_set(models_dir, {n: files[n] for n in todo})
    cache_dir.mkdir(parents=True, exist_ok=True)
    failed: Dict[str, str] = {}
    for label, path in training_files:
        pending = [n for n in todo if n not in failed]
        print(f"[removed] fitting {len(pending)} models on {label}", flush=True)
        for name, reason in fit_models_to_cache(pending, models_dir, path, cache_dir=cache_dir,
                                                **fit_kwargs).items():
            failed[name] = f"fit failed on {label} ({reason})"
        clear_fit_cache()
    for name in todo:
        result = {"unscored": failed[name]} if name in failed else score(name)
        clear_fit_cache()
        sr.save_scored(record_path(records_dir, name), inputs(name), result)
        print(f"[removed] {name}: {result}", flush=True)


def collect(names: Sequence[str], *, records_dir: Path,
            inputs: Callable[[str], Dict[str, Any]]) -> Dict[str, Any]:
    """Every model's record: ``{"elpd": {name: ...}, "unscored": {name: reason}}``.
    Raises naming the models that have no current record."""
    records = {n: sr.load_scored(record_path(records_dir, n), inputs(n)) for n in names}
    missing = [n for n, r in records.items() if r is None]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(names)} removed models have no current scored record in "
            f"{records_dir} (score them first with --removed-shard): {missing}"
        )
    elpd = {n: r["record"]["elpd"] for n, r in records.items() if "elpd" in r["record"]}
    unscored = {n: r["record"]["unscored"] for n, r in records.items() if "unscored" in r["record"]}
    return {"elpd": elpd, "unscored": unscored}
