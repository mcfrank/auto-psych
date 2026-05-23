"""Persistent model zoo.

Every successfully-fit candidate is recorded under ``<results_dir>/model_zoo/``
so Bayesian model comparison can score the full population (not just the
running tournament winner). Non-winners no longer disappear once the next round
beats them — they remain available as alternative hypotheses.

Layout::

    <results_dir>/model_zoo/
        iter_0__candidate_0/
            cognitive_model.py
            fit_result.json
            origin.json        # provenance: iteration, candidate_id, when_admitted
        iter_0__candidate_1/
        iter_init/             # the initial_model baseline (optional)
        ...

The zoo is content-addressed only by ``entry_id`` (caller-chosen). Re-recording
the same id is idempotent — useful for resume paths where a round is replayed.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from src.pipelines.inner_loop.fitting import FitResult
from src.pipelines.inner_loop.history import _fit_result_to_dict, _load_fit_result


@dataclass
class ZooEntry:
    entry_id: str
    dir: Path
    model_path: Path
    fit_path: Path
    origin_path: Path

    def load_fit(self) -> FitResult:
        return _load_fit_result(self.fit_path)


def candidate_entry_id(iteration: int, candidate_id: str) -> str:
    """Canonical zoo id for a tournament candidate."""
    return f"iter_{iteration}__{candidate_id}"


INITIAL_ENTRY_ID = "iter_init"


def _entry_paths(zoo_dir: Path, entry_id: str) -> ZooEntry:
    d = zoo_dir / entry_id
    return ZooEntry(
        entry_id=entry_id,
        dir=d,
        model_path=d / "cognitive_model.py",
        fit_path=d / "fit_result.json",
        origin_path=d / "origin.json",
    )


def record_entry(
    zoo_dir: Path,
    entry_id: str,
    model_path: Path,
    fit_result: FitResult,
    *,
    origin: dict | None = None,
) -> ZooEntry:
    """Copy ``model_path`` and dump ``fit_result`` under ``zoo_dir / entry_id``.

    Idempotent: overwrites any existing entry with the same id (useful when a
    round is re-fit during resume). Returns the populated ``ZooEntry``.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"model file not found: {model_path}")

    zoo_dir.mkdir(parents=True, exist_ok=True)
    entry = _entry_paths(zoo_dir, entry_id)
    entry.dir.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(model_path, entry.model_path)
    entry.fit_path.write_text(json.dumps(_fit_result_to_dict(fit_result), indent=2))

    payload = dict(origin or {})
    payload.setdefault("entry_id", entry_id)
    payload.setdefault("source_model_path", str(model_path))
    payload["recorded_at"] = datetime.now(timezone.utc).isoformat()
    entry.origin_path.write_text(json.dumps(payload, indent=2))

    return entry


def record_candidate(
    zoo_dir: Path,
    iteration: int,
    candidate_id: str,
    model_path: Path,
    fit_result: FitResult,
) -> ZooEntry:
    """Convenience wrapper: derive entry_id from (iteration, candidate_id)."""
    return record_entry(
        zoo_dir,
        candidate_entry_id(iteration, candidate_id),
        model_path,
        fit_result,
        origin={"iteration": iteration, "candidate_id": candidate_id},
    )


def record_initial(
    zoo_dir: Path,
    model_path: Path,
    fit_result: FitResult,
) -> ZooEntry:
    """Record the initial / baseline model under a fixed id."""
    return record_entry(
        zoo_dir,
        INITIAL_ENTRY_ID,
        model_path,
        fit_result,
        origin={"role": "initial_model"},
    )


def get_entry(zoo_dir: Path, entry_id: str) -> ZooEntry | None:
    entry = _entry_paths(zoo_dir, entry_id)
    return entry if entry.fit_path.exists() and entry.model_path.exists() else None


def iter_zoo(zoo_dir: Path) -> Iterator[ZooEntry]:
    """Yield every well-formed entry in the zoo, sorted by entry_id.

    Skips directories that lack ``cognitive_model.py`` or ``fit_result.json``
    so a partially-written entry doesn't break iteration.
    """
    if not zoo_dir.exists():
        return
    for child in sorted(zoo_dir.iterdir()):
        if not child.is_dir():
            continue
        entry = _entry_paths(zoo_dir, child.name)
        if entry.model_path.exists() and entry.fit_path.exists():
            yield entry


def zoo_size(zoo_dir: Path) -> int:
    return sum(1 for _ in iter_zoo(zoo_dir))
