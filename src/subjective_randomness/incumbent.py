"""The incumbent record: does the loop's exported best model ever change?

The loop-improvement plan (September 2026) found that across the three archived
``motif_stack`` cells of the 20-cell sweep — 27 scoring steps — the exported
best model was the same seed at every step: the loop had never once beaten its
own starting point, and RMSE drift hid that. This module makes the question a
first-class output of every cell, so later changes to the loop are judged on
it rather than on RMSE alone.

Vocabulary:

* A **scoring step** is one entry of an experiment's inner-loop
  ``history.json`` (the seed step, then one per candidate round). A cell's
  steps are its experiments' histories concatenated in experiment order,
  numbered by ``global_step`` from 0 — exactly the rows of the harness's
  trajectory.
* The **incumbent** at a step is its ``best_model``: the model the loop would
  export and carry (``scoring._best_exportable_model``).
* The **starting models** of a cell are the models scored at experiment 1's
  seed step — the project seeds the cell was seeded with (the held-out ground
  truth excluded), minus any dropped as unfittable before scoring. A
  **discovered** incumbent is one that is not a starting model. This is read
  from the run record itself rather than from a manifest, so an archived
  cell or an offline re-analysis needs no configuration to compute it.
* ``incumbent_changed`` at a step means its incumbent differs from the
  previous step's. Nothing precedes step 0, so it is never a change.

The per-step flags are appended to every trajectory row (``INCUMBENT_COLUMNS``
extends ``holdout_eval.TRAJECTORY_COLUMNS``) and the per-cell summary is
written as ``trajectory.json``'s ``incumbent`` block. The verifier
(``verify_holdout_run.sh``) warns — never fails — on a finished cell with zero
incumbent changes: zero is the true baseline value and must not block a run.
Everything here is plain Python over dicts: no MCMC, no numpy.
"""

from __future__ import annotations

import json
import re
import tarfile
from pathlib import Path
from typing import Any, Collection, Dict, FrozenSet, List, Mapping, Optional, Sequence

INCUMBENT_COLUMNS = ("incumbent_changed", "incumbent_is_discovered")

_HISTORY_MEMBER_RE = re.compile(r"^(?P<run>.*?)experiment(?P<num>\d+)/model_loop/history\.json$")


# ─────────────────────────────────────────────
# Per-step flags and per-cell summary
# ─────────────────────────────────────────────


def _require_starting_models(starting_models: Collection[str]) -> FrozenSet[str]:
    starting = frozenset(starting_models)
    if not starting:
        raise ValueError(
            "The starting model set is empty: every incumbent would read as "
            "discovered. Pass the models the cell was seeded with."
        )
    return starting


def annotate_incumbents(
    steps: Sequence[Mapping[str, Any]], starting_models: Collection[str]
) -> List[Dict[str, Any]]:
    """Copy the per-step rows, adding ``incumbent_changed`` and
    ``incumbent_is_discovered`` (see the module docstring for both).

    ``steps`` must be a whole cell in order: ``global_step`` consecutive from
    0, or the "previous step" would be the wrong one. The input is not mutated.
    """
    starting = _require_starting_models(starting_models)
    annotated: List[Dict[str, Any]] = []
    previous: Optional[str] = None
    for position, step in enumerate(steps):
        if int(step["global_step"]) != position:
            raise ValueError(
                f"Trajectory rows must be a whole cell in order (global_step "
                f"consecutive from 0); row {position} has global_step "
                f"{step['global_step']!r}."
            )
        incumbent = str(step["best_model"])
        row = dict(step)
        row["incumbent_changed"] = previous is not None and incumbent != previous
        row["incumbent_is_discovered"] = incumbent not in starting
        annotated.append(row)
        previous = incumbent
    return annotated


def summarise_incumbents(
    annotated: Sequence[Mapping[str, Any]], starting_models: Collection[str]
) -> Dict[str, Any]:
    """The per-cell record: how often the incumbent changed, at how many steps
    it was a discovered model, what each change was, and where it ended.

    Takes rows already passed through ``annotate_incumbents`` (a bare row
    raises ``KeyError`` on the missing flag rather than silently recounting).
    """
    starting = _require_starting_models(starting_models)
    changes: List[Dict[str, Any]] = []
    previous: Optional[str] = None
    n_discovered = 0
    for row in annotated:
        incumbent = str(row["best_model"])
        if row["incumbent_changed"]:
            changes.append(
                {
                    "global_step": int(row["global_step"]),
                    "experiment": row["experiment"],
                    "step": row["step"],
                    "from": previous,
                    "to": incumbent,
                }
            )
        if row["incumbent_is_discovered"]:
            n_discovered += 1
        previous = incumbent
    return {
        "starting_models": sorted(starting),
        "n_steps": len(annotated),
        "n_incumbent_changes": len(changes),
        "n_steps_discovered_incumbent": n_discovered,
        "final_incumbent": previous,
        "changes": changes,
    }


# ─────────────────────────────────────────────
# Reading the record out of history.json files
# ─────────────────────────────────────────────


def starting_model_set(first_history: Sequence[Mapping[str, Any]]) -> FrozenSet[str]:
    """The models a cell started with: those scored at experiment 1's seed step.

    The seed step is the first history entry (``step`` 0, ``iteration`` None);
    its ``posteriors`` keys are the model set scored before any candidate
    round. Anything else as the opening entry means this is not a whole
    experiment history, and raises.
    """
    if not first_history:
        raise ValueError("Cannot read the starting model set from an empty history.")
    first = first_history[0]
    if first.get("step") != 0 or first.get("iteration") is not None:
        raise ValueError(
            "The first history entry is not the seed step (step 0, iteration "
            f"None): step={first.get('step')!r} iteration={first.get('iteration')!r}."
        )
    names = frozenset(str(name) for name in first["posteriors"])
    if not names:
        raise ValueError("The seed step scored no models (empty posteriors).")
    return names


def starting_models_of_run(run_root: Path) -> FrozenSet[str]:
    """``starting_model_set`` of ``run_root/experiment1/model_loop/history.json``."""
    history_path = Path(run_root) / "experiment1" / "model_loop" / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(
            f"Cannot read the cell's starting model set: {history_path} does not exist."
        )
    return starting_model_set(json.loads(history_path.read_text(encoding="utf-8")))


def steps_from_histories(
    histories: Sequence[Sequence[Mapping[str, Any]]],
) -> List[Dict[str, Any]]:
    """Bare trajectory rows for a cell: one per scoring step of every
    experiment's history, in order, with 1-based ``experiment`` and
    ``global_step`` numbered from 0 — the same numbering
    ``holdout_eval.evaluate_trajectory`` gives its metric rows.
    """
    steps: List[Dict[str, Any]] = []
    for exp_num, history in enumerate(histories, start=1):
        if not history:
            raise ValueError(f"Empty inner-loop history for experiment {exp_num}.")
        for entry in history:
            steps.append(
                {
                    "experiment": exp_num,
                    "step": entry["step"],
                    "iteration": entry["iteration"],
                    "global_step": len(steps),
                    "best_model": entry["best_model"],
                }
            )
    return steps


def incumbent_summary_for_histories(
    histories: Sequence[Sequence[Mapping[str, Any]]],
) -> Dict[str, Any]:
    """The per-cell incumbent record straight from its experiments' histories
    (in experiment order), with the starting set read from the first one."""
    starting = starting_model_set(histories[0]) if histories else frozenset()
    return summarise_incumbents(
        annotate_incumbents(steps_from_histories(histories), starting), starting
    )


def _ordered_by_experiment(
    found: Mapping[int, Any], *, where: str
) -> List[Any]:
    """``found`` values in experiment order, requiring experiments 1..N with no gap."""
    if not found:
        raise FileNotFoundError(f"No experiment*/model_loop/history.json in {where}.")
    ordered = []
    for exp_num in range(1, max(found) + 1):
        if exp_num not in found:
            raise FileNotFoundError(
                f"{where} has experiment{max(found)} but no "
                f"experiment{exp_num}/model_loop/history.json — the cell's "
                "experiments must be contiguous from 1."
            )
        ordered.append(found[exp_num])
    return ordered


def histories_from_run_tree(run_root: Path) -> List[List[Dict[str, Any]]]:
    """Every ``experiment<n>/model_loop/history.json`` under ``run_root``, for
    n = 1..N contiguous, parsed and in numeric order (``experiment10`` after
    ``experiment9``, not after ``experiment1``)."""
    run_root = Path(run_root)
    found: Dict[int, Path] = {}
    for path in run_root.glob("experiment*/model_loop/history.json"):
        match = re.fullmatch(r"experiment(\d+)", path.parent.parent.name)
        if match:
            found[int(match.group(1))] = path
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in _ordered_by_experiment(found, where=str(run_root))
    ]


def histories_from_archive(tar_path: Path) -> List[List[Dict[str, Any]]]:
    """The same, read from a finished cell's ``agent_runs.tar.gz`` (the array
    task archives the agent's ``_runs/<gt>/`` tree and deletes the copy).

    Only members at ``<run>/experiment<n>/model_loop/history.json`` count — a
    candidate's own scratch copy deeper in the tree does not — and every such
    member must share one ``<run>`` prefix: an archive holding several runs is
    ambiguous and raises. Read in memory; nothing is extracted to disk.
    """
    tar_path = Path(tar_path)
    found: Dict[int, str] = {}
    runs = set()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            match = _HISTORY_MEMBER_RE.match(member.name)
            if not match or not member.isfile():
                continue
            runs.add(match.group("run"))
            found[int(match.group("num"))] = member.name
        if len(runs) > 1:
            raise ValueError(
                f"{tar_path} holds history.json files for more than one run "
                f"({sorted(runs)}); a cell archive must hold exactly one."
            )
        names = _ordered_by_experiment(found, where=str(tar_path))
        histories = []
        for name in names:
            handle = tar.extractfile(name)
            if handle is None:
                raise FileNotFoundError(f"Could not read {name} from {tar_path}.")
            histories.append(json.loads(handle.read().decode("utf-8")))
    return histories
