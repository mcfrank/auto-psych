"""What the pipeline needs to know about a project's task.

A ``TaskSpec`` per project: the raw response columns (all that agents and
models ever see), which of them make up a stimulus and which is the binary
response, the name of the per-trial probability every model reports, the
stimulus identity (for holdout exclusion and clustering), and the stimulus
pools (the design's, the novelty gate's, the recovery evaluation's) and the
simulator. The pools and simulators live with each task's research library
and are imported lazily, so the pipeline's coupling to them stays thin.

Found by project id where the caller knows its project, or by a response
file's columns where only the file is at hand: a response file carries exactly
its task's raw columns, so the columns name the task.

``participant_id`` is the one optional raw column: a run that hides it
(``hide_participant_id`` in a holdout config) writes response files without
it, so every model is population-level.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Hashable, List, Mapping, Sequence, Tuple

import numpy as np

Row = Dict[str, Any]

# Raw columns a response file may leave out (on every row).
OPTIONAL_COLUMNS = frozenset({"participant_id"})


@dataclass(frozen=True)
class TaskSpec:
    name: str
    raw_columns: Tuple[str, ...]
    stimulus_columns: Tuple[str, ...]
    response_column: str
    prediction_var: str
    # The stimulus a row shows, for holdout exclusion and stimulus clustering.
    stimulus_key: Callable[[Mapping[str, Any]], Hashable]
    stimulus_label: Callable[[Mapping[str, Any]], str]
    # One valid raw stimulus row, for checks that a model can bind raw rows.
    probe_row: Mapping[str, str]
    # Every stimulus the design may select.
    design_pool: Callable[[], List[Row]]
    # The novelty gate's fixed stimulus sample (never the evaluation pool).
    novelty_pool: Callable[[], List[Row]]
    # The recovery evaluation's candidate stimuli, before trained ones are removed.
    eval_pool: Callable[[Mapping[str, Any]], List[Row]]
    # Responses from a ground truth: ``simulate(predict, stimuli,
    # n_participants, rng, participant_id_offset)``, where ``predict(rows)``
    # is the ground truth's probability of a positive response on each row.
    simulate: Callable[..., List[Row]]
    # The exploration lenses the inner loop's exploratory slots walk.
    candidate_hints: Tuple[str, ...]
    # The candidate and critique agents' instructions (markdown files).
    theory_prompt: Path
    critique_prompt: Path

    @property
    def required_columns(self) -> Tuple[str, ...]:
        return tuple(c for c in self.raw_columns if c not in OPTIONAL_COLUMNS)

    def columns_of(self, row: Mapping[str, Any]) -> Tuple[str, ...]:
        """The raw columns a response row carries, in the task's order."""
        return tuple(c for c in self.raw_columns if c in row or c not in OPTIONAL_COLUMNS)

    def load_stimuli(self, path) -> List[Row]:
        """A design's ``stimuli.json``, keeping each stimulus's own columns."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Stimuli file must contain a list: {path}")
        missing = [i for i, item in enumerate(data) if not set(self.stimulus_columns) <= set(item)]
        if missing:
            raise ValueError(f"{path}: stimuli {missing[:5]} lack {self.name} columns {self.stimulus_columns}")
        return [{c: item[c] for c in self.stimulus_columns} for item in data]

    def stimulus_rows(self, stimuli: Sequence[Mapping[str, Any]]) -> List[Row]:
        """Raw rows for stimuli, with a placeholder response (models bind it)."""
        return [{**{c: s[c] for c in self.stimulus_columns}, self.response_column: 0} for s in stimuli]


def _tasks() -> Dict[str, TaskSpec]:
    from src.tasks.number_game import NUMBER_GAME
    from src.tasks.subjective_randomness import SUBJECTIVE_RANDOMNESS

    return {t.name: t for t in (SUBJECTIVE_RANDOMNESS, NUMBER_GAME)}


def all_tasks() -> List[TaskSpec]:
    return list(_tasks().values())


def task_for_project(project_id: str) -> TaskSpec:
    tasks = _tasks()
    if project_id not in tasks:
        raise KeyError(f"no task is defined for project {project_id!r}; known: {sorted(tasks)}")
    return tasks[project_id]


def task_for_columns(columns: Sequence[str]) -> TaskSpec:
    """The task whose raw columns are exactly ``columns`` (in any order),
    optional ones included or not."""
    matches = [t for t in _tasks().values() if set(t.columns_of(set(columns))) == set(columns)]
    if len(matches) != 1:
        raise ValueError(f"no task has exactly the columns {sorted(columns)}")
    return matches[0]


def task_of_file(path) -> TaskSpec:
    """The task of a response file: the one whose response column its header
    carries (``chose_left``, ``rating``; a featurised file has more columns)."""
    with Path(path).open(encoding="utf-8") as f:
        header = [c.strip() for c in f.readline().strip().split(",") if c.strip()]
    matches = [t for t in _tasks().values() if t.response_column in header]
    if len(matches) != 1:
        raise ValueError(f"{path}: no one task's response column is among its columns {header}")
    return matches[0]


def task_present_in(columns: Sequence[str], field: str = "raw_columns") -> TaskSpec:
    """The one task whose ``field`` columns (``raw_columns``, the optional ones
    aside, or ``stimulus_columns``) are all among ``columns``; other columns
    may be present too (a design's stimuli carry their EIG, say)."""
    field = "required_columns" if field == "raw_columns" else field
    matches = [t for t in _tasks().values() if set(getattr(t, field)) <= set(columns)]
    if len(matches) != 1:
        raise ValueError(f"no task's {field.replace('_', ' ')} are all among {sorted(columns)}")
    return matches[0]


def bernoulli_simulation(predict, stimuli, n_participants, rng: np.random.Generator, participant_id_offset, task):
    """Every participant answers every stimulus, in the design's order; each
    response is Bernoulli(the ground truth's probability on that stimulus)."""
    rows = task.stimulus_rows(stimuli)
    p = np.asarray(predict(rows), dtype=float)
    out: List[Row] = []
    for participant in range(n_participants):
        positive = rng.random(len(rows)) < p
        for trial_index, stim in enumerate(stimuli):
            out.append({
                **{c: stim[c] for c in task.stimulus_columns},
                "participant_id": participant_id_offset + participant,
                "trial_index": trial_index,
                task.response_column: int(positive[trial_index]),
            })
    return out
