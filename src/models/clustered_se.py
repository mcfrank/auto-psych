"""The stimulus-clustered standard error of an ELPD-LOO difference.

Every experiment shows the same pairs to all its participants, so the trials
on one pair tend to favour the same model together. az.compare's ``dse``
treats every trial as independent; measured on past runs it is about half the
clustered one (range 1.05-5.5x; see the LOO design effect memo). Summing the
pointwise differences within each stimulus and taking the standard error over
stimuli treats a stimulus as one observation — the nonparametric
(cluster-robust) counterpart of a stimulus random effect.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import numpy as np

def stimulus_clusters(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """One integer per row; rows showing the same stimulus share it.

    A stimulus is the task's ``stimulus_key`` (``src.tasks``; the task is the
    one whose stimulus columns the rows carry): for the coin-flip task the
    unordered pair, since left/right is counterbalanced, for the number game
    the (set, target) cell. Ids are numbered in order of first appearance.
    Raises naming the missing stimulus columns.
    """
    from src.tasks import task_present_in

    if not rows:
        return np.empty(0, dtype=np.int64)
    task = task_present_in(rows[0], "stimulus_columns")
    ids: Dict[Any, int] = {}
    out = np.empty(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        missing = [c for c in task.stimulus_columns if c not in row]
        if missing:
            raise KeyError(
                f"responses row {i} lacks the stimulus column(s) {missing}; "
                f"stimulus clustering needs {task.stimulus_columns}."
            )
        out[i] = ids.setdefault(task.stimulus_key(row), len(ids))
    return out


def cluster_dse(best_i: np.ndarray, other_i: np.ndarray, groups: np.ndarray) -> float:
    """The standard error of ``sum(best_i - other_i)`` with each group
    (stimulus) as one observation: ``sqrt(G · var(sum_g diff_i))``, the
    clustered counterpart of az.compare's ``sqrt(n · var(diff_i))``."""
    diff = np.asarray(best_i, dtype="float64") - np.asarray(other_i, dtype="float64")
    sums = np.bincount(np.asarray(groups), weights=diff)
    return float(np.sqrt(sums.shape[0] * np.var(sums)))
