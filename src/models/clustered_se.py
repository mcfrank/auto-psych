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

from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

STIMULUS_COLUMNS = ("sequence_a", "sequence_b")


def stimulus_clusters(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """One integer per row; rows showing the same pair share it.

    A stimulus is the unordered pair: left/right is counterbalanced, so both
    presentations of a pair are one stimulus. Ids are numbered in order of
    first appearance. Raises ``KeyError`` naming a missing stimulus column.
    """
    ids: Dict[Tuple[str, str], int] = {}
    out = np.empty(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        for column in STIMULUS_COLUMNS:
            if column not in row:
                raise KeyError(
                    f"responses row {i} lacks the stimulus column {column!r}; "
                    f"stimulus clustering needs {STIMULUS_COLUMNS}."
                )
        key = tuple(sorted((str(row["sequence_a"]), str(row["sequence_b"]))))
        out[i] = ids.setdefault(key, len(ids))
    return out


def cluster_dse(best_i: np.ndarray, other_i: np.ndarray, groups: np.ndarray) -> float:
    """The standard error of ``sum(best_i - other_i)`` with each group
    (stimulus) as one observation: ``sqrt(G · var(sum_g diff_i))``, the
    clustered counterpart of az.compare's ``sqrt(n · var(diff_i))``."""
    diff = np.asarray(best_i, dtype="float64") - np.asarray(other_i, dtype="float64")
    sums = np.bincount(np.asarray(groups), weights=diff)
    return float(np.sqrt(sums.shape[0] * np.var(sums)))
