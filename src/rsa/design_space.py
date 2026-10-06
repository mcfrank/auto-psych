"""The space of reference-game contexts: object x feature matrices up to relabeling.

Two matrices are the same game when one is the other with its objects
(rows) and features (columns) reordered, so the space is enumerated as one
canonical representative per equivalence class. A valid game has every
feature true of at least one object (a word true of nothing is never said)
and no two features with the same extension (two words with one meaning are
one word). Objects may repeat: the pragmods "twins" game has two identical
objects.

`context_pool` crosses the games with every word (and optionally a prior
query), the stimulus pool the design and the novelty gate draw from.
"""

from __future__ import annotations

import itertools
from functools import lru_cache
from typing import List, Optional, Sequence, Tuple

from src.rsa.context import Context

Matrix = Tuple[Tuple[int, ...], ...]


def _canonical(matrix: Matrix) -> Matrix:
    """Lexicographically smallest relabeling: permute columns, then sort rows."""
    n_feat = len(matrix[0])
    best = None
    for perm in itertools.permutations(range(n_feat)):
        rows = tuple(sorted(tuple(row[p] for p in perm) for row in matrix))
        if best is None or rows < best:
            best = rows
    return best


def _valid(matrix: Matrix) -> bool:
    columns = list(zip(*matrix))
    return all(any(col) for col in columns) and len(set(columns)) == len(columns)


@lru_cache(maxsize=None)
def games(n_objects: int, n_features: int) -> Tuple[Matrix, ...]:
    """Canonical representatives of every valid n_objects x n_features game."""
    rows = list(itertools.product((0, 1), repeat=n_features))
    seen = set()
    for combo in itertools.combinations_with_replacement(rows, n_objects):
        if _valid(combo):
            seen.add(_canonical(combo))
    return tuple(sorted(seen))


def feature_names(n_features: int) -> Tuple[str, ...]:
    return tuple(f"f{i}" for i in range(n_features))


def context_pool(
    sizes: Sequence[Tuple[int, int]],
    *,
    include_prior_queries: bool = False,
    names: Optional[Sequence[str]] = None,
) -> List[Context]:
    """Every (game, word) context for the given (n_objects, n_features) sizes."""
    pool: List[Context] = []
    for n_obj, n_feat in sizes:
        fnames = tuple(names[:n_feat]) if names is not None else feature_names(n_feat)
        for matrix in games(n_obj, n_feat):
            for word in range(n_feat):
                pool.append(Context(objects=matrix, feature_names=fnames, utterance=word))
            if include_prior_queries:
                pool.append(Context(objects=matrix, feature_names=fnames, utterance=None))
    return pool
