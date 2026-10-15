"""Frozen predictions: each model's posterior-mean choice-class probabilities on
the design pool (`src.rsa.loop.novelty.plain_pool`), as `src.rsa.prereg.pool_fit`
writes them (and `data/rsa/live/finals_pool_fit.py` before it).

A predictions file is fixed before the experiment's data exist: the analysis
reads probabilities from it and fits nothing, so what is tested is exactly what
was committed (its sha256 is in the pre-registration).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from src.rsa.context import Context, model_key
from src.rsa.loop.novelty import plain_pool, pool_digest

WIDTH = 4  # the most objects a pool display has


@dataclass
class PoolPredictions:
    names: List[str]
    probs: np.ndarray  # (models, displays, WIDTH): a class's probability at its first object, 0 elsewhere
    valid: np.ndarray  # (displays, WIDTH): the class slots
    pool: List[Context]
    sha256: Dict[str, str]  # each file's sha256

    def __post_init__(self) -> None:
        self._index = {model_key(ctx): i for i, ctx in enumerate(self.pool)}
        if len(self._index) != len(self.pool):
            raise ValueError("the pool has two displays a model cannot tell apart")

    def display(self, ctx: Context) -> int:
        """The pool index of a trial's display (raises when it is not a pool display)."""
        try:
            return self._index[model_key(ctx)]
        except KeyError:
            raise ValueError(f"display {ctx.objects} (word {ctx.utterance}) is not in the design pool") from None

    def model(self, name: str) -> np.ndarray:
        return self.probs[self.names.index(name)]


def load_predictions(paths: Sequence[Path]) -> PoolPredictions:
    """Every model in ``paths`` (names must not repeat), checked against this code's pool."""
    pool = plain_pool()
    digest = pool_digest(pool)
    valid = np.zeros((len(pool), WIDTH), dtype=bool)
    for i, ctx in enumerate(pool):
        for j, cls in enumerate(ctx.choice_classes()):
            valid[i, j] = cls == j
    names: List[str] = []
    rows: List[np.ndarray] = []
    shas = {}
    for path in paths:
        blob = Path(path).read_bytes()
        shas[str(path)] = hashlib.sha256(blob).hexdigest()
        doc = json.loads(blob)
        if doc["pool"]["digest"] != digest or doc["pool"]["size"] != len(pool):
            raise ValueError(f"{path} was written for another pool ({doc['pool']['digest']}, this code's is {digest})")
        n_entries = doc["pool"]["n_entries"]
        if n_entries != [len(c.objects) for c in pool]:
            raise ValueError(f"{path}: entries per display do not match the pool")
        bounds = np.cumsum([0] + n_entries)
        for name, model in doc["models"].items():
            if name in names:
                raise ValueError(f"{name} is in two predictions files")
            flat = np.asarray(model["probs"], dtype=np.float64)
            if flat.shape != (bounds[-1],):
                raise ValueError(f"{path}: {name} has {flat.shape[0]} entries, the pool {bounds[-1]}")
            p = np.zeros((len(pool), WIDTH))
            for i in range(len(pool)):
                p[i, : n_entries[i]] = flat[bounds[i]: bounds[i + 1]]
            if np.any(p[~valid] != 0) or not np.allclose(p.sum(-1), 1.0, atol=1e-4) or np.any(p < 0):
                raise ValueError(f"{path}: {name}'s probabilities are not class probabilities on every display")
            names.append(name)
            rows.append(p / p.sum(-1, keepdims=True))  # the file rounds to 6 decimals
    if not names:
        raise ValueError(f"no models in {list(map(str, paths))}")
    return PoolPredictions(names=names, probs=np.stack(rows), valid=valid, pool=pool, sha256=shas)
