"""Collecting a loop experiment from the Bigelow & Piantadosi dataset.

A dataset run is a live run whose participants are the dataset's: each
designed (set, target) cell gets ``responses_per_cell`` of the cell's real
responses (9-22 of them, from different people), the next ones in a seeded
order fixed per cell, after those earlier experiments of the run drew, so no
response is used twice; the design may pick only cells with that many left.
Responses keep their participant's id and trial.
"""

from __future__ import annotations

import csv
import hashlib
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

from src.number_game.data import DATA_CSV, load_responses


Cell = Tuple[str, str]  # (number_set "16,8", target "4"), as the responses CSV gives them


@lru_cache(maxsize=1)
def _cells() -> Dict[Cell, List[Dict[str, Any]]]:
    """Every response, by cell, in the file's order."""
    responses = load_responses(DATA_CSV)
    cells: Dict[Cell, List[Dict[str, Any]]] = {}
    for s, target, pid, trial, rating in zip(
        responses["set"], responses["target"], responses["id"], responses["trial"], responses["rating"]
    ):
        number_set = ",".join(map(str, s))
        cells.setdefault((number_set, str(target)), []).append(
            {"number_set": number_set, "target": str(target), "participant_id": int(pid),
             "trial_index": int(trial), "rating": int(rating)}
        )
    return cells


def _key(row: Mapping[str, Any]) -> Cell:
    return (str(row["number_set"]), str(row["target"]))


@dataclass(frozen=True)
class DatasetCollection:
    responses_per_cell: int
    seed: int

    @staticmethod
    def drawn(responses_paths: Sequence[Path]) -> Counter:
        """How many responses each cell got in these response files (a run's
        earlier experiments)."""
        used: Counter = Counter()
        for path in responses_paths:
            with Path(path).open(encoding="utf-8", newline="") as f:
                used.update(_key(row) for row in csv.DictReader(f))
        return used

    def cell_count(self, stimulus: Mapping[str, Any]) -> int:
        return len(_cells()[_key(stimulus)])

    def available(self, stimuli: Sequence[Mapping[str, Any]], used: Counter) -> List[Mapping[str, Any]]:
        """The stimuli whose cells have ``responses_per_cell`` responses left
        after ``used`` (a design pool, for the next experiment)."""
        cells = _cells()
        return [s for s in stimuli if len(cells.get(_key(s), ())) - used[_key(s)] >= self.responses_per_cell]

    def draw(self, stimuli: Sequence[Mapping[str, Any]], used: Counter) -> List[Dict[str, Any]]:
        """The next ``responses_per_cell`` responses of each stimulus's cell."""
        out = []
        for stimulus in stimuli:
            key = _key(stimulus)
            rows = _cells()[key]
            start = used[key]
            if len(rows) - start < self.responses_per_cell:
                raise ValueError(
                    f"cell {key} has {len(rows) - start} responses left, fewer than "
                    f"{self.responses_per_cell}; the design must not pick it"
                )
            digest = hashlib.sha256(f"{self.seed}|{key}".encode()).digest()
            order = np.random.default_rng(int.from_bytes(digest[:8], "big")).permutation(len(rows))
            out += [rows[i] for i in order[start : start + self.responses_per_cell]]
        return out
