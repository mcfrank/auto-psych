"""Collecting number-game responses from the Bigelow & Piantadosi dataset.

In a dataset run the loop's experiments are collected from the dataset instead
of from live participants: each designed (set, target) cell gets the next
``responses_per_cell`` of its real responses, in a seeded order fixed per
cell, after those earlier experiments drew; the design may only pick cells
with that many left.
"""

from __future__ import annotations

from collections import Counter

import pytest

from src.number_game.dataset import DATA_CSV, DatasetCollection
from src.tasks import task_for_project

POOL = task_for_project("number_game").design_pool()

needs_data = pytest.mark.skipif(not DATA_CSV.exists(), reason="number-game dataset not present")


def _key(row):
    return (row["number_set"], str(row["target"]))


@needs_data
def test_draws_never_reuse_a_response_and_follow_a_fixed_order():
    source = DatasetCollection(responses_per_cell=9, seed=1)
    pool = source.available(POOL, Counter())
    assert len(pool) == 25_499  # every cell with responses has at least 9 (one has none)
    stimuli = [s for s in pool if source.cell_count(s) >= 18][:40]  # each drawable twice
    first = source.draw(stimuli, Counter())
    assert len(first) == 40 * 9
    assert set(first[0]) == {"number_set", "target", "participant_id", "trial_index", "rating"}
    used = Counter(_key(r) for r in first)
    assert source.available(stimuli, used) == stimuli
    second = source.draw(stimuli, used)
    assert len(second) == 40 * 9
    responses = lambda rows: {(r["participant_id"], r["trial_index"]) for r in rows}  # noqa: E731
    assert not responses(first) & responses(second)
    assert source.draw(stimuli, Counter()) == first  # deterministic
    assert DatasetCollection(responses_per_cell=9, seed=2).draw(stimuli, Counter()) != first


@needs_data
def test_the_pool_holds_only_cells_with_enough_responses_left():
    source = DatasetCollection(responses_per_cell=9, seed=1)
    cell = POOL[0]
    used = Counter({_key(cell): 9})
    left = source.cell_count(cell) - 9
    assert (cell in source.available([cell], used)) == (left >= 9)
    with pytest.raises(ValueError, match="responses left"):
        source.draw([cell], Counter({_key(cell): source.cell_count(cell) - 1}))
