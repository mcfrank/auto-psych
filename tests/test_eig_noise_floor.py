"""Greedy joint-EIG selection stops when a pick's gain is Monte Carlo noise.

With 40 responses per stimulus a handful of picks already identifies the
model; after that, every candidate's gain is within noise and the greedy
choice is arbitrary (a random set came within ~0.03 bits of the greedy one).
The selection stops once the best candidate's gain is below twice the Monte
Carlo standard error of that gain across scenarios; the design fills the
remaining slots with single-response EIG, conditioned on the picks so far
(user decision 2026-09-26).
"""

from __future__ import annotations

import numpy as np

from src.models.eig_selection import select_n_joint_eig


def _one_decisive_stimulus():
    """Stimulus 0 separates the two models completely; the rest say nothing."""
    p = {"a": np.full((5, 6), 0.5), "b": np.full((5, 6), 0.5)}
    p["a"][:, 0], p["b"][:, 0] = 0.05, 0.95
    return p


def test_selection_stops_when_further_picks_are_noise():
    sel = select_n_joint_eig(
        _one_decisive_stimulus(),
        4,
        n_scenarios=2000,
        seed=1,
        n_responses=40,
        stop_below_noise=True,
    )
    assert sel.indices == [0]
    assert sel.stopped_at_noise_floor


def test_without_the_floor_it_fills_every_slot():
    sel = select_n_joint_eig(
        _one_decisive_stimulus(),
        4,
        n_scenarios=500,
        seed=1,
        n_responses=40,
    )
    assert len(sel.indices) == 4 and not sel.stopped_at_noise_floor


def test_a_fill_continues_from_the_picks_already_made():
    rng = np.random.default_rng(0)
    p = {m: rng.uniform(0.05, 0.95, size=(20, 30)) for m in ("a", "b", "c")}
    first = select_n_joint_eig(p, 2, n_scenarios=300, seed=3, n_responses=40)
    fill = select_n_joint_eig(p, 5, n_scenarios=300, seed=4, preselected=first.indices)
    assert len(fill.indices) == 5
    assert not set(fill.indices) & set(first.indices)
