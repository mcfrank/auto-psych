"""Seeds differ between repeats, ground truths, experiments and purposes.

The response seed used to be ``cell seed + experiment``, so repeat r's
experiment 2 reused repeat r+1's experiment 1 seed, every ground truth in a
repeat shared seeds, and every design used seed 42. Each seed is now derived
from all the identifiers of what it seeds.
"""

from __future__ import annotations

import itertools

from src.subjective_randomness.holdout_recovery import derive_seed


def test_no_two_cells_experiments_or_purposes_share_a_seed():
    seeds = [
        derive_seed(cell_seed, gt, exp, purpose)
        for cell_seed, gt, exp, purpose in itertools.product(
            range(100, 106),
            ["motif_stack", "falk_konold_dp", "finite_experience_occurrence", "local_representativeness"],
            (1, 2, 3),
            ("design", "responses"),
        )
    ]
    assert len(set(seeds)) == len(seeds)


def test_the_old_off_by_one_collision_is_gone():
    assert derive_seed(101, "motif_stack", 2, "responses") != derive_seed(
        102, "motif_stack", 1, "responses"
    )


def test_a_seed_is_reproducible_and_fits_numpy():
    a = derive_seed(101, "motif_stack", 1, "design")
    assert a == derive_seed(101, "motif_stack", 1, "design")
    assert 0 <= a < 2**31
