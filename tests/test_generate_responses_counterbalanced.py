"""Simulated participants see each pair in a random left/right order.

The human experiments counterbalance which sequence is shown on the left; the
recovery harness mirrors that when it samples from the ground truth. Each row
records the pair as displayed, and ``chose_left`` is drawn from the ground
truth's ``p_left`` for that displayed order. Participant ids are unique across
experiments: each experiment's ids start at an explicit offset.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.subjective_randomness import holdout_data
from src.subjective_randomness.holdout_data import generate_responses

STIMULI = [
    {"sequence_a": "HT", "sequence_b": "HH"},
    {"sequence_a": "HTTH", "sequence_b": "HHHH"},
]


@pytest.fixture(autouse=True)
def prefers_the_mixed_sequence(monkeypatch):
    """A ground truth that almost always picks the sequence that is not all H."""

    def fake_p_left(model_name, models_dir, stimuli, params, *, seed=0):
        return np.array(
            [0.95 if "T" in s["sequence_a"] else 0.05 for s in stimuli]
        )

    monkeypatch.setattr(holdout_data, "p_left_fixed_params", fake_p_left)


def _generate(**kwargs):
    return generate_responses(
        "gt", None, STIMULI, {}, 200, seed=3, participant_id_offset=0, **kwargs
    )


def test_about_half_of_the_trials_show_the_pair_swapped():
    rows = _generate()
    swapped = [r["sequence_a"] == STIMULI[r["trial_index"]]["sequence_b"] for r in rows]
    assert 0.4 < np.mean(swapped) < 0.6
    for r in rows:
        shown = {r["sequence_a"], r["sequence_b"]}
        assert shown == set(STIMULI[r["trial_index"]].values())


def test_the_choice_follows_the_displayed_order():
    rows = _generate()
    picked_mixed = [
        (r["chose_left"] == 1) == ("T" in r["sequence_a"]) for r in rows
    ]
    assert np.mean(picked_mixed) > 0.9


def test_participant_ids_start_at_the_offset():
    rows = generate_responses(
        "gt", None, STIMULI, {}, 3, seed=3, participant_id_offset=80
    )
    assert sorted({r["participant_id"] for r in rows}) == [80, 81, 82]


def test_the_offset_is_required():
    with pytest.raises(TypeError, match="participant_id_offset"):
        generate_responses("gt", None, STIMULI, {}, 3, seed=3)


def test_the_same_seed_gives_the_same_rows():
    assert _generate() == _generate()
