"""Figure 3's held-out predictions (scripts/paper/fig3_heldout_conditions.py): Figure 2's split per condition."""

import numpy as np
import pytest

from scripts.paper import fig3_heldout_conditions as H


def test_a_trials_leave_one_out_p_left_is_its_loo_density_of_the_observed_choice():
    loo_i = np.log([0.8, 0.3])  # P_loo(observed choice)
    p = H.loo_p_left(loo_i, np.array([1, 0]))
    assert p == pytest.approx([0.8, 0.7])  # chose left: 0.8; chose right with 0.3, so left 0.7


def test_leave_one_out_p_left_needs_binary_choices():
    with pytest.raises(ValueError, match="0 or 1"):
        H.loo_p_left(np.log([0.5]), np.array([2]))


def test_pointwise_values_are_summed_per_condition():
    sums = H.cell_sums([("a", 1), ("b", 2), ("a", 1)], np.array([-0.5, -1.0, -0.25]))
    assert sums == {("a", 1): -0.75, ("b", 2): -1.0}


def test_a_subjective_randomness_trials_condition_is_its_unordered_pair():
    rows = [{"sequence_a": "HT", "sequence_b": "HH"}, {"sequence_a": "HH", "sequence_b": "HT"}]
    assert H.sr_key(rows[0]) == H.sr_key(rows[1]) == ("HH", "HT")


def test_in_sample_p_left_is_the_mean_over_draws_of_each_trials_likelihood_of_left():
    log_lik = np.log([[0.9, 0.2], [0.7, 0.4]])  # (draws, trials); trial 0 chose left, trial 1 right
    assert H.in_sample_p_left(log_lik, np.array([1, 0])) == pytest.approx([0.8, 0.7])
