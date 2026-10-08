"""Joint EIG for reference-game designs, and the power it implies (no fitting)."""

import numpy as np
import pytest

from src.rsa.design.eig import power, select

VALID3 = np.array([[True, True, True]])


def models_on(displays, n_draws=40, seed=0, spread=0.0):
    """Class probabilities (draws, displays, 3) around the given per-display means."""
    rng = np.random.default_rng(seed)
    out = []
    for means in displays:
        base = np.asarray(means, float)
        draws = base[None] + spread * rng.normal(size=(n_draws,) + base.shape)
        draws = np.clip(draws, 1e-3, None)
        out.append(draws / draws.sum(-1, keepdims=True))
    return out


def test_the_display_where_models_disagree_is_picked_first_and_identifies_them():
    agree = [0.5, 0.3, 0.2]
    a = [agree, [0.9, 0.05, 0.05], agree]
    b = [agree, [0.05, 0.9, 0.05], agree]
    probs = models_on([a, b])
    valid = np.ones((3, 3), dtype=bool)
    sel = select(probs, valid, 2, n_responses=20, n_scenarios=400, seed=1)
    assert sel.indices[0] == 1 and sel.sources[0] == "eig"
    assert sel.joint_eig_bits[0] == pytest.approx(1.0, abs=0.02)
    # Once identified, more displays are noise: the rest is single-response fill.
    assert sel.sources[1] == "eig_single_response_fill"


def test_identical_models_give_no_information():
    same = [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]]
    probs = models_on([same, same], spread=0.0)
    sel = select(probs, np.ones((2, 3), dtype=bool), 2, n_responses=50, n_scenarios=600, seed=3)
    assert all(src == "eig_single_response_fill" for src in sel.sources)
    rows = power(probs, np.ones((2, 3), dtype=bool), [0, 1], [50], n_scenarios=1000, seed=4)
    assert rows[0]["joint_eig_bits"] < 0.05 and abs(rows[0]["p_correct"] - 0.5) < 0.07


def test_power_grows_with_the_number_of_participants():
    a = [[0.50, 0.30, 0.20]]
    b = [[0.42, 0.36, 0.22]]  # close: hard to tell apart with few responses
    probs = models_on([a, b], spread=0.005)
    rows = power(probs, VALID3, [0], [10, 100, 1000], n_scenarios=1500, seed=5, names=["a", "b"])
    p = [r["p_correct"] for r in rows]
    assert p[0] < p[1] < p[2] and p[2] > 0.9
    assert set(rows[2]["by_model"]) == {"a", "b"}


def test_a_class_slot_outside_the_layout_is_refused():
    probs = models_on([[[0.5, 0.5, 0.0]], [[0.4, 0.6, 0.0]]])
    probs[0][:, 0, 2] = 0.1
    with pytest.raises(ValueError):
        select(probs, np.array([[True, True, False]]), 1, n_responses=5, n_scenarios=50)
