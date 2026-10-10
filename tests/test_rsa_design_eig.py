"""Joint EIG for reference-game designs, and the power it implies (no fitting)."""

import numpy as np
import pytest

from src.rsa.design.eig import Quota, composition, power, select

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


def _kinds_design(seed=0):
    """Displays 0-5 tell the models apart (kind "big"), 6-11 barely do ("small")."""
    rng = np.random.default_rng(seed)
    a, b = [], []
    for d in range(12):
        gap = 0.4 if d < 6 else 0.04 + 0.01 * rng.random()
        a.append([0.45 + gap / 2, 0.35 - gap / 2, 0.2])
        b.append([0.45 - gap / 2, 0.35 + gap / 2, 0.2])
    small = np.arange(12) >= 6
    return models_on([a, b], spread=0.05, seed=seed), np.ones((12, 3), dtype=bool), small


def test_quotas_take_the_best_displays_of_each_kind_and_leave_the_rest_free():
    probs, valid, small = _kinds_design()
    free = select(probs, valid, 6, n_responses=5, n_scenarios=400, seed=2)
    assert not any(small[free.indices])  # free EIG never goes to the uninformative kind
    quota = Quota("small", "size", small, 2)
    sel = select(probs, valid, 6, n_responses=5, n_scenarios=400, seed=2, quotas=[quota])
    assert int(small[sel.indices].sum()) == 2 and composition([quota], sel.indices) == {"small": 2}
    # the quota waits until it must: the informative kind comes first
    assert not any(small[sel.indices[:4]])
    assert sel.joint_eig_bits[-1] <= free.joint_eig_bits[-1] + 0.02


def test_quotas_on_crossing_dimensions_are_all_met():
    probs, valid, small = _kinds_design(seed=1)
    odd = np.arange(12) % 2 == 1
    quotas = [Quota("small", "size", small, 2), Quota("big", "size", ~small, 2),
              Quota("odd", "parity", odd, 3), Quota("even", "parity", ~odd, 2)]
    sel = select(probs, valid, 6, n_responses=5, n_scenarios=300, seed=3, quotas=quotas)
    got = composition(quotas, sel.indices)
    assert all(got[q.name] >= q.minimum for q in quotas) and len(set(sel.indices)) == 6


def test_impossible_quotas_are_refused():
    probs, valid, small = _kinds_design()
    with pytest.raises(ValueError, match="pool has 6"):
        select(probs, valid, 8, n_responses=5, n_scenarios=50, quotas=[Quota("small", "size", small, 7)])
    with pytest.raises(ValueError, match="need 7 displays of 6"):
        select(probs, valid, 6, n_responses=5, n_scenarios=50,
               quotas=[Quota("small", "size", small, 4), Quota("big", "size", ~small, 3)])
    with pytest.raises(ValueError, match="overlap"):
        select(probs, valid, 6, n_responses=5, n_scenarios=50,
               quotas=[Quota("small", "size", small, 1), Quota("all", "size", np.ones(12, bool), 1)])
