"""Tests for joint-EIG stimulus-set selection (src/models/eig_selection.py).

``select_n_joint_eig`` greedily picks the N stimuli maximizing the *joint*
EIG I(M; R_1..R_N) about model identity, evaluated with per-draw prior
predictive ``p_left`` (so parameter-induced correlation between stimuli is
respected, unlike a mean-based independence approximation).

The unit tests pin the estimator to exactly computable cases: with per-draw
probabilities over a small discrete space, I(M; R_S) can be enumerated by hand,
so the Monte Carlo estimate must land near the closed-form value.
"""

from __future__ import annotations


import numpy as np
import pytest

from src.models import pymc_inference as pi
from src.models.eig_selection import estimate_joint_eig, select_n_joint_eig
from src.models.model_loading import clear_model_cache
from tests.paths import PYMC_MODEL_FIXTURES_DIR


def test_estimate_joint_eig_matches_closed_form_single_stimulus():
    # No parameter uncertainty: A predicts 0.8, B predicts 0.2 (two identical
    # draws each, since a scenario's likelihood average leaves its generating
    # draw out). Exact I(M; R) = 1 - H_b(0.8) = 0.278072 bits.
    draws = {"A": np.array([[0.8], [0.8]]), "B": np.array([[0.2], [0.2]])}
    est = estimate_joint_eig(draws, [0], n_scenarios=20000, seed=1)
    assert est == pytest.approx(0.278072, abs=0.02)


def test_estimate_joint_eig_correlated_pair_beats_zero_marginals():
    # Model A's parameter draw makes BOTH stimuli lean the same way (p = 0.9 on
    # one draw, 0.1 on the other); model B predicts 0.5 always. Each stimulus
    # alone carries exactly zero information about M (marginal p_left is 0.5
    # under both models), but the PAIR does: under A the two responses agree
    # 82% of the time, under B 50%. Exact enumeration gives I = 0.0848 bits.
    # Only per-draw likelihoods can see this; a mean-based estimator returns 0.
    # Each parameter value is 200 draws, so leaving a scenario's generating
    # draw out of its average moves the posterior by well under the tolerance.
    draws = {
        "A": np.repeat(np.array([[0.9, 0.9], [0.1, 0.1]]), 200, axis=0),
        "B": np.repeat(np.array([[0.5, 0.5], [0.5, 0.5]]), 200, axis=0),
    }
    marginal = estimate_joint_eig(draws, [0], n_scenarios=5000, seed=2)
    # Not exactly 0: without its generating draw, A's average leans 199:200
    # away from the value that generated the response.
    assert marginal == pytest.approx(0.0, abs=1e-5)
    joint = estimate_joint_eig(draws, [0, 1], n_scenarios=20000, seed=2)
    assert joint == pytest.approx(0.0848, abs=0.02)


def test_greedy_prefers_independent_probe_over_correlated_duplicate():
    # Stimuli 0 and 1 are identical columns (perfectly correlated within each
    # model through the parameter draw); stimulus 2 is an independent, weaker
    # probe. Marginals: I(0) = I(1) = 0.268 > I(2) = 0.240, so a top-k-marginal
    # picker takes {0, 1}. But jointly (exact enumeration):
    #   I({0, 1}) = 0.383  <  I({0, 2}) = 0.428
    # so per-draw greedy must pick the duplicate's complement, not its clone.
    draws = {
        "A": np.array([[0.99, 0.99, 0.78], [0.60, 0.60, 0.78]]),
        "B": np.array([[0.01, 0.01, 0.22], [0.40, 0.40, 0.22]]),
    }
    result = select_n_joint_eig(draws, 2, n_scenarios=20000, seed=4)
    # Stimuli 0 and 1 are exact clones, so either may be picked first.
    assert set(result.indices) in ({0, 2}, {1, 2})


def test_lazy_greedy_tracks_exact_greedy():
    # Joint EIG with per-draw likelihoods is not submodular (synergy exists —
    # see the correlated-pair test), so lazy selection's stale stored gains may
    # legitimately diverge from exact greedy between full passes. The contract
    # is that lazy matches the first pick (always a full pass) and lands near
    # exact greedy's achieved joint EIG, not that the index sets are identical.
    rng = np.random.default_rng(8)
    draws = {
        name: rng.uniform(0.05, 0.95, size=(4, 30)) for name in ("A", "B", "C")
    }
    kwargs = dict(n_scenarios=4000, seed=13)
    lazy = select_n_joint_eig(
        draws, 5, lazy=True, lazy_batch_size=4, refresh_every=3, **kwargs
    )
    full = select_n_joint_eig(draws, 5, lazy=False, **kwargs)
    assert lazy.indices[0] == full.indices[0]
    eig_lazy = estimate_joint_eig(draws, lazy.indices, n_scenarios=20000, seed=555)
    eig_full = estimate_joint_eig(draws, full.indices, n_scenarios=20000, seed=555)
    assert eig_lazy >= eig_full - 0.05


def test_joint_eig_bounded_by_model_prior_entropy():
    rng = np.random.default_rng(21)
    draws = {name: rng.uniform(0.05, 0.95, size=(3, 20)) for name in ("A", "B")}
    result = select_n_joint_eig(
        draws, 6, model_weights={"A": 9.0, "B": 1.0}, n_scenarios=4000, seed=3
    )
    h_m = -(0.9 * np.log2(0.9) + 0.1 * np.log2(0.1))  # 0.469 bits
    assert all(v <= h_m + 0.02 for v in result.joint_eig_bits)


def test_select_n_joint_eig_validates_inputs():
    good = {"A": np.array([[0.8, 0.3]]), "B": np.array([[0.2, 0.6]])}
    with pytest.raises(ValueError):
        select_n_joint_eig(good, 0, n_scenarios=100, seed=0)
    with pytest.raises(ValueError):
        select_n_joint_eig(good, 3, n_scenarios=100, seed=0)  # > n_stim
    with pytest.raises(ValueError):
        select_n_joint_eig(good, 1, n_scenarios=0, seed=0)
    with pytest.raises(ValueError):
        select_n_joint_eig({}, 1, n_scenarios=100, seed=0)
    with pytest.raises(ValueError):
        select_n_joint_eig(
            {"A": np.array([[0.8]]), "B": np.array([[0.2, 0.6]])}, 1,
            n_scenarios=100, seed=0,
        )  # mismatched n_stim
    with pytest.raises(ValueError):
        select_n_joint_eig(
            {"A": np.array([[1.4]]), "B": np.array([[0.2]])}, 1,
            n_scenarios=100, seed=0,
        )  # p outside [0, 1]


@pytest.mark.slow
def _random_sequence(rng, length):
    """Generate a random H/T sequence of the given length."""
    return "".join(rng.choice(["H", "T"], size=length))


def test_select_n_joint_eig_end_to_end_on_pymc_models():
    """Full path: batched per-draw prior predictive -> greedy joint selection."""
    rng = np.random.default_rng(3)
    rows = []
    for _ in range(12):
        n_a, n_b = int(rng.integers(4, 9)), int(rng.integers(4, 9))
        rows.append(
            {"sequence_a": _random_sequence(rng, n_a),
             "sequence_b": _random_sequence(rng, n_b),
             "chose_left": 0}
        )
    names = ["bayesian_fair_coin", "representativeness"]
    clear_model_cache()
    draws = pi.prior_predict_p_left_draws(
        names, PYMC_MODEL_FIXTURES_DIR, rows, n_samples=50, seed=5
    )
    assert set(draws.keys()) == set(names)
    for name in names:
        assert draws[name].shape == (50, len(rows))

    result = select_n_joint_eig(draws, 4, n_scenarios=2000, seed=11)

    assert len(result.indices) == 4
    assert len(set(result.indices)) == 4
    assert all(0 <= i < len(rows) for i in result.indices)
    # Joint EIG over 2 models is bounded by H(M) = 1 bit; the in-sample
    # trajectory should grow (up to MC noise) as stimuli are added.
    assert len(result.joint_eig_bits) == 4
    for prev, cur in zip([0.0, *result.joint_eig_bits], result.joint_eig_bits):
        assert cur >= prev - 0.02
        assert cur <= 1.0 + 1e-9
    # A fresh-scenario estimate of the chosen set is positive: the fixture
    # models genuinely disagree somewhere in this pool.
    fresh = estimate_joint_eig(draws, result.indices, n_scenarios=2000, seed=99)
    assert fresh > 0.0

    # Deterministic given the seed.
    again = select_n_joint_eig(draws, 4, n_scenarios=2000, seed=11)
    assert again.indices == result.indices
