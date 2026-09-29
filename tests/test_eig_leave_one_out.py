"""The joint-EIG estimator leaves a scenario's generating draw out of its own
model's likelihood average (first audit, C5).

A scenario is a model, one of its parameter draws and responses generated from
that draw. The model posterior of the scenario averages each model's
likelihood over its draws. When the average of the scenario's own model
includes the generating draw, that draw explains its own responses better than
any independent draw would: the true model is rewarded for *memorising* the
draw, the posterior is overconfident and the joint EIG (and the noise-floor
stop, and the recorded ``joint_eig_bits``) too high. With the draw left out,
the average is independent of the scenario.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models import eig_selection
from src.models.eig_selection import estimate_joint_eig, select_n_joint_eig


def _identical_models(n_draws: int, n_stim: int, seed: int = 0):
    """Two models with the same parameter distribution, each represented by
    its own independent draws: there is nothing to learn about which is true."""
    rng = np.random.default_rng(seed)
    return {
        m: np.clip(0.5 + 0.1 * rng.standard_normal((n_draws, n_stim)), 0.01, 0.99)
        for m in ("a", "b")
    }


def _true_model_posterior(p, cols, *, leave_one_out, n_responses=40, n_scenarios=4000):
    """Mean posterior probability of each scenario's generating model."""
    prior = np.full(len(p), 1 / len(p))
    state = eig_selection._ScenarioState(
        p, prior, n_scenarios, np.random.default_rng(1), n_responses, leave_one_out
    )
    for j in cols:
        state.observe(j)
    posterior = state.posterior()
    return float(posterior[np.arange(n_scenarios), state.m_idx].mean())


def test_identical_models_do_not_reward_the_generating_model():
    """Identical models: the posterior of the model that generated a scenario
    must stay at its prior on average, however well the responses pin down
    the draw. Including it, the true model's average contains the one draw that
    fits the responses best, so the posterior 'recognises' it."""
    p = _identical_models(n_draws=20, n_stim=8)
    with_draw = _true_model_posterior(p, range(8), leave_one_out=False)
    without_draw = _true_model_posterior(p, range(8), leave_one_out=True)
    assert with_draw > 0.75
    assert without_draw == pytest.approx(0.5, abs=0.03)


def test_with_few_draws_per_model_memorising_the_draw_is_not_rewarded():
    """The extreme case: two draws per model, responses that identify a draw.
    With the draw included, the posterior names the true model almost every time; left
    out, which model's remaining draw lies nearer is a coin flip."""
    rng = np.random.default_rng(3)
    p = {m: rng.choice([0.1, 0.9], size=(2, 12)) for m in ("a", "b")}
    with_draw = _true_model_posterior(p, range(12), leave_one_out=False)
    without_draw = _true_model_posterior(p, range(12), leave_one_out=True)
    assert with_draw > 0.95
    assert without_draw < 0.6


def test_identical_models_carry_less_spurious_eig_left_out():
    """The joint EIG of a design that cannot discriminate identical models is
    0. Neither estimate reaches it once the design identifies draws (a finite
    average over draws is noisy either way), but the one including the draw
    is inflated beyond the other; on a design that does not identify the
    draws both are ~0."""
    p = _identical_models(n_draws=200, n_stim=8)
    kwargs = dict(n_scenarios=4000, seed=1, n_responses=40)
    with_draw = estimate_joint_eig(p, range(8), leave_one_out=False, **kwargs)
    without_draw = estimate_joint_eig(p, range(8), leave_one_out=True, **kwargs)
    assert without_draw < 0.8 * with_draw
    weak = dict(n_scenarios=4000, seed=1, n_responses=1)
    assert estimate_joint_eig(p, range(4), leave_one_out=True, **weak) < 0.001


def test_the_generating_draw_is_the_only_draw_left_out():
    """The posterior without the draw is the posterior (all draws averaged) of
    the model set with the scenario's generating draw deleted from its model."""
    rng = np.random.default_rng(5)
    p = {m: rng.uniform(0.1, 0.9, size=(6, 5)) for m in ("a", "b", "c")}
    prior = np.full(3, 1 / 3)
    state = eig_selection._ScenarioState(p, prior, 50, np.random.default_rng(2), 7, True)
    for j in range(5):
        state.observe(j)
    posterior = state.posterior()
    for t in range(50):
        own = state.names[state.m_idx[t]]
        averages = []
        for name in state.names:
            log_l = state.logL[name][t]
            if name == own:
                log_l = np.delete(log_l, state.d_idx[t])
            averages.append(np.exp(log_l).mean())
        expected = np.array(averages) / np.sum(averages)
        np.testing.assert_allclose(posterior[t], expected, rtol=1e-9)


def test_selection_scores_candidates_with_the_generating_draw_left_out():
    """The candidate scoring (next_entropy) and the posterior after the pick
    agree: the expected next entropy is the average posterior entropy after
    observing the pick, in float32 scoring too."""
    rng = np.random.default_rng(7)
    p = {m: rng.uniform(0.05, 0.95, size=(30, 40)) for m in ("a", "b")}
    for dtype in ("float64", "float32"):
        state = eig_selection._ScenarioState(p, np.full(2, 0.5), 300, np.random.default_rng(9), 5)
        state.observe(3)
        h_now = state.posterior_entropy()
        scored = state.next_entropy(np.array([11]), np.dtype(dtype))[:, 0]
        # Averaging the scored next entropy over many response draws is what
        # observing and re-computing would give on average.
        draws = []
        for seed in range(200):
            trial = eig_selection._ScenarioState(p, np.full(2, 0.5), 300, np.random.default_rng(9), 5)
            trial.observe(3)
            trial.rng = np.random.default_rng(1000 + seed)
            trial.observe(11)
            draws.append(trial.posterior_entropy())
        assert np.mean(draws) == pytest.approx(scored.mean(), abs=0.01)
        assert h_now.mean() > scored.mean()


def test_a_model_with_a_single_draw_is_refused():
    """With one draw there is nothing left to average once the generating draw
    is left out; a point model is given as two identical draws."""
    p = {"a": np.array([[0.8]]), "b": np.array([[0.2], [0.3]])}
    with pytest.raises(ValueError, match=r"\['a'\] have a single draw"):
        estimate_joint_eig(p, [0])
    with pytest.raises(ValueError, match="single draw"):
        select_n_joint_eig(p, 1)
    assert estimate_joint_eig(p, [0], leave_one_out=False) > 0


def test_leaving_the_draw_out_is_the_default():
    p = _identical_models(n_draws=20, n_stim=8)
    kwargs = dict(n_scenarios=500, seed=1, n_responses=40)
    assert estimate_joint_eig(p, range(8), **kwargs) == estimate_joint_eig(
        p, range(8), leave_one_out=True, **kwargs
    )
    assert select_n_joint_eig(p, 3, **kwargs).joint_eig_bits == select_n_joint_eig(
        p, 3, leave_one_out=True, **kwargs
    ).joint_eig_bits
