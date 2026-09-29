"""Lazy batched greedy joint-EIG selection, float32 scoring and threaded scoring.

An exact greedy pick scores every candidate of the pool (43,434 pairs in the
design); lazy selection re-scores only the candidates with the highest stored
gains between exact full passes. It is exact when gains only shrink, and an
approximation otherwise (joint EIG is not submodular), so these tests pin what
must hold exactly: with batches that cover the pool, or a full pass at every
pick, it *is* exact greedy; on a pool with diminishing returns it picks what
exact greedy picks while scoring fewer candidates; the noise-floor stop
still fires. Scoring precision and thread count are knobs of the same search.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models import eig_selection
from src.models.eig_selection import (
    estimate_joint_eig,
    scenario_posterior_entropies,
    select_n_joint_eig,
)


def _random_pool(n_stim: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {m: rng.uniform(0.05, 0.95, size=(20, n_stim)) for m in ("a", "b", "c")}


def _graded_pool(n_stim: int = 2000):
    """Two models with no parameter uncertainty that disagree by a different,
    well-separated margin on every stimulus: responses are independent given
    the model, so gains only shrink as picks accumulate (diminishing returns),
    and the best remaining stimulus is always the most discriminating one.
    Each is two identical draws: a scenario's likelihood average leaves out
    the draw that generated it, and the copy keeps the point model exact."""
    margin = np.linspace(0.0, 0.4, n_stim)
    order = np.random.default_rng(1).permutation(n_stim)  # best ones scattered
    return {
        "a": np.repeat((0.5 + margin[order])[None, :], 2, axis=0),
        "b": np.repeat((0.5 - margin[order])[None, :], 2, axis=0),
    }


def _count_scored_columns(monkeypatch) -> list:
    scored: list = []
    original = eig_selection._ScenarioState.next_entropy

    def counting(self, cols, dtype=np.dtype(np.float64)):
        scored.append(len(cols))
        return original(self, cols, dtype)

    monkeypatch.setattr(eig_selection._ScenarioState, "next_entropy", counting)
    return scored


def test_lazy_with_batches_covering_the_pool_is_exact_greedy():
    p = _random_pool()
    kwargs = dict(n_scenarios=300, seed=5, n_responses=3)
    exact = select_n_joint_eig(p, 12, **kwargs)
    lazy = select_n_joint_eig(p, 12, lazy=True, lazy_batch_size=300, refresh_every=7, **kwargs)
    assert lazy.indices == exact.indices
    assert lazy.joint_eig_bits == exact.joint_eig_bits


def test_lazy_with_a_full_pass_at_every_pick_is_exact_greedy():
    p = _random_pool()
    kwargs = dict(n_scenarios=300, seed=6, n_responses=3)
    exact = select_n_joint_eig(p, 12, **kwargs)
    lazy = select_n_joint_eig(p, 12, lazy=True, lazy_batch_size=8, refresh_every=1, **kwargs)
    assert lazy.indices == exact.indices
    assert lazy.joint_eig_bits == exact.joint_eig_bits


def test_lazy_matches_exact_greedy_under_diminishing_returns_with_far_less_scoring(
    monkeypatch,
):
    p = _graded_pool()
    kwargs = dict(n_scenarios=400, seed=2, n_responses=1)
    scored = _count_scored_columns(monkeypatch)
    exact = select_n_joint_eig(p, 20, **kwargs)
    exact_columns = sum(scored)
    scored.clear()
    lazy = select_n_joint_eig(p, 20, lazy=True, lazy_batch_size=64, refresh_every=8, **kwargs)
    lazy_columns = sum(scored)

    assert lazy.indices == exact.indices
    np.testing.assert_allclose(lazy.joint_eig_bits, exact.joint_eig_bits, atol=1e-12)
    # Every pick shrinks every gain here, so a lazy pick re-scores batches
    # until the fresh best clears the stale gains below it; still far less
    # than a full pass per pick.
    assert exact_columns >= 20 * 1990
    assert lazy_columns < exact_columns / 1.5


def test_the_noise_floor_stop_fires_in_lazy_selection():
    p = {"a": np.full((5, 1000), 0.5), "b": np.full((5, 1000), 0.5)}
    p["a"][:, 0], p["b"][:, 0] = 0.05, 0.95  # the one decisive stimulus
    sel = select_n_joint_eig(
        p, 4, n_scenarios=2000, seed=1, n_responses=40, stop_below_noise=True,
        lazy=True, lazy_batch_size=64, refresh_every=16,
    )
    assert sel.indices == [0]
    assert sel.stopped_at_noise_floor


def test_the_noise_floor_is_judged_on_the_lazily_chosen_pick():
    """Past the first pick the noise test sees a lazily chosen stimulus: it
    must be the same stimulus, and the same verdict, as exact greedy's."""
    p = _graded_pool(1000)
    kwargs = dict(n_scenarios=500, seed=3, n_responses=40, stop_below_noise=True)
    exact = select_n_joint_eig(p, 30, **kwargs)
    lazy = select_n_joint_eig(p, 30, lazy=True, lazy_batch_size=64, refresh_every=16, **kwargs)
    assert exact.stopped_at_noise_floor and lazy.stopped_at_noise_floor
    assert lazy.indices == exact.indices


def test_float32_scoring_picks_what_float64_picks():
    p = _graded_pool(500)
    kwargs = dict(n_scenarios=400, seed=4, n_responses=1)
    double = select_n_joint_eig(p, 6, **kwargs)
    single = select_n_joint_eig(p, 6, dtype="float32", **kwargs)
    assert single.indices == double.indices
    # The trajectory is the float64 scenario posterior, whatever the scoring dtype.
    assert single.joint_eig_bits == double.joint_eig_bits


def test_float32_gains_agree_with_float64_on_forty_responses():
    p = _random_pool(200, seed=9)
    state = eig_selection._ScenarioState(
        eig_selection._validated_p(p)[0], np.full(3, 1 / 3), 500, np.random.default_rng(0), 40
    )
    for j in (3, 50, 120):
        state.observe(j)
    cols = np.arange(200)
    double = state.next_entropy(cols).mean(axis=0)
    single = state.next_entropy(cols, np.dtype(np.float32)).mean(axis=0, dtype=np.float64)
    assert np.abs(single - double).max() < 1e-4
    assert double.std() > 100 * np.abs(single - double).max()


def test_the_thread_count_does_not_change_the_selection():
    p = _random_pool(700, seed=2)
    kwargs = dict(n_scenarios=200, seed=8, n_responses=5, chunk_size=32, lazy=True,
                  lazy_batch_size=64, refresh_every=4)
    one = select_n_joint_eig(p, 9, n_threads=1, **kwargs)
    four = select_n_joint_eig(p, 9, n_threads=4, **kwargs)
    assert four.indices == one.indices
    assert four.joint_eig_bits == one.joint_eig_bits


def test_paired_scenario_entropies_average_to_the_joint_eig():
    p = _random_pool(50, seed=4)
    h = scenario_posterior_entropies(p, [1, 2, 3], n_scenarios=500, seed=7, n_responses=4)
    assert h.shape == (500,)
    eig = estimate_joint_eig(p, [1, 2, 3], n_scenarios=500, seed=7, n_responses=4)
    assert eig == pytest.approx(np.log2(3) - h.mean())


@pytest.mark.parametrize(
    "knob",
    [
        {"lazy_batch_size": 0},
        {"refresh_every": 0},
        {"n_threads": 0},
        {"chunk_size": 0},
        {"dtype": "float16"},
    ],
)
def test_search_knobs_are_validated(knob):
    with pytest.raises(ValueError):
        select_n_joint_eig(_random_pool(10), 2, n_scenarios=10, lazy=True, **knob)
