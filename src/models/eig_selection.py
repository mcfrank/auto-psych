"""Greedy selection of the N-stimulus set with maximal joint EIG.

Objective: choose stimuli S (|S| = N) maximizing I(M; R_S) — the mutual
information between model identity M and the joint response vector — from
per-model, per-draw prior-predictive ``p_left`` arrays. Working per draw
(rather than from each model's mean ``p_left``) keeps the correlation that
shared parameters induce between stimuli *within* a model, so a near-duplicate
of an already-selected stimulus is correctly scored as mostly redundant.

Estimation is by Monte Carlo scenarios. A scenario is one simulated "world":
a model sampled from the model prior, one of its parameter draws, and Bernoulli
responses for the selected stimuli generated from that draw's ``p_left``. Each
scenario tracks per-draw log-likelihoods for every model, giving a posterior
over models via p(r_S | m) = mean over draws of the product likelihood; joint
EIG is H(M) minus the mean posterior entropy across scenarios.

Selection is greedy: at each step add the stimulus with the largest expected
posterior-entropy reduction. Marginal gains for all candidates are computed in
one vectorized pass per step (a matmul over draws per model, chunked over
candidates), re-scored exactly at every step by default. ``lazy=True`` uses
CELF lazy re-evaluation instead — valid when gains only shrink as the set
grows (submodularity), which per-draw likelihoods deliberately break: a
stimulus can become *more* informative after a correlated partner is chosen
(synergy), and CELF's stale ranking can miss exactly those candidates. Lazy
mode is therefore an approximation for very large pools or quick iteration;
its achieved joint EIG tracks exact greedy closely but not identically.

The ``joint_eig_bits`` trajectory is estimated from the same scenarios used
for selection (in-sample); use :func:`estimate_joint_eig` with a fresh seed
for an unbiased estimate of a chosen set.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Numeric floor for probabilities entering logs. The PyMC models already clip
# p_left to [1e-6, 1 - 1e-6]; this only guards hand-built arrays at 0 or 1.
_P_CLIP = 1e-12


def _validated_p(
    p_left_draws: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], int]:
    """Validate per-model (n_draws, n_stim) probability arrays; return n_stim."""
    if not p_left_draws:
        raise ValueError("p_left_draws must be non-empty.")
    n_stim: Optional[int] = None
    out: Dict[str, np.ndarray] = {}
    for name, arr in p_left_draws.items():
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[0] < 1 or arr.shape[1] < 1:
            raise ValueError(
                f"Model {name!r}: expected a (n_draws, n_stim) array, got shape "
                f"{arr.shape}."
            )
        if n_stim is None:
            n_stim = arr.shape[1]
        elif arr.shape[1] != n_stim:
            raise ValueError(
                f"Model {name!r} has {arr.shape[1]} stimuli but another model "
                f"has {n_stim}; all models must score the same stimulus pool."
            )
        if not np.isfinite(arr).all() or arr.min() < 0.0 or arr.max() > 1.0:
            raise ValueError(
                f"Model {name!r}: p_left values must be finite and in [0, 1]."
            )
        out[name] = np.clip(arr, _P_CLIP, 1.0 - _P_CLIP)
    assert n_stim is not None
    return out, n_stim


def _model_prior(
    names: Sequence[str], model_weights: Optional[Dict[str, float]]
) -> np.ndarray:
    """Normalized model prior; uniform when weights are absent or degenerate.

    Mirrors ``eig_from_prior_means``: a weights dict whose mass on these models
    is zero falls back to uniform rather than failing.
    """
    if model_weights:
        w = np.array([model_weights.get(n, 0.0) for n in names], dtype=float)
        if w.sum() <= 0:
            w = np.ones(len(names))
    else:
        w = np.ones(len(names))
    return w / w.sum()


def _entropy_bits(w: np.ndarray, axis: int = -1) -> np.ndarray:
    """Shannon entropy in bits along ``axis``, with 0·log(0) = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(w > 0, w * np.log2(w), 0.0)
    return -terms.sum(axis=axis)


class _ScenarioState:
    """Monte Carlo scenarios with per-draw model likelihoods for observed stimuli.

    Holds, for each scenario t and model m, the log-likelihood of the responses
    observed so far under every parameter draw d of m. The model posterior of a
    scenario is prior(m) · mean_d exp(logL[t, m, d]), normalized over m.
    """

    def __init__(
        self,
        p: Dict[str, np.ndarray],
        prior: np.ndarray,
        n_scenarios: int,
        rng: np.random.Generator,
    ) -> None:
        self.p = p
        self.names = list(p)
        self.prior = prior
        self.rng = rng
        self.m_idx = rng.choice(len(self.names), size=n_scenarios, p=prior)
        n_draws = np.array([p[n].shape[0] for n in self.names])
        self.d_idx = rng.integers(0, n_draws[self.m_idx])
        self.logL = {
            n: np.zeros((n_scenarios, p[n].shape[0])) for n in self.names
        }
        self._lhat_cache: Optional[Dict[str, np.ndarray]] = None

    def _scaled_likelihoods(self) -> Dict[str, np.ndarray]:
        """exp(logL - c_t) per model — likelihoods scaled by a per-scenario
        constant that cancels when the posterior is normalized over models.

        Cached between observations: logL only changes in ``observe``, but CELF
        calls this once per candidate re-evaluation, so recomputing the
        exponentials each time dominated lazy selection's runtime.
        """
        if self._lhat_cache is not None:
            return self._lhat_cache
        c = np.max(
            np.stack([self.logL[n].max(axis=1) for n in self.names]), axis=0
        )
        if not np.isfinite(c).all():
            raise FloatingPointError(
                "A scenario's observed responses have zero likelihood under "
                "every model and draw; p_left clipping should prevent this."
            )
        self._lhat_cache = {
            n: np.exp(self.logL[n] - c[:, None]) for n in self.names
        }
        return self._lhat_cache

    def posterior_entropy(self) -> np.ndarray:
        """Entropy (bits) of each scenario's current model posterior, shape (T,)."""
        lhat = self._scaled_likelihoods()
        marg = np.stack([lhat[n].mean(axis=1) for n in self.names], axis=1)
        w = marg * self.prior[None, :]
        w /= w.sum(axis=1, keepdims=True)
        return _entropy_bits(w, axis=1)

    def generative_p(self, cols: np.ndarray) -> np.ndarray:
        """Each scenario's true p_left for ``cols`` (from its model + draw),
        shape (T, len(cols))."""
        q = np.empty((len(self.m_idx), len(cols)))
        for k, n in enumerate(self.names):
            rows = np.nonzero(self.m_idx == k)[0]
            if rows.size:
                q[rows] = self.p[n][np.ix_(self.d_idx[rows], cols)]
        return q

    def marginal_gains(self, cols: np.ndarray, h_current: np.ndarray) -> np.ndarray:
        """Expected posterior-entropy reduction from adding each candidate.

        One matmul per model per response outcome contracts the draw axis:
        mean_d(L[t, d] · p[d, j]) = (L @ P) / n_draws, giving each candidate's
        marginal model likelihood without materializing a (T, D, n) tensor.
        """
        lhat = self._scaled_likelihoods()
        marg_left, marg_right = [], []
        for n in self.names:
            p_cols = self.p[n][:, cols]
            n_draws = p_cols.shape[0]
            marg_left.append(lhat[n] @ p_cols / n_draws)
            marg_right.append(lhat[n] @ (1.0 - p_cols) / n_draws)
        h_left = self._entropy_of(np.stack(marg_left, axis=1))
        h_right = self._entropy_of(np.stack(marg_right, axis=1))
        q = self.generative_p(cols)
        h_next = q * h_left + (1.0 - q) * h_right
        return h_current.mean() - h_next.mean(axis=0)

    def _entropy_of(self, marg: np.ndarray) -> np.ndarray:
        """Posterior entropy from (T, K, C) marginal likelihoods, shape (T, C)."""
        w = marg * self.prior[None, :, None]
        w /= w.sum(axis=1, keepdims=True)
        return _entropy_bits(w, axis=1)

    def observe(self, col: int) -> None:
        """Sample each scenario's response to ``col`` and fold it into logL."""
        q = self.generative_p(np.array([col]))[:, 0]
        r = self.rng.random(len(q)) < q
        for n in self.names:
            p_col = self.p[n][:, col]
            self.logL[n] += np.where(
                r[:, None], np.log(p_col)[None, :], np.log1p(-p_col)[None, :]
            )
        self._lhat_cache = None


@dataclass(frozen=True)
class JointEIGSelection:
    """Result of greedy joint-EIG selection."""

    indices: List[int]  # selected stimulus indices, in selection order
    joint_eig_bits: List[float]  # in-sample I(M; R_S) after each selection
    n_scenarios: int


def estimate_joint_eig(
    p_left_draws: Dict[str, np.ndarray],
    indices: Sequence[int],
    *,
    model_weights: Optional[Dict[str, float]] = None,
    n_scenarios: int = 1000,
    seed: int = 42,
) -> float:
    """Monte Carlo estimate of I(M; R_S) in bits for the stimulus set ``indices``."""
    p, n_stim = _validated_p(p_left_draws)
    cols = np.asarray(list(indices), dtype=int)
    if cols.size == 0:
        raise ValueError("indices must be non-empty.")
    if cols.min() < 0 or cols.max() >= n_stim:
        raise ValueError(f"indices out of range for a pool of {n_stim} stimuli.")
    if n_scenarios < 1:
        raise ValueError(f"n_scenarios must be >= 1, got {n_scenarios}.")

    prior = _model_prior(list(p), model_weights)
    state = _ScenarioState(p, prior, n_scenarios, np.random.default_rng(seed))
    # Sample all responses at once and fold them in with one matmul per model:
    # logL[t, d] = sum_i [r_ti · log p_di + (1 - r_ti) · log(1 - p_di)].
    q = state.generative_p(cols)
    r = state.rng.random(q.shape) < q
    for n in state.names:
        p_cols = state.p[n][:, cols]
        state.logL[n] = r @ np.log(p_cols).T + (~r) @ np.log1p(-p_cols).T
    state._lhat_cache = None  # logL set directly, bypassing observe()
    h_prior = float(_entropy_bits(prior))
    return h_prior - float(state.posterior_entropy().mean())


def select_n_joint_eig(
    p_left_draws: Dict[str, np.ndarray],
    n_select: int,
    *,
    model_weights: Optional[Dict[str, float]] = None,
    n_scenarios: int = 1000,
    seed: int = 42,
    lazy: bool = False,
    chunk_size: int = 4096,
) -> JointEIGSelection:
    """Greedily select ``n_select`` stimuli maximizing joint EIG about M.

    p_left_draws: ``{model_name: (n_draws, n_stim) array}`` of prior-predictive
        p_left (e.g. from ``prior_predict_p_left_draws``). Draw counts may
        differ across models; stimulus counts may not.
    model_weights: optional model prior (registry weights); uniform if omitted.
    n_scenarios: Monte Carlo scenarios; estimate error shrinks as 1/sqrt(T).
    lazy: ``False`` (default) re-scores every candidate at every step — exact
        greedy. ``True`` uses CELF lazy re-evaluation: much faster, but can
        miss synergistic candidates whose gain grew (see module docstring).
    chunk_size: candidates per vectorized pass (memory/perf knob only).
    """
    p, n_stim = _validated_p(p_left_draws)
    if not 1 <= n_select <= n_stim:
        raise ValueError(
            f"n_select must be in [1, {n_stim}] for this pool, got {n_select}."
        )
    if n_scenarios < 1:
        raise ValueError(f"n_scenarios must be >= 1, got {n_scenarios}.")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")

    prior = _model_prior(list(p), model_weights)
    state = _ScenarioState(p, prior, n_scenarios, np.random.default_rng(seed))
    h_prior = float(_entropy_bits(prior))
    h_current = state.posterior_entropy()

    def all_gains() -> np.ndarray:
        gains = np.empty(n_stim)
        for start in range(0, n_stim, chunk_size):
            cols = np.arange(start, min(start + chunk_size, n_stim))
            gains[cols] = state.marginal_gains(cols, h_current)
        return gains

    selected: List[int] = []
    trajectory: List[float] = []
    gains = all_gains()
    if lazy:
        heap = [(-gains[j], j) for j in range(n_stim)]
        heapq.heapify(heap)
        evaluated_at = np.zeros(n_stim, dtype=int)

    for step in range(n_select):
        if lazy:
            while True:
                neg_gain, j = heapq.heappop(heap)
                if j in selected:
                    continue
                if evaluated_at[j] == step:
                    break
                gain = state.marginal_gains(np.array([j]), h_current)[0]
                evaluated_at[j] = step
                heapq.heappush(heap, (-gain, j))
        else:
            if step > 0:
                gains = all_gains()
            gains[selected] = -np.inf
            j = int(np.argmax(gains))

        state.observe(j)
        selected.append(int(j))
        h_current = state.posterior_entropy()
        trajectory.append(h_prior - float(h_current.mean()))

    return JointEIGSelection(
        indices=selected, joint_eig_bits=trajectory, n_scenarios=n_scenarios
    )
