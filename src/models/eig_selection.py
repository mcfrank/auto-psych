"""Greedy selection of the N-stimulus set with maximal joint EIG.

Objective: choose stimuli S (|S| = N) maximizing I(M; R_S) — the mutual
information between model identity M and the joint response vector — from
per-model, per-draw prior-predictive ``p_left`` arrays. Working per draw
(rather than from each model's mean ``p_left``) keeps the correlation that
shared parameters induce between stimuli *within* a model, so a near-duplicate
of an already-selected stimulus is correctly scored as mostly redundant.

Estimation is by Monte Carlo scenarios. A scenario is one simulated "world":
a model sampled from the model prior, one of its parameter draws, and responses
for the selected stimuli generated from that draw's ``p_left``. Each stimulus is
answered ``n_responses`` times (an experiment shows every stimulus to all its
participants, who share one ``p_left``), so a stimulus yields a count
k ~ Binomial(n_responses, p_left); ``n_responses=1`` is a single Bernoulli
response. Each scenario tracks per-draw log-likelihoods for every model, giving
a posterior over models via p(k_S | m) = mean over draws of the product
likelihood; joint EIG is H(M) minus the mean posterior entropy across scenarios.
The binomial coefficient is the same for every model and draw, so it cancels
from the posterior and is left out of the likelihoods.

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
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.registry.io import validate_theory_weights

# Numeric floor for probabilities entering logs. The PyMC models already clip
# p_left to [1e-6, 1 - 1e-6]; this only guards hand-built arrays at 0 or 1.
_P_CLIP = 1e-12

# A gain this small is floating-point dust, not information: once the model is
# identified the remaining gains are ~1e-50 bits, and their "noise" is dust of
# the same size, so the 2-standard-error rule alone cannot call them noise.
NEGLIGIBLE_GAIN_BITS = 1e-6


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
        validated = validate_theory_weights(model_weights)
        w = np.array([validated.get(n, 0.0) for n in names], dtype=float)
        if w.sum() <= 0:
            w = np.ones(len(names))
    else:
        w = np.ones(len(names))
    return w / w.sum()


def _validate_n_responses(n_responses: int) -> None:
    if not isinstance(n_responses, (int, np.integer)) or n_responses < 1:
        raise ValueError(f"n_responses must be an integer >= 1, got {n_responses!r}.")


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
        n_responses: int = 1,
    ) -> None:
        self.p = p
        self.n_responses = n_responses
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

        For each outcome k = 0..n_responses (the count of "left" responses),
        one matmul per model contracts the draw axis:
        mean_d(L[t, d] · p[d, j]^k (1 - p[d, j])^(n - k)) = (L @ P_k) / n_draws,
        giving each candidate's marginal model likelihood of that outcome
        without materializing a (T, D, n) tensor. The posterior entropy after
        outcome k is weighted by its probability under the scenario's true
        p_left, Binomial(k; n, q).
        """
        return h_current.mean() - self.next_entropy(cols).mean(axis=0)

    def next_entropy(self, cols: np.ndarray) -> np.ndarray:
        """Each scenario's expected posterior entropy after adding each
        candidate, shape (T, len(cols)); see ``marginal_gains``."""
        n = self.n_responses
        lhat = self._scaled_likelihoods()
        q = self.generative_p(cols)
        h_next = np.zeros_like(q)
        for k in range(n + 1):
            marg = []
            for name in self.names:
                p_cols = self.p[name][:, cols]
                n_draws = p_cols.shape[0]
                outcome_lik = p_cols**k * (1.0 - p_cols) ** (n - k)
                marg.append(lhat[name] @ outcome_lik / n_draws)
            h_k = self._entropy_of(np.stack(marg, axis=1))
            h_next += math.comb(n, k) * q**k * (1.0 - q) ** (n - k) * h_k
        return h_next

    def _entropy_of(self, marg: np.ndarray) -> np.ndarray:
        """Posterior entropy from (T, K, C) marginal likelihoods, shape (T, C).

        An outcome whose likelihood underflows to 0 under every model has
        (numerically) zero probability; its entropy is set to 0 rather than NaN.
        """
        w = marg * self.prior[None, :, None]
        total = w.sum(axis=1, keepdims=True)
        w = np.divide(w, total, out=np.zeros_like(w), where=total > 0)
        return _entropy_bits(w, axis=1)

    def _draw_counts(self, q: np.ndarray) -> np.ndarray:
        """Count of "left" among n_responses Bernoulli(q) responses, per entry.

        Drawn as n_responses uniform arrays (not rng.binomial) so that with one
        response the random stream is exactly the single-response one.
        """
        return (self.rng.random((self.n_responses,) + q.shape) < q).sum(axis=0)

    def observe(self, col: int) -> None:
        """Sample each scenario's responses to ``col`` and fold them into logL."""
        q = self.generative_p(np.array([col]))[:, 0]
        k = self._draw_counts(q)
        n = self.n_responses
        for name in self.names:
            p_col = self.p[name][:, col]
            self.logL[name] += (
                k[:, None] * np.log(p_col)[None, :]
                + (n - k)[:, None] * np.log1p(-p_col)[None, :]
            )
        self._lhat_cache = None


@dataclass(frozen=True)
class JointEIGSelection:
    """Result of greedy joint-EIG selection."""

    indices: List[int]  # selected stimulus indices, in selection order
    joint_eig_bits: List[float]  # in-sample I(M; R_S) after each selection
    n_scenarios: int
    # True when selection stopped early because the best gain was noise.
    stopped_at_noise_floor: bool = False


def estimate_joint_eig(
    p_left_draws: Dict[str, np.ndarray],
    indices: Sequence[int],
    *,
    model_weights: Optional[Dict[str, float]] = None,
    n_scenarios: int = 1000,
    seed: int = 42,
    n_responses: int = 1,
) -> float:
    """Monte Carlo estimate of I(M; R_S) in bits for the stimulus set ``indices``,
    each stimulus answered ``n_responses`` times."""
    _validate_n_responses(n_responses)
    p, n_stim = _validated_p(p_left_draws)
    cols = np.asarray(list(indices), dtype=int)
    if cols.size == 0:
        raise ValueError("indices must be non-empty.")
    if cols.min() < 0 or cols.max() >= n_stim:
        raise ValueError(f"indices out of range for a pool of {n_stim} stimuli.")
    if n_scenarios < 1:
        raise ValueError(f"n_scenarios must be >= 1, got {n_scenarios}.")

    prior = _model_prior(list(p), model_weights)
    state = _ScenarioState(
        p, prior, n_scenarios, np.random.default_rng(seed), n_responses
    )
    # Sample all response counts at once and fold them in with one matmul per
    # model: logL[t, d] = sum_i [k_ti · log p_di + (n - k_ti) · log(1 - p_di)].
    q = state.generative_p(cols)
    k = state._draw_counts(q)
    for name in state.names:
        p_cols = state.p[name][:, cols]
        state.logL[name] = k @ np.log(p_cols).T + (n_responses - k) @ np.log1p(-p_cols).T
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
    n_responses: int = 1,
    stop_below_noise: bool = False,
    preselected: Sequence[int] = (),
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
    n_responses: responses each selected stimulus receives (an experiment's
        participant count). Scoring cost grows linearly with it.
    stop_below_noise: stop (with fewer than ``n_select`` picks) once the best
        candidate's gain is at most twice the Monte Carlo standard error of that
        gain across scenarios — the pick would be chosen on noise — or below
        ``NEGLIGIBLE_GAIN_BITS``. Exact greedy only.
    preselected: stimuli already chosen (by another objective): their
        responses are observed first, they are never picked again, and they
        are not part of the returned ``indices``.
    """
    if stop_below_noise and lazy:
        raise ValueError("stop_below_noise needs exact greedy (lazy=False).")
    _validate_n_responses(n_responses)
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
    state = _ScenarioState(
        p, prior, n_scenarios, np.random.default_rng(seed), n_responses
    )
    h_prior = float(_entropy_bits(prior))
    for j in preselected:
        state.observe(int(j))
    h_current = state.posterior_entropy()

    def all_gains() -> np.ndarray:
        gains = np.empty(n_stim)
        for start in range(0, n_stim, chunk_size):
            cols = np.arange(start, min(start + chunk_size, n_stim))
            gains[cols] = state.marginal_gains(cols, h_current)
        return gains

    selected: List[int] = []
    trajectory: List[float] = []
    taken = [int(j) for j in preselected]
    stopped = False
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
            gains[selected + taken] = -np.inf
            j = int(np.argmax(gains))
            if stop_below_noise:
                per_scenario = h_current - state.next_entropy(np.array([j]))[:, 0]
                noise = per_scenario.std(ddof=1) / np.sqrt(len(per_scenario))
                if per_scenario.mean() <= max(2 * noise, NEGLIGIBLE_GAIN_BITS):
                    stopped = True
                    break

        state.observe(j)
        selected.append(int(j))
        h_current = state.posterior_entropy()
        trajectory.append(h_prior - float(h_current.mean()))

    return JointEIGSelection(
        indices=selected,
        joint_eig_bits=trajectory,
        n_scenarios=n_scenarios,
        stopped_at_noise_floor=stopped,
    )
