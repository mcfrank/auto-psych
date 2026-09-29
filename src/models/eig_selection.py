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
posterior-entropy reduction. Scoring one candidate costs, for every outcome
k = 0..n_responses and every model, a matmul over the draw axis plus the
entropy of each scenario's posterior, so one pass over the 43,434-pair design
pool at 40 responses is minutes of CPU. Exact greedy (``lazy=False``) makes a
full pass per pick. ``lazy=True`` is *lazy batched greedy*: an exact full pass
stores every candidate's gain; each later pick re-scores only the
``lazy_batch_size`` candidates with the highest stored gain, in one vectorized
call, and accepts the best fresh gain once it is at least the highest stored
gain among the candidates not re-scored in this step (otherwise it re-scores
the next batch). A full pass every ``refresh_every`` picks refreshes every
stored gain.

Lazy selection would be exact if gains only shrank as the set grows
(submodularity: a stored gain would then bound the current one from above).
Joint mutual information about model identity is **not** submodular in
general: per-draw likelihoods deliberately keep the correlation that shared
parameters induce, so a stimulus can become *more* informative once a
correlated partner is chosen (synergy). A candidate whose gain grew while it
sat outside the re-scored batches is missed until the next full pass, so lazy
selection is an approximation of exact greedy. How close it comes on real
designs is measured by ``scripts/subjective_randomness/validate_lazy_eig.py``.

Candidates are scored in chunks of ``chunk_size`` columns, on ``n_threads``
threads (NumPy releases the GIL in the matmuls and element-wise passes; BLAS
itself is held to one thread per chunk). The chunk boundaries do not depend on
the thread count, so neither do the results. ``dtype="float32"`` scores in
single precision, about twice as fast; the scenario likelihoods stay float64
and are rescaled per scenario (and each outcome's likelihood by its maximum
over p_left), so nothing that matters underflows. The noise-floor stop always
judges the chosen pick in float64.

The ``joint_eig_bits`` trajectory is estimated from the same scenarios used
for selection (in-sample); use :func:`estimate_joint_eig` with a fresh seed
for an unbiased estimate of a chosen set.
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
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


def _log_binomial_coefficients(n: int) -> np.ndarray:
    """log C(n, k) for k = 0..n."""
    return np.array(
        [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) for k in range(n + 1)]
    )


def _outcome_kernel_maxima(n: int) -> np.ndarray:
    """max over p of log[p^k (1 - p)^(n - k)] (attained at p = k/n), for k = 0..n.

    Dividing an outcome's likelihood by this maximum is a constant per outcome,
    common to every model and draw, so it cancels from the posterior; it keeps
    the likelihood of the draws that make the outcome probable near 1, where
    float32 would otherwise underflow (0.05**40 is below its range).
    """
    out = np.zeros(n + 1)
    for k in range(n + 1):
        if k > 0:
            out[k] += k * math.log(k / n)
        if k < n:
            out[k] += (n - k) * math.log1p(-k / n)
    return out


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
        self._weighted_cache: Dict[np.dtype, List[np.ndarray]] = {}
        self._log_comb = _log_binomial_coefficients(n_responses)
        self._kernel_max = _outcome_kernel_maxima(n_responses)

    def _scaled_likelihoods(self) -> Dict[str, np.ndarray]:
        """exp(logL - c_t) per model — likelihoods scaled by a per-scenario
        constant that cancels when the posterior is normalized over models.

        Cached between observations: logL only changes in ``observe``.
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

    def _weighted_likelihoods(self, dtype: np.dtype) -> List[np.ndarray]:
        """prior(m) / n_draws(m) · scaled likelihoods, per model, in ``dtype``:
        the left factor of the matmul that marginalizes over draws. Cached
        between observations."""
        if dtype not in self._weighted_cache:
            lhat = self._scaled_likelihoods()
            self._weighted_cache[dtype] = [
                (lhat[n] * (self.prior[i] / lhat[n].shape[1])).astype(dtype)
                for i, n in enumerate(self.names)
            ]
        return self._weighted_cache[dtype]

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
        """Expected posterior-entropy reduction from adding each candidate
        (float64 scoring; see ``next_entropy``)."""
        return h_current.mean() - self.next_entropy(cols).mean(axis=0)

    def next_entropy(self, cols: np.ndarray, dtype: np.dtype = np.dtype(np.float64)) -> np.ndarray:
        """Each scenario's expected posterior entropy after adding each
        candidate, shape (T, len(cols)), computed in ``dtype``.

        For each outcome k = 0..n_responses (the count of "left" responses),
        one matmul per model contracts the draw axis:
        mean_d(L[t, d] · p[d, j]^k (1 - p[d, j])^(n - k)) = (L @ P_k) / n_draws,
        giving each candidate's marginal model likelihood of that outcome
        without materializing a (T, D, n) tensor. P_k is computed as
        exp(k·log p + (n - k)·log(1 - p) - max over p), the constant cancelling
        in the posterior. The posterior entropy after outcome k is weighted by
        its probability under the scenario's true p_left, Binomial(k; n, q).
        """
        dtype = np.dtype(dtype)
        n = self.n_responses
        weighted = self._weighted_likelihoods(dtype)
        # Outcome log-likelihoods are linear in k: n·log(1-p) + k·log(p/(1-p)).
        slopes, bases = [], []
        for name in self.names:
            log_p = np.log(self.p[name][:, cols])
            log_1mp = np.log1p(-self.p[name][:, cols])
            slopes.append((log_p - log_1mp).astype(dtype))
            bases.append((n * log_1mp).astype(dtype))
        q = self.generative_p(cols)
        log_q, log_1mq = np.log(q), np.log1p(-q)
        q_slope = (log_q - log_1mq).astype(dtype)
        q_base = (n * log_1mq).astype(dtype)

        n_models, n_scen = len(self.names), len(self.m_idx)
        post = np.empty((n_models, n_scen, len(cols)), dtype=dtype)
        log_post = np.empty_like(post)
        h_next = np.zeros((n_scen, len(cols)), dtype=dtype)
        tiny = np.finfo(dtype).tiny
        for k in range(n + 1):
            for i in range(n_models):
                outcome_lik = np.exp(bases[i] + k * slopes[i] - dtype.type(self._kernel_max[k]))
                np.matmul(weighted[i], outcome_lik, out=post[i])
            # Posterior entropy, normalized first so that it is accurate in
            # float32 too. An outcome whose likelihood underflows (below the
            # smallest normal number) under every model has numerically zero
            # probability; its entropy is set to 0.
            total = post.sum(axis=0)
            inv_total = np.divide(1.0, total, out=np.zeros_like(total), where=total >= tiny)
            post *= inv_total
            np.maximum(post, tiny, out=log_post)
            np.log(log_post, out=log_post)
            log_post *= post
            h_k = log_post.sum(axis=0)
            weight = np.exp(q_base + k * q_slope + dtype.type(self._log_comb[k]))
            weight *= h_k
            h_next -= weight
        h_next /= dtype.type(math.log(2.0))
        return h_next

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
        self.forget_likelihoods()

    def forget_likelihoods(self) -> None:
        """Drop the likelihood caches after ``logL`` changed."""
        self._lhat_cache = None
        self._weighted_cache = {}


@dataclass(frozen=True)
class JointEIGSelection:
    """Result of greedy joint-EIG selection."""

    indices: List[int]  # selected stimulus indices, in selection order
    joint_eig_bits: List[float]  # in-sample I(M; R_S) after each selection
    n_scenarios: int
    # True when selection stopped early because the best gain was noise.
    stopped_at_noise_floor: bool = False


def scenario_posterior_entropies(
    p_left_draws: Dict[str, np.ndarray],
    indices: Sequence[int],
    *,
    model_weights: Optional[Dict[str, float]] = None,
    n_scenarios: int = 1000,
    seed: int = 42,
    n_responses: int = 1,
) -> np.ndarray:
    """Each scenario's model-posterior entropy (bits) after observing the
    stimulus set ``indices``, each answered ``n_responses`` times; shape (T,).

    Two sets scored with the same ``seed`` share their scenarios' models and
    draws (the responses differ with the stimuli), so the per-scenario
    difference gives a paired Monte Carlo error for comparing them.
    """
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
    state.forget_likelihoods()  # logL set directly, bypassing observe()
    return state.posterior_entropy()


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
    h_posterior = scenario_posterior_entropies(
        p_left_draws,
        indices,
        model_weights=model_weights,
        n_scenarios=n_scenarios,
        seed=seed,
        n_responses=n_responses,
    )
    names = list(p_left_draws)
    h_prior = float(_entropy_bits(_model_prior(names, model_weights)))
    return h_prior - float(h_posterior.mean())


def select_n_joint_eig(
    p_left_draws: Dict[str, np.ndarray],
    n_select: int,
    *,
    model_weights: Optional[Dict[str, float]] = None,
    n_scenarios: int = 1000,
    seed: int = 42,
    lazy: bool = False,
    lazy_batch_size: int = 512,
    refresh_every: int = 16,
    chunk_size: int = 64,
    n_threads: int = 1,
    dtype: str = "float64",
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
    lazy: ``False`` (default) re-scores every candidate at every pick — exact
        greedy. ``True`` is lazy batched greedy (see the module docstring): an
        approximation, since joint EIG is not submodular.
    lazy_batch_size: candidates re-scored per batch between full passes
        (lazy only). A batch that covers every remaining candidate is a full
        pass, so ``lazy_batch_size >= n_stim`` is exact greedy.
    refresh_every: picks between exact full passes (lazy only); the first
        pick of this call is always a full pass, and ``refresh_every=1`` is
        exact greedy.
    chunk_size: candidates per vectorized scoring call. Results depend on it
        in the last bits (BLAS blocks a matmul differently for different
        widths), never on ``n_threads``.
    n_threads: threads scoring chunks concurrently.
    dtype: ``"float64"`` or ``"float32"``, the precision of candidate scoring.
    n_responses: responses each selected stimulus receives (an experiment's
        participant count). Scoring cost grows linearly with it.
    stop_below_noise: stop (with fewer than ``n_select`` picks) once the best
        candidate's gain is at most twice the Monte Carlo standard error of that
        gain across scenarios — the pick would be chosen on noise — or below
        ``NEGLIGIBLE_GAIN_BITS``. The chosen pick's per-scenario gain is
        re-scored in float64 for this test, whatever ``dtype``.
    preselected: stimuli already chosen (by another objective): their
        responses are observed first, they are never picked again, and they
        are not part of the returned ``indices``.
    """
    _validate_n_responses(n_responses)
    p, n_stim = _validated_p(p_left_draws)
    if not 1 <= n_select <= n_stim:
        raise ValueError(
            f"n_select must be in [1, {n_stim}] for this pool, got {n_select}."
        )
    if n_scenarios < 1:
        raise ValueError(f"n_scenarios must be >= 1, got {n_scenarios}.")
    for knob, value in (
        ("chunk_size", chunk_size),
        ("n_threads", n_threads),
        ("lazy_batch_size", lazy_batch_size),
        ("refresh_every", refresh_every),
    ):
        if value < 1:
            raise ValueError(f"{knob} must be >= 1, got {value}.")
    if dtype not in _SCORING_DTYPES:
        raise ValueError(f"dtype must be one of {sorted(_SCORING_DTYPES)}, got {dtype!r}.")
    scoring_dtype = np.dtype(dtype)

    prior = _model_prior(list(p), model_weights)
    state = _ScenarioState(
        p, prior, n_scenarios, np.random.default_rng(seed), n_responses
    )
    h_prior = float(_entropy_bits(prior))
    for j in preselected:
        state.observe(int(j))
    h_current = state.posterior_entropy()

    selected: List[int] = []
    trajectory: List[float] = []
    available = np.ones(n_stim, dtype=bool)
    available[[int(j) for j in preselected]] = False
    stopped = False
    # Every candidate's gain as last scored: fresh after a full pass, stale
    # (an approximate upper bound) between full passes in lazy mode.
    gains = np.full(n_stim, -np.inf)

    with ThreadPoolExecutor(max_workers=n_threads) as pool, _single_threaded_blas():

        def score(cols: np.ndarray) -> np.ndarray:
            """Expected next posterior entropy, (T, len(cols)), chunk by chunk."""
            state._weighted_likelihoods(scoring_dtype)  # fill the cache before the threads read it
            chunks = [cols[i : i + chunk_size] for i in range(0, len(cols), chunk_size)]
            parts = pool.map(lambda c: state.next_entropy(c, scoring_dtype), chunks)
            return np.concatenate(list(parts), axis=1)

        def gain_of(next_h: np.ndarray) -> np.ndarray:
            return h_current.mean() - next_h.mean(axis=0, dtype=np.float64)

        def full_pass() -> int:
            candidates = np.flatnonzero(available)
            gains[:] = -np.inf
            for start in range(0, len(candidates), _FULL_PASS_BLOCK):
                block = candidates[start : start + _FULL_PASS_BLOCK]
                gains[block] = gain_of(score(block))
            return int(np.argmax(gains))

        def lazy_pass() -> int:
            """Re-score batches of the best stored gains until the best fresh
            gain is at least every stored gain not re-scored in this step."""
            candidates = np.flatnonzero(available)
            if lazy_batch_size >= len(candidates):
                return full_pass()
            # Highest stored gain first; ties in index order, as in argmax.
            order = candidates[np.argsort(-gains[candidates], kind="stable")]
            best, best_gain = -1, -np.inf
            for start in range(0, len(order), lazy_batch_size):
                batch = np.sort(order[start : start + lazy_batch_size])
                gains[batch] = gain_of(score(batch))
                top = int(batch[np.argmax(gains[batch])])
                if gains[top] > best_gain or (gains[top] == best_gain and top < best):
                    best, best_gain = top, float(gains[top])
                rest = order[start + lazy_batch_size :]
                if not rest.size or best_gain >= gains[rest[0]]:
                    return best
            return best

        for step in range(n_select):
            if not lazy or step % refresh_every == 0:
                j = full_pass()
            else:
                j = lazy_pass()
            if stop_below_noise:
                per_scenario = h_current - state.next_entropy(np.array([j]))[:, 0]
                noise = per_scenario.std(ddof=1) / np.sqrt(len(per_scenario))
                if per_scenario.mean() <= max(2 * noise, NEGLIGIBLE_GAIN_BITS):
                    stopped = True
                    break

            state.observe(j)
            selected.append(j)
            available[j] = False
            gains[j] = -np.inf
            h_current = state.posterior_entropy()
            trajectory.append(h_prior - float(h_current.mean()))

    return JointEIGSelection(
        indices=selected,
        joint_eig_bits=trajectory,
        n_scenarios=n_scenarios,
        stopped_at_noise_floor=stopped,
    )


# Precisions candidate scoring supports.
_SCORING_DTYPES = {"float32", "float64"}

# Columns handed to the thread pool at once in a full pass: bounds the
# (T, columns) next-entropy array a full pass holds, whatever the pool size.
_FULL_PASS_BLOCK = 4096


def _single_threaded_blas():
    """BLAS held to one thread while chunks are scored on the thread pool.

    Scoped (``threadpoolctl``, which PyMC already depends on) rather than set
    in the environment, so the caller's BLAS settings are back in force once
    selection returns. The holdout harness already pins BLAS to one thread;
    this keeps a caller that did not from oversubscribing the CPUs.
    """
    from threadpoolctl import threadpool_limits

    return threadpool_limits(limits=1, user_api="blas")
