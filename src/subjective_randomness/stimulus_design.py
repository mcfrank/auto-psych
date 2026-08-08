"""Rank and select stimuli by how well they DISCRIMINATE between models.

Model recovery is limited by stimulus diagnosticity: if the candidate models
make near-identical predictions on the stimuli, no amount of data or MCMC draws
can separate them. This module scores a stimulus by the expected information it
carries about *which model* produced the response — the mutual information (in
bits) between model identity and the binary choice — computed from each model's
``p_left``.

This is the fast, MCMC-free counterpart to ``src/pipelines/outer_loop/eig.py``,
which computes the same quantity from the PyMC models' prior predictive. Use
this module (pure-Python reference families) for quick design iteration; use
``eig.py`` for the full prior-predictive EIG over the fitted PyMC models.
"""

from __future__ import annotations

import heapq
import importlib
import itertools
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from src.models.model_manifest import read_manifest_names
from src.subjective_randomness.pymc_model_families import REGISTRY_DIR

# A predictor maps a stimulus ({"sequence_a", "sequence_b"}) to P(choose left).
PredictFn = Callable[[Mapping[str, str]], float]

# Sorted, because the legacy path was written when it called
# default_model_family_names(), which enumerated this package with pkgutil and
# sorted the result. Model order is not cosmetic here: it permutes the columns of
# the prediction matrix (changing the float summation order in _marginal_eig) and
# permutes which model each sampled scenario identity means in the greedy loop.
_LEGACY_COMPAT_MODEL_FAMILIES: Tuple[str, ...] = (
    "bayesian_diagnosticity",
    "encoding_compressibility",
    "prototype_similarity",
    "window_typicality",
)


def generate_candidate_pool(
    n_pairs: int = 200,
    *,
    lengths: Tuple[int, ...] = (6, 8),
    seed: int = 0,
) -> List[Dict[str, str]]:
    """Sample a diverse pool of candidate stimulus pairs to mine for high EIG.

    For each length in ``lengths`` the full sequence space (``2**length`` H/T
    strings) is enumerated; ``n_pairs`` distinct unordered same-length pairs are
    then sampled across lengths. Full enumeration makes the pool maximally
    varied (every run/alternation/imbalance structure is represented), and
    sampling is deterministic given ``seed``. Lengths are capped at 12 to bound
    enumeration.
    """
    if n_pairs < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}.")
    if any(length > 12 for length in lengths):
        raise ValueError("Sequence lengths are capped at 12 to bound enumeration.")

    sequences_by_length = {
        length: ["".join(bits) for bits in itertools.product("HT", repeat=length)]
        for length in lengths
    }
    total_pairs = sum(
        len(seqs) * (len(seqs) - 1) // 2 for seqs in sequences_by_length.values()
    )
    if n_pairs > total_pairs:
        raise ValueError(
            f"Requested {n_pairs} pairs but only {total_pairs} distinct pairs "
            f"exist for lengths {lengths}."
        )

    rng = np.random.default_rng(seed)
    seen: set = set()
    pool: List[Dict[str, str]] = []
    max_pairs_by_length = {
        length: len(seqs) * (len(seqs) - 1) // 2
        for length, seqs in sequences_by_length.items()
    }
    taken_by_length = {length: 0 for length in sequences_by_length}
    lengths_cycle = list(lengths)
    while len(pool) < n_pairs:
        if not lengths_cycle:
            raise RuntimeError(
                "All lengths exhausted before reaching n_pairs; the total-pairs "
                "feasibility check above should have caught this."
            )
        length = lengths_cycle[len(pool) % len(lengths_cycle)]
        # A short length can run out of distinct pairs before its round-robin
        # share is met (length 4 has only 120); once exhausted, every further
        # draw for it would be rejected forever, so hand its remaining slots to
        # the other lengths.
        if taken_by_length[length] == max_pairs_by_length[length]:
            lengths_cycle.remove(length)
            continue
        seqs = sequences_by_length[length]
        i, j = rng.integers(0, len(seqs), size=2)
        if i == j:
            continue
        key = (seqs[i], seqs[j]) if i < j else (seqs[j], seqs[i])
        if key in seen:
            continue
        seen.add(key)
        taken_by_length[length] += 1
        pool.append({"sequence_a": key[0], "sequence_b": key[1]})
    return pool


def enumerate_all_pairs(lengths: Sequence[int]) -> List[Dict[str, str]]:
    """Every distinct unordered H/T pair over all sequences of the given lengths.

    The full ``2**L`` sequence space is enumerated for each length ``L`` in
    ``lengths`` and pooled into one sequence set; every unordered pair of two
    distinct sequences from that pool is emitted (deterministic order),
    *including cross-length pairs* (e.g. a length-5 sequence vs a length-7 one).
    This is the exhaustive counterpart to :func:`generate_candidate_pool`:
    instead of sampling ``n_pairs``, it returns the *whole* pair space over the
    union of the lengths — every run/alternation/imbalance contrast both within
    and across lengths. For lengths ``1..8`` the pool is 510 sequences, so
    ``C(510, 2) = 129,795`` pairs. Duplicate lengths are ignored; lengths are
    capped at 12 to bound enumeration.
    """
    lengths = tuple(sorted(set(lengths)))
    if not lengths:
        raise ValueError("lengths must be non-empty.")
    if any(length < 1 for length in lengths):
        raise ValueError(f"Sequence lengths must be >= 1, got {lengths}.")
    if any(length > 12 for length in lengths):
        raise ValueError("Sequence lengths are capped at 12 to bound enumeration.")

    sequences: List[str] = []
    for length in lengths:
        sequences.extend(
            "".join(bits) for bits in itertools.product("HT", repeat=length)
        )
    return [
        {"sequence_a": seq_a, "sequence_b": seq_b}
        for seq_a, seq_b in itertools.combinations(sequences, 2)
    ]


def binary_entropy(p: float) -> float:
    """Binary entropy in bits; 0 at the endpoints ``p in {0, 1}``."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def model_discrimination_eig(
    stimulus: Mapping[str, str],
    predict_fns: Mapping[str, PredictFn],
    *,
    model_weights: Optional[Mapping[str, float]] = None,
) -> float:
    """Expected information (bits) about model identity from one response.

    With models ``{m}`` (prior weights ``w_m``, uniform by default) each
    predicting ``p_m = P(choose left | stimulus, m)``, the response ``R`` is
    ``Bernoulli(p̄)`` with ``p̄ = Σ_m w_m p_m``, and the mutual information
    between model identity and ``R`` is::

        I = H(p̄) − Σ_m w_m H(p_m)

    (``H`` = binary entropy). It is 0 when all models agree and grows as their
    predictions diverge — exactly the quantity an experiment should maximize to
    tell the models apart.
    """
    names = list(predict_fns)
    if not names:
        raise ValueError("Need at least one model to score discrimination.")
    if model_weights is None:
        weights = {n: 1.0 / len(names) for n in names}
    else:
        total = sum(model_weights.get(n, 0.0) for n in names)
        if total <= 0:
            raise ValueError("Model weights must sum to a positive value.")
        weights = {n: model_weights.get(n, 0.0) / total for n in names}
    p = {n: float(predict_fns[n](stimulus)) for n in names}
    p_bar = sum(weights[n] * p[n] for n in names)
    return binary_entropy(p_bar) - sum(weights[n] * binary_entropy(p[n]) for n in names)


def rank_stimuli(
    stimuli: Sequence[Mapping[str, Any]],
    predict_fns: Mapping[str, PredictFn],
    *,
    model_weights: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """Annotate each stimulus with ``discrimination_eig``, sorted descending."""
    scored = [
        {
            **dict(stim),
            "discrimination_eig": model_discrimination_eig(
                stim, predict_fns, model_weights=model_weights
            ),
        }
        for stim in stimuli
    ]
    return sorted(scored, key=lambda s: s["discrimination_eig"], reverse=True)


def select_discriminating_stimuli(
    stimuli: Sequence[Mapping[str, Any]],
    predict_fns: Mapping[str, PredictFn],
    k: int,
    *,
    model_weights: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, Any]]:
    """The ``k`` most model-discriminating stimuli (annotated, sorted).

    Greedy top-``k`` by per-stimulus information. This ignores redundancy
    between chosen items (two high-scoring stimuli may probe the same
    distinction); for a first design pass that is usually fine, but a set that
    spreads across distinctions can beat the naive top-``k``.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if not stimuli:
        raise ValueError("No candidate stimuli to select from.")
    return rank_stimuli(stimuli, predict_fns, model_weights=model_weights)[:k]


def _predict_matrix(
    stimuli: Sequence[Mapping[str, Any]],
    predict_fns: Mapping[str, PredictFn],
) -> "np.ndarray":
    """(M, K) matrix of P(choose left) for each stimulus under each model."""
    names = list(predict_fns)
    P = np.array(
        [[float(predict_fns[n](stim)) for n in names] for stim in stimuli],
        dtype=float,
    )
    return np.clip(P, 1e-6, 1.0 - 1e-6)


def _marginal_eig(P: "np.ndarray", weights: "np.ndarray") -> "np.ndarray":
    """Per-stimulus EIG about model identity, ``H(p̄) − Σ w_k H(p_k)`` (vectorized)."""

    def h(p):  # binary entropy in bits, elementwise, safe at the endpoints
        return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))

    p_bar = P @ weights
    return h(p_bar) - (h(P) * weights).sum(axis=1)


def _weight_vector(
    names: Sequence[str], model_weights: Optional[Mapping[str, float]]
) -> "np.ndarray":
    """Normalized prior/posterior weight over models (uniform when unspecified)."""
    if model_weights is None:
        return np.full(len(names), 1.0 / len(names))
    raw = np.array([model_weights.get(n, 0.0) for n in names], dtype=float)
    if raw.sum() <= 0:
        raise ValueError("Model weights must sum to a positive value.")
    return raw / raw.sum()


def _greedy_select_indices(
    P_pool: "np.ndarray",
    w: "np.ndarray",
    k: int,
    *,
    n_scenarios: int = 512,
    seed: int = 0,
    lazy: bool = False,
    lazy_audit: bool = False,
) -> List[int]:
    """The greedy-selection core of :func:`select_informative_stimuli`, factored
    out so callers that already have a pool's ``(pool_size, n_models)``
    probability matrix (e.g. ``build_exhaustive_design``'s fast path, which
    computes it via ``exhaustive_search.pair_probabilities`` instead of scalar
    ``predict_fns`` calls) can run the same Monte Carlo greedy without paying to
    rebuild ``P`` through ``_predict_matrix``. Returns indices into ``P_pool``'s
    rows, in selection order -- exactly what ``select_informative_stimuli``'s own
    ``chosen`` list held before this extraction.
    """
    if k > len(P_pool):
        raise ValueError(f"Requested k={k} but only {len(P_pool)} candidates in the pool.")
    logP = np.log(P_pool)
    log1mP = np.log(1.0 - P_pool)

    rng = np.random.default_rng(seed)
    true_model = rng.choice(len(w), size=n_scenarios, p=w)  # (N,)
    unif = rng.random((n_scenarios, len(P_pool)))  # (N, Mp), common random numbers
    p_true = P_pool[:, true_model].T  # (N, Mp): true model's p_left per scenario/candidate
    responses = (unif < p_true).astype(float)  # (N, Mp)

    log_belief = np.tile(np.log(w), (n_scenarios, 1))  # (N, K), belief over M per scenario
    remaining = list(range(len(P_pool)))
    if lazy:
        return _celf_select(log_belief, responses, logP, log1mP, remaining, k, audit=lazy_audit)

    chosen: List[int] = []
    for _ in range(k):
        rem = np.array(remaining)
        mean_ent = _mean_posterior_entropy(log_belief, responses, logP, log1mP, rem)
        best = remaining[int(np.argmin(mean_ent))]
        chosen.append(best)
        remaining.remove(best)
        log_belief = log_belief + (
            responses[:, best][:, None] * logP[best]
            + (1.0 - responses[:, best])[:, None] * log1mP[best]
        )
    return chosen


def select_informative_stimuli(
    stimuli: Sequence[Mapping[str, Any]],
    predict_fns: Mapping[str, PredictFn],
    k: int,
    *,
    model_weights: Optional[Mapping[str, float]] = None,
    n_scenarios: int = 512,
    prefilter: int = 2000,
    seed: int = 0,
    lazy: bool = False,
    lazy_audit: bool = False,
) -> List[Dict[str, Any]]:
    """Greedily select ``k`` stimuli that jointly tell the models apart.

    Approximates the (intractable) most-informative set of ``k`` pairs: it
    maximizes the mutual information between the *set* of responses and which
    model is correct, ``I(R_S; M)``. Because each added stimulus has diminishing
    returns (a redundant probe of an already-resolved distinction adds little),
    this objective is monotone submodular, so greedy selection enjoys the
    standard ``1 − 1/e`` guarantee and naturally spreads across distinctions
    rather than doubling up like top-``k`` (:func:`select_discriminating_stimuli`).

    ``I(R_S; M)`` itself is intractable (a sum over ``2**k`` response patterns),
    so the expected posterior entropy over models is estimated by Monte Carlo:
    ``n_scenarios`` scenarios each draw a "true" model from ``model_weights`` and
    Bernoulli responses from its ``p_left``. Common random numbers (one fixed draw
    per stimulus, reused across greedy steps) keep the marginal gains consistent
    and the result deterministic given ``seed``.

    To stay fast on an exhaustive candidate pool, only the top ``prefilter``
    stimuli by marginal EIG are considered for the joint selection (a stimulus
    with ~zero marginal information cannot help any set). Annotates each returned
    stimulus with ``eig`` (marginal) and ``selection_order``.

    ``lazy=True`` selects via CELF (lazy greedy, :func:`_celf_select`) instead of
    the full-scan greedy loop: for the *population* mutual-information objective,
    submodularity guarantees this returns the identical sequence, in a fraction of
    the evaluations (only round 0 scores every remaining candidate; later rounds
    mostly re-rank stale upper bounds instead of rescoring everything).

    For the *finite-``n_scenarios``* Monte Carlo estimator actually used here,
    submodularity is only approximate: CELF's correctness depends on a
    candidate's estimated gain never increasing as the selected set grows, and
    sampling noise can violate that for two closely-competing candidates (a
    "close but not tied" gap, not just a bitwise tie). Measured on small pools
    (candidate lengths 3-5, k=8): CELF disagreed with the full-scan greedy on
    5/15 random trials at ``n_scenarios=300``, 1/15 at 512 (this function's
    default), and 0/15 at 8000 — real but shrinking as ``n_scenarios`` grows, not
    a rare bit-level curiosity like ``top_pairs_by_marginal_eig``'s draw-order
    sensitivity. Because of this, ``lazy`` defaults to ``False``: CELF is an
    explicit, informed opt-in for callers who want the speed and can accept the
    accuracy trade-off (e.g. by raising ``n_scenarios`` to compensate), not a
    silent default. ``lazy_audit=True`` additionally runs the full scan alongside
    CELF every round and raises on the first disagreement (for tests; adds back
    the full per-round cost, so it is not meant for production use).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}.")
    if not stimuli:
        raise ValueError("No candidate stimuli to select from.")
    names = list(predict_fns)
    if not names:
        raise ValueError("Need at least one model to score discrimination.")
    if k > len(stimuli):
        raise ValueError(f"Requested k={k} but only {len(stimuli)} stimuli available.")

    w = _weight_vector(names, model_weights)

    P = _predict_matrix(stimuli, predict_fns)  # (M, K)
    marg = _marginal_eig(P, w)  # (M,)

    # Prefilter to the most marginally informative candidates (cheap), then run
    # the joint greedy on that pool. Keep at least k.
    pool_size = min(len(stimuli), max(int(prefilter), k))
    pool_idx = np.argsort(-marg)[:pool_size]
    Pp = P[pool_idx]  # (Mp, K)

    chosen = _greedy_select_indices(
        Pp, w, k, n_scenarios=n_scenarios, seed=seed, lazy=lazy, lazy_audit=lazy_audit
    )

    out: List[Dict[str, Any]] = []
    for order, local in enumerate(chosen):
        global_idx = int(pool_idx[local])
        out.append(
            {
                **dict(stimuli[global_idx]),
                # "eig" is the per-stimulus marginal information (matches the
                # stimuli.json contract used by the agent/eig.py design path);
                # "selection_order" is this stimulus's rank in the greedy set.
                "eig": round(float(marg[global_idx]), 6),
                "selection_order": order,
            }
        )
    return out


def _posterior_entropy(log_belief: "np.ndarray") -> "np.ndarray":
    """Entropy (nats) of the softmax of ``log_belief`` over its last axis."""
    shifted = log_belief - log_belief.max(axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs /= probs.sum(axis=-1, keepdims=True)
    return -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=-1)


def _mean_posterior_entropy(
    log_belief: "np.ndarray",
    responses: "np.ndarray",
    logP: "np.ndarray",
    log1mP: "np.ndarray",
    idx: "np.ndarray",
    *,
    chunk: int = 256,
) -> "np.ndarray":
    """Mean (over scenarios) posterior entropy after tentatively adding each
    candidate in ``idx``, chunked over the candidate axis to bound peak memory —
    the unchunked version materializes an ``(n_scenarios, len(idx), n_models)``
    tensor (plus several same-shaped temporaries inside :func:`_posterior_entropy`)
    all at once, which is what drove ``select_informative_stimuli``'s peak memory
    (measured 319MB on a 3000-candidate pool at ``n_scenarios=512``).

    Bit-identical to the unchunked computation, not merely close: each
    candidate's tentative belief and entropy depends only on its own column of
    ``responses``/``logP``/``log1mP``, so chunking the candidate axis never
    reassociates a floating-point sum — unlike chunking over parameter draws
    (see ``exhaustive_search.pair_probabilities``), which does. The one change
    from the original expression, ``np.where(r != 0, logP, log1mP)`` in place of
    ``r*logP + (1-r)*log1mP``, is also exact rather than approximate: ``responses``
    is built from a boolean comparison cast to float
    (``(unif < p_true).astype(float)``), so ``r`` is always exactly ``0.0`` or
    ``1.0``, and IEEE754 guarantees ``1.0*a + 0.0*b == a`` and ``0.0*a + 1.0*b ==
    b`` exactly (``log1mP``/``logP`` are always finite — ``P`` is clipped away
    from 0 and 1 in :func:`_predict_matrix`), so the two expressions select the
    identical stored float either way.
    """
    out = np.empty(len(idx), dtype=np.float64)
    for start in range(0, len(idx), chunk):
        cols = idx[start : start + chunk]
        r = responses[:, cols]  # (N, C)
        contrib = np.where(r[:, :, None] != 0.0, logP[cols][None], log1mP[cols][None])
        tentative = log_belief[:, None, :] + contrib  # (N, C, K)
        ent = _posterior_entropy(tentative)  # (N, C)
        out[start : start + chunk] = ent.mean(axis=0)
    return out


def _celf_select(
    log_belief: "np.ndarray",
    responses: "np.ndarray",
    logP: "np.ndarray",
    log1mP: "np.ndarray",
    remaining: List[int],
    k: int,
    *,
    audit: bool = False,
) -> List[int]:
    """Lazy-greedy (CELF) selection of ``k`` indices from ``remaining``.

    Exploits that each candidate's marginal *gain* (current mean posterior
    entropy minus its tentative mean posterior entropy) can only shrink as the
    selected set grows — so a gain computed in an earlier round is a valid upper
    bound on that candidate's true current gain, and a max-heap of (possibly
    stale) gains lets most rounds skip rescoring every remaining candidate: round
    0 scores everyone once; every later round pops the heap, and only rescores an
    entry when it turns out to be stale (its stamp doesn't match the current
    round), pushing the refreshed gain back and continuing until the popped entry
    is confirmed still on top after a refresh.

    This selects the exact same sequence :func:`select_informative_stimuli`'s
    full-scan loop does whenever the objective is exactly submodular. Here it is
    only *estimated* by ``n_scenarios`` Monte Carlo draws, so submodularity holds
    up to sampling noise — see the module-level test suite for how a genuine
    near-tie can (rarely) resolve differently, exactly analogous to
    ``exhaustive_search.top_pairs_by_marginal_eig``'s documented draw-order
    sensitivity.
    """
    log_belief = log_belief.copy()
    remaining_arr = np.array(remaining, dtype=np.int64)
    current_entropy = float(_posterior_entropy(log_belief).mean())
    initial_ent = _mean_posterior_entropy(log_belief, responses, logP, log1mP, remaining_arr)

    heap = [
        (-(current_entropy - float(initial_ent[pos])), int(c), 0)
        for pos, c in enumerate(remaining_arr)
    ]
    heapq.heapify(heap)

    chosen: List[int] = []
    for round_idx in range(k):
        while True:
            neg_gain, c, stamp = heapq.heappop(heap)
            if stamp == round_idx:
                if audit:
                    rem = np.array([r for r in remaining if r not in chosen], dtype=np.int64)
                    full_ent = _mean_posterior_entropy(
                        log_belief, responses, logP, log1mP, rem
                    )
                    true_best = int(rem[int(np.argmin(full_ent))])
                    if true_best != c:
                        raise AssertionError(
                            f"CELF disagreed with the full-scan greedy at round "
                            f"{round_idx}: CELF picked {c}, full scan picked "
                            f"{true_best}."
                        )
                chosen.append(c)
                break
            fresh_ent = float(
                _mean_posterior_entropy(log_belief, responses, logP, log1mP, np.array([c]))[0]
            )
            fresh_gain = current_entropy - fresh_ent
            heapq.heappush(heap, (-fresh_gain, c, round_idx))
        log_belief = log_belief + (
            responses[:, c][:, None] * logP[c] + (1.0 - responses[:, c])[:, None] * log1mP[c]
        )
        current_entropy = float(_posterior_entropy(log_belief).mean())
    return chosen


def build_exhaustive_design(
    k: int = 32,
    *,
    lengths: Sequence[int] = (2, 3, 4, 5, 6, 7, 8),
    model_names: Optional[Sequence[str]] = None,
    model_weights: Optional[Mapping[str, float]] = None,
    param_samples: Optional[int] = 200,
    param_sets_by_model: Optional[Mapping[str, Sequence[Mapping[str, float]]]] = None,
    n_scenarios: int = 512,
    prefilter: int = 3000,
    seed: int = 0,
    max_length: int = 20,
    lazy: bool = True,
    on_quotient_violation: str = "fallback",
    legacy_compat: bool = False,
) -> List[Dict[str, Any]]:
    """Select ``k`` jointly-informative pairs from the *full* H/T pair space.

    Enumerates every distinct unordered pair over the given ``lengths``, scores
    them under the pure-Python reference families (the synced twins of the PyMC
    seed models), and greedily picks a diverse ``k`` via
    :func:`select_informative_stimuli` — replacing an agent's hand-written
    candidate pool with a principled, reproducible design over the whole space.

    Predictions account for parameter uncertainty: by default ``p_left`` is
    averaged over ``param_samples`` prior draws (experiment 1). Pass
    ``param_sets_by_model`` (e.g. posterior draws from a prior experiment's fit)
    and ``model_weights`` (posterior model probabilities) to design later
    experiments under the current posterior instead of the prior.

    The default path never materializes the pair space or rescoring it one
    ``predict_left`` call at a time:

    1. Enumerate every sequence, quotient it into feature-equivalence classes
       (``sequence_stats.build_sequence_classes``) using the *union* of the
       model set's own declared ``SUFFICIENT_STATS``
       (``exhaustive_search.quotient_stat_names`` — the full 11-statistic
       superset if any model lacks a declaration, so an undeclared model is
       never over-merged), additionally folding H<->T complements when every
       model unanimously declares ``COMPLEMENT_INVARIANT``. The quotient is
       then audited (``exhaustive_search.audit_quotient``) against every
       model's actual ``score_sequence`` rather than trusted outright; see
       ``on_quotient_violation``.
    2. Score every class representative once at each model's ``DEFAULT_PARAMS``
       (``exhaustive_search.build_score_table``) and scan every class pair for
       marginal EIG in tiles, without ever materializing the full pair list
       (``exhaustive_search.top_pairs_by_marginal_eig``), to prefilter to the
       top ``prefilter`` pairs.
    3. Re-score only the classes that prefiltered pool actually touches — at
       most ``2 * prefilter`` classes, independent of ``lengths`` — averaged
       over ``param_samples``/``param_sets_by_model`` draws, then run the same
       greedy joint-information selection (:func:`_greedy_select_indices`) the
       legacy path used, directly on the resulting probability matrix.

    ``lazy`` (default ``True`` here, unlike :func:`select_informative_stimuli`'s
    own conservative default) selects CELF for the step-3 greedy. Measured: at
    this function's production defaults, the full-scan greedy alone (every other
    speedup applied) still costs ~3.7s of a ~4-5s total — CELF is what actually
    gets the whole default pipeline down near the target ~0.5s at
    ``lengths=(2..8)``. That is traded against the real (not merely
    theoretical) MC-estimator divergence documented on
    ``select_informative_stimuli`` — pass ``lazy=False`` here to keep every other
    speedup but fall back to the exact full-scan greedy.

    ``legacy_compat=True`` instead runs the original algorithm exactly as
    written before this pipeline existed — no quotienting, the original
    shared-RNG-stream prior-draw generation (every model's prior draws come off
    an identical ``np.random.default_rng(seed)`` stream, correlating their
    shared parameters like ``beta``/``side_bias`` — a known quirk, fixed
    separately and later since fixing it changes output), and the plain
    full-scan greedy (``lazy`` is ignored) — kept specifically so a golden test
    can pin that the fast path above is a faithful acceleration, not a
    different computation. It does not scale the way the default path does
    (materializes every pair up front) and is not meant for production use at
    ``lengths`` beyond the historical default.

    ``on_quotient_violation`` controls what happens if the audit in step 1 finds
    a model that actually distinguishes two sequences the quotient merged
    (``exhaustive_search.QuotientViolation``): ``"fallback"`` (default) prints a
    warning and rebuilds with the identity quotient (no merging at all --
    ``sequence_stats.identity_classes``), so an outer-loop run in progress
    degrades to a slower-but-correct design instead of crashing; ``"raise"``
    propagates the error.
    """
    if model_names:
        names = list(model_names)
    elif legacy_compat:
        names = list(_LEGACY_COMPAT_MODEL_FAMILIES)
    else:
        names = default_model_family_names()

    if legacy_compat:
        candidates = enumerate_all_pairs(lengths)
        point_P = _predict_matrix(candidates, family_predict_fns(names))
        w = _weight_vector(names, model_weights)
        pool_size = min(len(candidates), max(int(prefilter), k))
        pool_idx = np.argsort(-_marginal_eig(point_P, w))[:pool_size]
        pool = [candidates[int(i)] for i in pool_idx]
        scoring_fns = family_predict_fns(
            names,
            param_samples=param_samples,
            param_sets_by_model=param_sets_by_model,
            seed=seed,
        )
        return select_informative_stimuli(
            pool,
            scoring_fns,
            k,
            model_weights=model_weights,
            n_scenarios=n_scenarios,
            prefilter=pool_size,
            seed=seed,
            lazy=False,
        )

    from . import exhaustive_search as es  # local import: avoids a module cycle
    from . import sequence_stats as ss

    modules = [
        importlib.import_module(f"src.subjective_randomness.model_families.{name}")
        for name in names
    ]
    w = _weight_vector(names, model_weights)

    stat_names = es.quotient_stat_names(modules)
    complement_canonical = es.complement_invariant(modules)
    classes = ss.build_sequence_classes(
        lengths,
        stat_names=stat_names,
        complement_canonical=complement_canonical,
        max_length=max_length,
        seed=seed,
    )
    try:
        es.audit_quotient(classes, modules, seed=seed)
    except es.QuotientViolation as exc:
        if on_quotient_violation == "raise":
            raise
        if on_quotient_violation != "fallback":
            raise ValueError(
                f"Unknown on_quotient_violation={on_quotient_violation!r}; "
                "expected 'fallback' or 'raise'."
            ) from exc
        print(
            f"  [design] quotient audit failed ({exc}); falling back to the "
            "identity quotient (no merging) for this design.",
            flush=True,
        )
        classes = ss.identity_classes(lengths, max_length=max_length)

    # Stage 1 — cheap point-parameter scores prefilter the class-pair space by
    # marginal EIG, streamed in tiles (never materializes the full pair list).
    point_table = es.build_score_table(modules, classes.representatives)
    pool_size = max(int(prefilter), k)
    i_idx, j_idx, _ = es.top_pairs_by_marginal_eig(point_table, w, pool_size)
    if len(i_idx) < k:
        # top_pairs_by_marginal_eig silently clips to however many distinct
        # class-pairs actually exist -- a narrow quotient (e.g. a model set that
        # collapses everything to very few classes) can make that smaller than
        # k. select_informative_stimuli raises rather than silently returning
        # fewer than requested; match that here instead of letting the greedy
        # loop crash on an exhausted pool.
        raise ValueError(
            f"Only {len(i_idx)} distinct class-pair(s) available after "
            f"quotienting {classes.n_sequences} sequences to {classes.n_classes} "
            f"classes over lengths {tuple(sorted(set(lengths)))}; cannot select "
            f"k={k}. Widen `lengths`, or pass on_quotient_violation aside -- this "
            "is the quotient legitimately having too little to work with, not a "
            "quotient-safety issue."
        )

    # Stage 2 — accurate (parameter-averaged) scores on only the classes the
    # prefiltered pool touches (<= 2*prefilter, independent of `lengths`).
    used = np.unique(np.concatenate([i_idx, j_idx]))
    used_seqs = [classes.representatives[int(u)] for u in used]
    remap = {int(g): local for local, g in enumerate(used)}
    table = es.build_score_table(
        modules,
        used_seqs,
        param_sets_by_model=param_sets_by_model,
        param_samples=param_samples if param_sets_by_model is None else None,
        seed=seed,
    )
    # n_probe_pairs is lower than audit_decomposition's own default (64): its
    # cost scales with n_probe_pairs * n_models * n_draws, and at this
    # function's production draw counts (param_samples up to a few hundred) the
    # default probe count is measurably not free (~0.7s). A systematic
    # decomposition bug (the only thing this check can catch -- see its
    # docstring) shows up on essentially any probed pair, so a smaller sample
    # still catches it while keeping this a genuinely cheap safety net here.
    es.audit_decomposition(table, modules, used_seqs, seed=seed, n_probe_pairs=16)

    i_local = np.array([remap[int(v)] for v in i_idx])
    j_local = np.array([remap[int(v)] for v in j_idx])
    P_pool = es.pair_probabilities(table, i_local, j_local)  # (pool_size, K)

    chosen = _greedy_select_indices(P_pool, w, k, n_scenarios=n_scenarios, seed=seed, lazy=lazy)

    marg_pool = _marginal_eig(P_pool, w)
    out: List[Dict[str, Any]] = []
    for order, local in enumerate(chosen):
        out.append(
            {
                "sequence_a": used_seqs[i_local[local]],
                "sequence_b": used_seqs[j_local[local]],
                "eig": round(float(marg_pool[local]), 6),
                "selection_order": order,
            }
        )
    return out


def default_model_family_names() -> List[str]:
    """Active seed-model names from the recovery registry manifest.

    The manifest (``pymc_model_families/models_manifest.yaml``) is the single
    source of truth for the active seed set. Superseded family modules stay
    importable for archival refits but are deliberately NOT picked up here —
    enumerating the package directory would resurrect them.
    """
    return read_manifest_names(REGISTRY_DIR)


def _point_predictor(module: Any) -> PredictFn:
    def predict(stimulus: Mapping[str, str]) -> float:
        return float(module.predict_left(stimulus, module.DEFAULT_PARAMS))

    return predict


def _average_predictor(
    module: Any, param_sets: Sequence[Mapping[str, float]]
) -> PredictFn:
    """A predictor that averages ``p_left`` over the given parameter sets."""
    sets = [dict(p) for p in param_sets]
    if not sets:
        raise ValueError("Need at least one parameter set to average over.")

    def predict(stimulus: Mapping[str, str]) -> float:
        return float(np.mean([module.predict_left(stimulus, params) for params in sets]))

    return predict


def _prior_param_sets(module: Any, n_samples: int, seed: int) -> List[Dict[str, float]]:
    """``n_samples`` parameter draws from the family's (uniform) ``PARAM_BOUNDS``.

    For these families ``PARAM_BOUNDS`` matches the PyMC seed model's uniform
    priors exactly, so averaging over these draws is the faithful prior predictive.
    """
    rng = np.random.default_rng(seed)
    return [
        {name: float(rng.uniform(lo, hi)) for name, (lo, hi) in module.PARAM_BOUNDS.items()}
        for _ in range(n_samples)
    ]


def posterior_param_sets(
    idata: Any,
    param_names: Sequence[str],
    *,
    n_draws: int = 256,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Subsample ``n_draws`` parameter sets from a fitted model's posterior.

    Reads ``idata.posterior[name]`` for each ``param_names`` (the family's free
    parameters, whose names match the PyMC model's random variables), flattens
    over (chain, draw), and draws ``n_draws`` joint samples (same index across
    parameters, preserving their posterior correlation). Fails loudly if the
    posterior lacks a requested variable.
    """
    posterior = idata.posterior
    columns: Dict[str, "np.ndarray"] = {}
    for name in param_names:
        if name not in posterior:
            raise KeyError(
                f"Posterior has no variable {name!r}; available: {list(posterior.data_vars)}"
            )
        columns[name] = np.asarray(posterior[name].values).reshape(-1)
    n_total = len(next(iter(columns.values()))) if columns else 0
    if n_total == 0:
        raise ValueError("Posterior is empty; cannot draw parameter sets.")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_total, size=min(n_draws, n_total))
    return [{name: float(columns[name][i]) for name in param_names} for i in idx]


def family_predict_fns(
    model_names: Sequence[str],
    *,
    param_samples: Optional[int] = None,
    param_sets_by_model: Optional[Mapping[str, Sequence[Mapping[str, float]]]] = None,
    seed: int = 0,
) -> Dict[str, PredictFn]:
    """Build ``p_left`` predictors from the pure-Python model families.

    Prediction mode, in priority order:

    - ``param_sets_by_model``: average ``p_left`` over the explicit parameter sets
      for each model (e.g. posterior draws from a previous experiment's fit).
    - ``param_samples=N``: average over ``N`` prior draws from each family's
      ``PARAM_BOUNDS`` — the prior predictive (reflects parameter uncertainty).
    - neither: predict at each family's ``DEFAULT_PARAMS`` (a point prediction).

    Deterministic given ``seed``.
    """
    fns: Dict[str, PredictFn] = {}
    for name in model_names:
        module = importlib.import_module(
            f"src.subjective_randomness.model_families.{name}"
        )
        if param_sets_by_model is not None:
            fns[name] = _average_predictor(module, param_sets_by_model[name])
        elif param_samples is not None:
            fns[name] = _average_predictor(module, _prior_param_sets(module, param_samples, seed))
        else:
            fns[name] = _point_predictor(module)
    return fns
