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

import importlib
import itertools
import math
import pkgutil
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

# A predictor maps a stimulus ({"sequence_a", "sequence_b"}) to P(choose left).
PredictFn = Callable[[Mapping[str, str]], float]


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
    lengths_cycle = list(lengths)
    while len(pool) < n_pairs:
        length = lengths_cycle[len(pool) % len(lengths_cycle)]
        seqs = sequences_by_length[length]
        i, j = rng.integers(0, len(seqs), size=2)
        if i == j:
            continue
        key = (seqs[i], seqs[j]) if i < j else (seqs[j], seqs[i])
        if key in seen:
            continue
        seen.add(key)
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


def select_informative_stimuli(
    stimuli: Sequence[Mapping[str, Any]],
    predict_fns: Mapping[str, PredictFn],
    k: int,
    *,
    model_weights: Optional[Mapping[str, float]] = None,
    n_scenarios: int = 512,
    prefilter: int = 2000,
    seed: int = 0,
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
    logP = np.log(Pp)
    log1mP = np.log(1.0 - Pp)

    rng = np.random.default_rng(seed)
    true_model = rng.choice(len(names), size=n_scenarios, p=w)  # (N,)
    unif = rng.random((n_scenarios, len(pool_idx)))  # (N, Mp), common random numbers
    p_true = Pp[:, true_model].T  # (N, Mp): true model's p_left per scenario/candidate
    responses = (unif < p_true).astype(float)  # (N, Mp)

    log_belief = np.tile(np.log(w), (n_scenarios, 1))  # (N, K), belief over M per scenario
    remaining = list(range(len(pool_idx)))
    chosen: List[int] = []
    for _ in range(k):
        rem = np.array(remaining)
        # Tentative log-belief if each remaining candidate were added: (N, R, K).
        r = responses[:, rem]  # (N, R)
        contrib = r[:, :, None] * logP[rem][None] + (1.0 - r)[:, :, None] * log1mP[rem][None]
        tentative = log_belief[:, None, :] + contrib  # (N, R, K)
        ent = _posterior_entropy(tentative)  # (N, R)
        best = remaining[int(np.argmin(ent.mean(axis=0)))]
        chosen.append(best)
        remaining.remove(best)
        log_belief = log_belief + (
            responses[:, best][:, None] * logP[best]
            + (1.0 - responses[:, best])[:, None] * log1mP[best]
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
) -> List[Dict[str, Any]]:
    """Select ``k`` jointly-informative pairs from the *full* H/T pair space.

    Enumerates every distinct unordered pair over the given ``lengths``
    (:func:`enumerate_all_pairs`), scores them under the pure-Python reference
    families (the synced twins of the PyMC seed models), and greedily picks a
    diverse ``k`` via :func:`select_informative_stimuli` — replacing an agent's
    hand-written candidate pool with a principled, reproducible design over the
    whole space.

    Predictions account for parameter uncertainty: by default ``p_left`` is
    averaged over ``param_samples`` prior draws (experiment 1). Pass
    ``param_sets_by_model`` (e.g. posterior draws from a prior experiment's fit)
    and ``model_weights`` (posterior model probabilities) to design later
    experiments under the current posterior instead of the prior.

    Two-stage for speed: cheap point predictions prefilter the (~129k) pool to the
    top ``prefilter`` by marginal EIG, then the expensive parameter-averaged
    predictions and greedy joint selection run only on that pool.
    """
    names = list(model_names) if model_names else default_model_family_names()
    candidates = enumerate_all_pairs(lengths)

    # Stage 1 — cheap point predictions to prefilter the full pool by marginal EIG.
    point_P = _predict_matrix(candidates, family_predict_fns(names))
    w = _weight_vector(names, model_weights)
    pool_size = min(len(candidates), max(int(prefilter), k))
    pool_idx = np.argsort(-_marginal_eig(point_P, w))[:pool_size]
    pool = [candidates[int(i)] for i in pool_idx]

    # Stage 2 — accurate (parameter-averaged) predictions on the pool only, then
    # the greedy joint-information selection (prefilter is now a no-op).
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
    )


def default_model_family_names() -> List[str]:
    """Names of the pure-Python reference model families (excludes ``common``)."""
    import src.subjective_randomness.model_families as families_pkg

    return sorted(
        module.name
        for module in pkgutil.iter_modules(families_pkg.__path__)
        if module.name != "common"
    )


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
