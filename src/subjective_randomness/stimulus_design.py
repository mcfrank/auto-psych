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


def _prior_predictive_predictor(module: Any, n_samples: int, seed: int) -> PredictFn:
    bounds = module.PARAM_BOUNDS
    rng = np.random.default_rng(seed)
    param_sets = [
        {name: float(rng.uniform(lo, hi)) for name, (lo, hi) in bounds.items()}
        for _ in range(n_samples)
    ]

    def predict(stimulus: Mapping[str, str]) -> float:
        return float(
            np.mean([module.predict_left(stimulus, params) for params in param_sets])
        )

    return predict


def family_predict_fns(
    model_names: Sequence[str],
    *,
    param_samples: Optional[int] = None,
    seed: int = 0,
) -> Dict[str, PredictFn]:
    """Build ``p_left`` predictors from the pure-Python model families.

    With ``param_samples=None`` each family predicts at its ``DEFAULT_PARAMS``
    (a point prediction). With ``param_samples=N``, ``p_left`` is averaged over
    ``N`` parameter draws from the family's ``PARAM_BOUNDS`` — a cheap
    prior-predictive that reflects parameter uncertainty. Deterministic given
    ``seed``.
    """
    fns: Dict[str, PredictFn] = {}
    for name in model_names:
        module = importlib.import_module(
            f"src.subjective_randomness.model_families.{name}"
        )
        if param_samples is None:
            fns[name] = _point_predictor(module)
        else:
            fns[name] = _prior_predictive_predictor(module, param_samples, seed)
    return fns
