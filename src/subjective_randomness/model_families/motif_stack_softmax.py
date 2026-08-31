"""Softmax (marginalising) variant of the four-motif stack-automaton model.

This is :mod:`.motif_stack` with both of its maxes replaced by their
temperature-1 softmax (log-sum-exp) counterparts.  The Viterbi model scores the
regular hypothesis by the single most probable hidden path and production
method::

    P(x | regular) = max_{z, M} P(x, z | M) P(M)

The softmax variant marginalises instead of maximising::

    P(x | regular) = sum_{z, M} P(x, z | M) P(M)
                   = sum_M P(M) * (forward probability of x under method M)

Concretely, two substitutions:

* the Viterbi max-product recursion over hidden states becomes the forward
  sum-product recursion — ``sum`` over the previous state instead of ``max``;
* the ``max`` over the four production methods becomes their mixture — a plain
  sum of ``P(M) * P(x | M)`` over the methods x can have been produced by.

Because a sum dominates a max, ``P(x | regular)`` here is never below the
Viterbi value, so the randomness score ``log P(x|fair) - log P(x|regular)`` is
never above it.  Unlike the Viterbi score, this one *is* a normalised
distribution over strings of a given length (the forward algorithm and the
method mixture both sum to one over length-n strings), so no length-specific
normaliser is missing.  The same-length restriction on comparisons is kept
regardless, matching the Viterbi twin and reflecting that the model was
estimated on fixed-length stimuli.

The motivation is purely computational: the maxes give the Viterbi
log-likelihood non-differentiable ridges where the argmax switches, which is
what makes it awkward for NUTS.  The marginal likelihood is smooth in the
parameters everywhere, so this variant exists to test whether it is easier to
fit.  The parameter names, bounds, and defaults are inherited unchanged from the
Viterbi model — the defaults are its Griffiths et al. estimates, not re-fit to
the marginal likelihood, and serve only as valid points for data generation and
equivalence testing.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

from .common import (
    Stimulus,
    choice_probability,
    clean_sequence,
    clipped,
    distribution,
    merge_params,
    normalize_stimulus,
)

MODEL_NAME = "motif_stack_softmax"
N_STATES = 6
_EMITS = "HTHTHT"

DEFAULT_PARAMS: Dict[str, float] = {
    "delta": 0.5493,
    "alpha": 0.2073,
    # Inherited from the Viterbi model's Experiment 2 estimates (normalised
    # because the rounded preprint values do not sum to one). Not re-fit to the
    # marginal likelihood — see the module docstring.
    "repetition_weight": 0.6839,
    "mirror_share": 0.4989,
    "complement_share": 0.3984,
    "beta": 1.0,
    "side_bias": 0.0,
}

PARAM_BOUNDS: Dict[str, tuple[float, float]] = {
    "delta": (0.01, 0.99),
    "alpha": (0.01, 0.99),
    "repetition_weight": (0.001, 0.999),
    "mirror_share": (0.001, 0.999),
    "complement_share": (0.001, 0.999),
    "beta": (0.2, 12.0),
    "side_bias": (-2.0, 2.0),
}


def _matrices(delta: float, alpha: float) -> tuple[List[float], List[List[float]]]:
    a, a2, d = clipped(alpha), clipped(alpha) ** 2, clipped(delta)
    rows = [
        [d, a, a2, 0.0, 0.0, a2],
        [a, d, a2, 0.0, 0.0, a2],
        [a, a, 0.0, d, 0.0, a2],
        [a, a, d, 0.0, 0.0, a2],
        [a, a, a2, 0.0, 0.0, d],
        [a, a, a2, 0.0, d, 0.0],
    ]
    transition = [[value / sum(row) for value in row] for row in rows]
    init_raw = [a, a, a2, 0.0, 0.0, a2]
    init_total = sum(init_raw)
    return [value / init_total for value in init_raw], transition


def _forward_prefix_probabilities(
    seq: str, delta: float, alpha: float
) -> List[float]:
    """Total hidden-path probability after each observed prefix (forward pass).

    The sum-product counterpart of :mod:`.motif_stack`'s Viterbi recursion:
    ``sum`` over the previous state where the Viterbi version takes ``max``, so
    each entry is the marginal probability of the observed prefix rather than the
    single best path's joint probability.
    """
    init, transition = _matrices(delta, alpha)
    forward = [
        init[state] if _EMITS[state] == seq[0] else 0.0
        for state in range(N_STATES)
    ]
    prefix_probabilities = [sum(forward)]
    for symbol in seq[1:]:
        forward = [
            sum(
                forward[previous] * transition[previous][state]
                for previous in range(N_STATES)
            )
            if _EMITS[state] == symbol
            else 0.0
            for state in range(N_STATES)
        ]
        prefix_probabilities.append(sum(forward))
    return prefix_probabilities


def _method_weights(params: Mapping[str, float]) -> Dict[str, float]:
    repetition = clipped(float(params["repetition_weight"]))
    mirror_share = clipped(float(params["mirror_share"]))
    complement_share = clipped(float(params["complement_share"]))
    remaining = 1.0 - repetition
    mirror = remaining * mirror_share
    remaining_after_mirror = remaining * (1.0 - mirror_share)
    complement = remaining_after_mirror * complement_share
    duplication = remaining_after_mirror * (1.0 - complement_share)
    return {
        "repetition": repetition,
        "mirror": mirror,
        "complement": complement,
        "duplication": duplication,
    }


def _memory_patterns(seq: str) -> Dict[str, bool]:
    prefix_length = (len(seq) + 1) // 2
    prefix = seq[:prefix_length]
    mirrored_source = prefix[:-1] if len(seq) % 2 else prefix
    suffix = seq[prefix_length:]
    complement = {"H": "T", "T": "H"}
    return {
        "mirror": suffix == mirrored_source[::-1],
        "complement": suffix
        == "".join(complement[symbol] for symbol in mirrored_source[::-1]),
        "duplication": len(seq) % 2 == 0 and suffix == prefix,
    }


def log_p_regular(
    seq: str, params: Mapping[str, float] | None = None
) -> float:
    p = merge_params(DEFAULT_PARAMS, params)
    s = clean_sequence(seq)
    prefixes = _forward_prefix_probabilities(
        s, float(p["delta"]), float(p["alpha"])
    )
    weights = _method_weights(p)
    method_probabilities = [weights["repetition"] * prefixes[-1]]
    prefix_probability = prefixes[(len(s) + 1) // 2 - 1]
    for method, matches in _memory_patterns(s).items():
        if matches:
            method_probabilities.append(weights[method] * prefix_probability)
    # Mixture over production methods (softmax over M) rather than the Viterbi
    # argmax: sum P(M) P(x | M) over the methods x can have been produced by.
    regular_probability = sum(method_probabilities)
    if regular_probability <= 0.0:
        raise ValueError(f"Motif stack assigned zero probability to {s!r}")
    return math.log(regular_probability)


def score_sequence(
    seq: str, params: Mapping[str, float] | None = None
) -> float:
    s = clean_sequence(seq)
    return len(s) * math.log(0.5) - log_p_regular(s, params)


def predict_left(
    stimulus: Stimulus | Mapping[str, str], params: Mapping[str, float] | None = None
) -> float:
    seq_a, seq_b = normalize_stimulus(stimulus)
    if len(seq_a) != len(seq_b):
        raise ValueError(
            "motif_stack_softmax is defined only for same-length sequence "
            f"comparisons; got lengths {len(seq_a)} and {len(seq_b)}"
        )
    p = merge_params(DEFAULT_PARAMS, params)
    return choice_probability(score_sequence(seq_a, p), score_sequence(seq_b, p), p)


def predict(
    stimulus: Stimulus | Mapping[str, str],
    response_options: Sequence[str] = ("left", "right"),
    params: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    return distribution(predict_left(stimulus, params), response_options)
