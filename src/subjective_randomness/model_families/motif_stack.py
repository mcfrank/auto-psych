"""Griffiths et al. (2018) four-motif stack-automaton model.

The paper's best model for simultaneously presented binary sequences augments
the four-motif finite-state process with three memory-based production methods:
mirror symmetry, complement symmetry, and duplication.  Following the paper's
definition, the regular-process score is the probability of the most likely
hidden path and production method::

    P(x | regular) = max_{z, M} P(x, z | M) P(M)

The repetition method generates the whole sequence with the motif HMM.  Each
memory method generates the first half with that HMM and deterministically
affixes, respectively, its reversal, complemented reversal, or duplicate.
For odd lengths the middle symbol belongs to the generated prefix and is not
repeated; that convention is an explicit extension beyond the paper's
length-eight stimuli.

The returned randomness score is ``log P(x|fair) - log P(x|regular)``.  The
Viterbi score is not a normalized distribution over strings, but its missing
normalizer cancels in the same-length pairwise comparisons for which this
family is defined.
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

MODEL_NAME = "motif_stack"
N_STATES = 6
_EMITS = "HTHTHT"

DEFAULT_PARAMS: Dict[str, float] = {
    "delta": 0.5493,
    "alpha": 0.2073,
    # Griffiths et al.'s Experiment 2 four-motif stack estimates, normalized
    # because the rounded values printed in the preprint do not sum to one.
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


def _viterbi_prefix_probabilities(
    seq: str, delta: float, alpha: float
) -> List[float]:
    """Maximum hidden-path joint probability after each observed prefix."""
    init, transition = _matrices(delta, alpha)
    best = [
        init[state] if _EMITS[state] == seq[0] else 0.0
        for state in range(N_STATES)
    ]
    prefix_probabilities = [max(best)]
    for symbol in seq[1:]:
        best = [
            max(
                best[previous] * transition[previous][state]
                for previous in range(N_STATES)
            )
            if _EMITS[state] == symbol
            else 0.0
            for state in range(N_STATES)
        ]
        prefix_probabilities.append(max(best))
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
    prefixes = _viterbi_prefix_probabilities(
        s, float(p["delta"]), float(p["alpha"])
    )
    weights = _method_weights(p)
    method_probabilities = [weights["repetition"] * prefixes[-1]]
    prefix_probability = prefixes[(len(s) + 1) // 2 - 1]
    for method, matches in _memory_patterns(s).items():
        if matches:
            method_probabilities.append(weights[method] * prefix_probability)
    regular_probability = max(method_probabilities)
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
            "motif_stack is defined only for same-length sequence comparisons; "
            f"got lengths {len(seq_a)} and {len(seq_b)}"
        )
    p = merge_params(DEFAULT_PARAMS, params)
    return choice_probability(score_sequence(seq_a, p), score_sequence(seq_b, p), p)


def predict(
    stimulus: Stimulus | Mapping[str, str],
    response_options: Sequence[str] = ("left", "right"),
    params: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    return distribution(predict_left(stimulus, params), response_options)
