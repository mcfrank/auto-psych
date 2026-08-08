"""Hahn & Warren (2009) finite-experience occurrence model of randomness.

"Perceptions of randomness: Why three heads are better than four"
(Psychological Review, 116(2), 454-461). Human experience of coin flips is
finite: people see a global data stream of limited length through a moving
short-term-memory window. The probability that a specific string of length k
occurs *at least once* within a global sequence of n fair flips is therefore
not equal across equal-length strings — e.g. within n = 20 flips HHHT occurs
with probability ~.75 but HHHH only ~.48 (their footnote 1 / p. 458). A
string seems random to the extent that it is the kind of thing one actually
encounters when watching a fair coin:

    score = log( sum_N w_N * P(occurs at least once in N flips) )

with the global experience length N uncertain over {10, 20, 50} (20 is the
paper's focal, ecologically motivated illustration; 10 and 50 bracket it) and
stick-breaking weights: w_10 = short_weight, w_20 = (1 - short_weight) *
mid_share, w_50 = the remainder. This reproduces both signatures of the
account: aversion to long runs AND aversion to perfect alternation (HTHT has
the second-longest wait time of the length-4 strings, 20 vs 30 for HHHH).

CROSS-LENGTH CAVEAT: the paper only ever compares equal-length strings inside
a common window. Comparing sequences of different lengths by raw occurrence
probability — which mechanically favours shorter strings — is this model's
auxiliary assumption, not Hahn & Warren's.

Free parameters: short_weight, mid_share (the experience-length prior), and
the choice rule's beta / side_bias. The paper itself is a normative analysis
with no fitted parameters; the weights are the minimal cognitive addition
needed to make it a choice model.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

from .common import (
    Stimulus,
    choice_probability,
    distribution,
    merge_params,
    normalize_stimulus,
    occurrence_probability,
)

MODEL_NAME = "finite_experience_occurrence"

# Global experienced-sequence lengths; mirrors features.EXPERIENCE_LENGTHS.
EXPERIENCE_LENGTHS = (10, 20, 50)

DEFAULT_PARAMS: Dict[str, float] = {
    # Stick-breaking prior over the experience length N:
    # w_10 = short_weight; w_20 = (1 - short_weight) * mid_share; w_50 = rest.
    # Defaults put most mass on the paper's focal N = 20.
    "short_weight": 0.15,
    "mid_share": 0.70,
    "beta": 4.0,
    "side_bias": 0.0,
}

PARAM_BOUNDS: Dict[str, tuple[float, float]] = {
    "short_weight": (0.01, 0.99),
    "mid_share": (0.01, 0.99),
    "beta": (0.2, 12.0),
    "side_bias": (-2.0, 2.0),
}


def experience_weights(params: Mapping[str, float]) -> Dict[int, float]:
    short = max(0.0, min(1.0, float(params["short_weight"])))
    mid_share = max(0.0, min(1.0, float(params["mid_share"])))
    w10 = short
    w20 = (1.0 - short) * mid_share
    w50 = (1.0 - short) * (1.0 - mid_share)
    return {10: w10, 20: w20, 50: w50}


def score_sequence(seq: str, params: Mapping[str, float] | None = None) -> float:
    p = merge_params(DEFAULT_PARAMS, params)
    weights = experience_weights(p)
    expected_occurrence = sum(
        weights[w] * occurrence_probability(seq, w) for w in EXPERIENCE_LENGTHS
    )
    if expected_occurrence <= 0.0:
        raise ValueError(
            f"occurrence probability underflowed to zero for {seq!r}; "
            f"weights={weights}"
        )
    return math.log(expected_occurrence)


def predict_left(
    stimulus: Stimulus | Mapping[str, str], params: Mapping[str, float] | None = None
) -> float:
    seq_a, seq_b = normalize_stimulus(stimulus)
    p = merge_params(DEFAULT_PARAMS, params)
    return choice_probability(score_sequence(seq_a, p), score_sequence(seq_b, p), p)


def predict(
    stimulus: Stimulus | Mapping[str, str],
    response_options: Sequence[str] = ("left", "right"),
    params: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    return distribution(predict_left(stimulus, params), response_options)
