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

    score = log P(occurs at least once in 20 flips)

Twenty flips is the paper's focal finite stream. This reproduces both
signatures of the account: aversion to long runs and aversion to perfect
alternation (HTHT has the second-longest wait time of the length-4 strings,
20 versus 30 for HHHH). The paper only compares equal-length strings inside a
common stream, so the choice function rejects cross-length comparisons.

Free parameters: only the choice rule's beta and side bias. The paper itself
is a normative analysis and supplies no fitted cognitive parameters.
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

EXPERIENCE_LENGTH = 20

DEFAULT_PARAMS: Dict[str, float] = {
    "beta": 4.0,
    "side_bias": 0.0,
}

PARAM_BOUNDS: Dict[str, tuple[float, float]] = {
    "beta": (0.2, 12.0),
    "side_bias": (-2.0, 2.0),
}


def score_sequence(seq: str, params: Mapping[str, float] | None = None) -> float:
    occurrence = occurrence_probability(seq, EXPERIENCE_LENGTH)
    if occurrence <= 0.0:
        raise ValueError(
            f"occurrence probability underflowed to zero for {seq!r}; "
            f"experience_length={EXPERIENCE_LENGTH}"
        )
    return math.log(occurrence)


def predict_left(
    stimulus: Stimulus | Mapping[str, str], params: Mapping[str, float] | None = None
) -> float:
    seq_a, seq_b = normalize_stimulus(stimulus)
    if len(seq_a) != len(seq_b):
        raise ValueError(
            "finite_experience_occurrence is defined only for same-length "
            f"sequence comparisons; got lengths {len(seq_a)} and {len(seq_b)}"
        )
    p = merge_params(DEFAULT_PARAMS, params)
    return choice_probability(score_sequence(seq_a, p), score_sequence(seq_b, p), p)


def predict(
    stimulus: Stimulus | Mapping[str, str],
    response_options: Sequence[str] = ("left", "right"),
    params: Mapping[str, float] | None = None,
) -> Dict[str, float]:
    return distribution(predict_left(stimulus, params), response_options)
