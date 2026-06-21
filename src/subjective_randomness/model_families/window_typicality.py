"""Hahn & Warren (2009) finite-window model of subjective randomness.

"Perceptions of randomness: Why three heads are better than one" (Psychological
Review, 116(2), 454-461). People experience sequences through a limited memory
window of length ``window``; a sequence looks random when its longest run is
*typical* of what a fair coin produces within that window, and non-random when a
salient streak exceeds the window-expected longest run. Because long runs are
rarely encountered inside a short window, this rational account reproduces both
the aversion to streaks and the preference for over-alternation.

The expected longest run of a fair coin over an effective length
``m = min(n, window)`` is approximated by ``log2(m)``. Randomness is penalised
asymmetrically around that expectation:

    score = -(softplus(max_run - e) + over_alt_penalty * softplus(e - max_run))
    e = log2(min(n, window))

The first term penalises runs *longer* than expected (the Hahn & Warren effect);
the second, scaled by ``over_alt_penalty``, optionally penalises runs *shorter*
than expected (over-alternation). With over_alt_penalty = 0 only long runs are
penalised; the data decide how much, if at all, over-alternation also counts.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

from .common import (
    Stimulus,
    choice_probability,
    distribution,
    max_run_length,
    merge_params,
    normalize_stimulus,
    softplus,
)

MODEL_NAME = "window_typicality"

DEFAULT_PARAMS: Dict[str, float] = {
    # Effective memory window over which the longest run is judged.
    "window": 5.0,
    # How much an unexpectedly short longest run (over-alternation) also reduces
    # perceived randomness. Non-trivial: a window that rarely contains long runs
    # also rarely contains none at all, so perfect alternation reads as atypical
    # (the over-alternation aversion). Free, but defaults to a moderate value.
    "over_alt_penalty": 0.60,
    "beta": 4.0,
    "side_bias": 0.0,
}

PARAM_BOUNDS: Dict[str, tuple[float, float]] = {
    "window": (2.0, 8.0),
    "over_alt_penalty": (0.0, 1.0),
    "beta": (0.2, 12.0),
    "side_bias": (-2.0, 2.0),
}


def score_sequence(seq: str, params: Mapping[str, float] | None = None) -> float:
    p = merge_params(DEFAULT_PARAMS, params)
    window = float(p["window"])
    over_alt_penalty = float(p["over_alt_penalty"])

    n = len(seq)
    expected_run = math.log2(min(float(n), window))
    max_run = float(max_run_length(seq))
    too_long = softplus(max_run - expected_run)
    too_short = softplus(expected_run - max_run)
    return -(too_long + over_alt_penalty * too_short)


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
