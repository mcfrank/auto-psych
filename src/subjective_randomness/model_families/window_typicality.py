"""Finite-window longest-run typicality model of subjective randomness.

Loosely inspired by Hahn & Warren (2009), "Perceptions of randomness: Why three
heads are better than four" (Psychological Review, 116(2), 454-461): because
human experience of coin flips is finite and filtered through limited short-term
memory, long runs are genuinely rare in experience, so streak aversion and a
preference for moderate over-alternation are apt rather than erroneous.

FIDELITY NOTE: this model is a heuristic adaptation, not an implementation of
Hahn & Warren's analysis. Their formal quantity is the probability that a given
substring occurs at least once within a finite global sequence of ~10-20
experienced flips (implemented faithfully in the
``finite_experience_occurrence`` family). The longest-run-typicality score
below — comparing the observed longest run against a log2 approximation to the
expected longest run — comes from the longest-run literature (e.g. Schilling,
1990), not from Hahn & Warren, and orders some sequences differently than their
occurrence probabilities do (e.g. their analysis gives HHTT and HHHT identical
wait times; this score does not).

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
