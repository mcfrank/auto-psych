"""Bayesian diagnosticity model for subjective randomness.

This is a small, fit-able version of the Tenenbaum & Griffiths style account:
a sequence looks random when it is better evidence for a fair/random generator
than for salient alternatives such as alternating, biased, or streaky generators.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

from .common import (
    Stimulus,
    bernoulli_log_prob,
    choice_probability,
    clipped,
    distribution,
    logsumexp,
    merge_params,
    n_switches,
    normalize_stimulus,
)

MODEL_NAME = "bayesian_diagnosticity"

DEFAULT_PARAMS: Dict[str, float] = {
    # Stick-breaking alternative weights:
    # pi_alt = alt_prior
    # pi_bias = (1 - alt_prior) * bias_share
    # pi_streak = (1 - alt_prior) * (1 - bias_share)
    "alt_prior": 0.34,
    "bias_share": 0.50,
    "beta": 4.0,
    "side_bias": 0.0,
}

PARAM_BOUNDS: Dict[str, tuple[float, float]] = {
    "alt_prior": (0.01, 0.98),
    "bias_share": (0.01, 0.99),
    "beta": (0.2, 12.0),
    "side_bias": (-2.0, 2.0),
}

_ALT_SWITCH_PROB = 0.95
_STREAK_SWITCH_PROB = 0.15
_BIAS_HEAD_PROB = 0.85


def alternative_weights(params: Mapping[str, float]) -> Dict[str, float]:
    """Return mixture weights over non-random alternatives."""
    alt = clipped(float(params.get("alt_prior", DEFAULT_PARAMS["alt_prior"])))
    bias_share = clipped(float(params.get("bias_share", DEFAULT_PARAMS["bias_share"])))
    remaining = 1.0 - alt
    return {
        "alternating": alt,
        "biased": remaining * bias_share,
        "streaky": remaining * (1.0 - bias_share),
    }


def _iid_log_prob(seq: str, p_heads: float) -> float:
    heads = sum(1 for c in seq if c == "H")
    tails = len(seq) - heads
    return bernoulli_log_prob(heads, tails, p_heads)


def _markov_log_prob(seq: str, switch_prob: float) -> float:
    if len(seq) <= 1:
        return math.log(0.5)
    switches = n_switches(seq)
    stays = (len(seq) - 1) - switches
    return math.log(0.5) + bernoulli_log_prob(switches, stays, switch_prob)


def _length_normalized(log_prob: float, seq: str) -> float:
    return log_prob / max(1, len(seq))


def score_sequence(seq: str, params: Mapping[str, float] | None = None) -> float:
    """
    Diagnosticity for the fair-random generator over non-random alternatives.

    Scores are length-normalized so mixed-length comparisons are not dominated by
    raw probability mass.
    """
    p = merge_params(DEFAULT_PARAMS, params)
    weights = alternative_weights(p)

    fair = _length_normalized(_iid_log_prob(seq, 0.5), seq)
    alt = _length_normalized(_markov_log_prob(seq, _ALT_SWITCH_PROB), seq)
    streak = _length_normalized(_markov_log_prob(seq, _STREAK_SWITCH_PROB), seq)
    bias_h = _length_normalized(_iid_log_prob(seq, _BIAS_HEAD_PROB), seq)
    bias_t = _length_normalized(_iid_log_prob(seq, 1.0 - _BIAS_HEAD_PROB), seq)
    biased = logsumexp([math.log(0.5) + bias_h, math.log(0.5) + bias_t])

    alternatives = logsumexp(
        [
            math.log(weights["alternating"]) + alt,
            math.log(weights["biased"]) + biased,
            math.log(weights["streaky"]) + streak,
        ]
    )
    return fair - alternatives


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
