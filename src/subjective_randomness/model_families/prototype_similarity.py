"""Prototype/similarity model for subjective randomness.

This model treats judged randomness as distance from an internal prototype:
roughly balanced heads/tails and an expected alternation rate.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

from .common import (
    Stimulus,
    alternation_rate,
    choice_probability,
    distribution,
    imbalance,
    merge_params,
    normalize_stimulus,
)

MODEL_NAME = "prototype_similarity"

DEFAULT_PARAMS: Dict[str, float] = {
    # The ideal alternation rate. Values above 0.5 allow human overalternation.
    "theta_alt": 0.65,
    # Feature-weight simplex: balance_weight = 1 - alt_weight.
    "alt_weight": 0.55,
    "beta": 4.0,
    "side_bias": 0.0,
}

PARAM_BOUNDS: Dict[str, tuple[float, float]] = {
    "theta_alt": (0.35, 0.95),
    "alt_weight": (0.01, 0.99),
    "beta": (0.2, 12.0),
    "side_bias": (-2.0, 2.0),
}

# Declarative only -- read by exhaustive_search.quotient_stat_names /
# complement_invariant to build a safe (never over-merging) equivalence quotient
# over the sequence space for design-time search. score_sequence above reads the
# sequence only through imbalance() and alternation_rate(); alternation_rate is an
# exact linear rescaling of the alts/(n-1) statistic, hence "p_alts" here (see
# sequence_stats.CANONICAL_STAT_NAMES). Verified (tests/test_sequence_stats.py):
# no two sequences agreeing on these ever get different scores.
SUFFICIENT_STATS: tuple[str, ...] = ("imbalance", "p_alts")

# imbalance() and alternation_rate() are both provably invariant under H<->T
# complementation (imbalance = 2|h/n-0.5| is symmetric in h<->n-h; alternation_rate
# depends only on the run-transition pattern, not which symbol is which).
COMPLEMENT_INVARIANT: bool = True


def score_sequence(seq: str, params: Mapping[str, float] | None = None) -> float:
    p = merge_params(DEFAULT_PARAMS, params)
    alt_weight = max(0.0, min(1.0, p["alt_weight"]))
    balance_weight = 1.0 - alt_weight
    balance_distance = imbalance(seq)
    alternation_distance = abs(alternation_rate(seq) - p["theta_alt"])
    return -(balance_weight * balance_distance + alt_weight * alternation_distance)


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
