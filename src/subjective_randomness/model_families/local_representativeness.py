"""Kahneman & Tversky (1972) local-representativeness model of randomness.

"Subjective probability: A judgment of representativeness" (Cognitive
Psychology, 3, 430-454), p. 435: "a representative sample is one in which the
essential characteristics of the parent population are represented not only
globally in the entire sample, but also locally in each of its parts."

This is a quantitative operationalization rather than an equation supplied by
K&T. It represents their two stated properties separately: local balance and
irregularity. Local imbalance detects clumping; irregularity combines distance
from an over-alternating prototype with a repeating-template penalty, so their
examples HTHTHTHT and TTHHTTHH cannot look random merely because they are
locally balanced.

    structural = (1-periodic_share)*|p_alts-theta_alt| + periodic_share*periodicity
    score = -((1-alt_weight)*local_imbalance + alt_weight*structural)

Free parameters: theta_alt, alt_weight, periodic_share, beta, and side_bias.
The maximum local scale is a fixed design constant; K&T give no numeric value,
so four is an explicit short-term-memory operationalization.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

from .common import (
    Stimulus,
    alternation_rate,
    choice_probability,
    distribution,
    merge_params,
    multiscale_local_imbalance,
    normalize_stimulus,
    periodicity_score,
)

MODEL_NAME = "local_representativeness"

DEFAULT_PARAMS: Dict[str, float] = {
    "theta_alt": 0.65,
    # Higher than prototype_similarity's default: under strict local balance a
    # perfectly alternating sequence is flawless on the balance term, so only
    # the alternation/irregularity term can carry K&T's own example that
    # HTHTHTHT "fail[s] to reflect the randomness of the process" (p. 434).
    "alt_weight": 0.75,
    "periodic_share": 0.45,
    "beta": 4.0,
    "side_bias": 0.0,
}

PARAM_BOUNDS: Dict[str, tuple[float, float]] = {
    "theta_alt": (0.5001, 0.95),
    "alt_weight": (0.01, 0.99),
    "periodic_share": (0.01, 0.99),
    "beta": (0.2, 12.0),
    "side_bias": (-2.0, 2.0),
}


def score_sequence(seq: str, params: Mapping[str, float] | None = None) -> float:
    p = merge_params(DEFAULT_PARAMS, params)
    alt_weight = max(0.0, min(1.0, p["alt_weight"]))
    balance_weight = 1.0 - alt_weight
    balance_distance = multiscale_local_imbalance(seq)
    alternation_distance = abs(alternation_rate(seq) - p["theta_alt"])
    periodic_share = max(0.0, min(1.0, p["periodic_share"]))
    irregularity_distance = (
        (1.0 - periodic_share) * alternation_distance
        + periodic_share * periodicity_score(seq)
    )
    return -(balance_weight * balance_distance + alt_weight * irregularity_distance)


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
