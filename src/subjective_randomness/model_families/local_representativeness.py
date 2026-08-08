"""Kahneman & Tversky (1972) local-representativeness model of randomness.

"Subjective probability: A judgment of representativeness" (Cognitive
Psychology, 3, 430-454), p. 435: "a representative sample is one in which the
essential characteristics of the parent population are represented not only
globally in the entire sample, but also locally in each of its parts."

This is ``prototype_similarity`` with K&T's signature locality restored: the
balance term is the *worst* H/T imbalance over sliding windows of length
min(n, 4) (``local_imbalance``) rather than whole-sequence imbalance, so a
globally balanced but locally clumped sequence (HHHHTTTT) is penalised where
the global model is blind. The alternation term keeps the prototype form —
distance from an ideal alternation rate ``theta_alt``, whose default above .5
reflects the over-alternation K&T predict from local balance ("too many
alternations and too few clusters").

    score = -((1 - alt_weight) * local_imbalance + alt_weight * |p_alts - theta_alt|)

Free parameters: theta_alt, alt_weight, beta, side_bias. The window length is
a fixed design constant (K&T give no numeric value; 4 matches the
short-term-memory motivation used across this literature).
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

from .common import (
    Stimulus,
    alternation_rate,
    choice_probability,
    distribution,
    local_imbalance,
    merge_params,
    normalize_stimulus,
)

MODEL_NAME = "local_representativeness"

DEFAULT_PARAMS: Dict[str, float] = {
    "theta_alt": 0.65,
    # Higher than prototype_similarity's default: under strict local balance a
    # perfectly alternating sequence is flawless on the balance term, so only
    # the alternation/irregularity term can carry K&T's own example that
    # HTHTHTHT "fail[s] to reflect the randomness of the process" (p. 434).
    "alt_weight": 0.75,
    "beta": 4.0,
    "side_bias": 0.0,
}

PARAM_BOUNDS: Dict[str, tuple[float, float]] = {
    "theta_alt": (0.35, 0.95),
    "alt_weight": (0.01, 0.99),
    "beta": (0.2, 12.0),
    "side_bias": (-2.0, 2.0),
}


def score_sequence(seq: str, params: Mapping[str, float] | None = None) -> float:
    p = merge_params(DEFAULT_PARAMS, params)
    alt_weight = max(0.0, min(1.0, p["alt_weight"]))
    balance_weight = 1.0 - alt_weight
    balance_distance = local_imbalance(seq)
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
