"""Encoding/compressibility model for subjective randomness.

The core idea is that sequences look non-random when they have a short mental
description: all one outcome, perfect alternation, or large obvious chunks.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

from .common import (
    Stimulus,
    choice_probability,
    distribution,
    imbalance,
    max_run_norm,
    merge_params,
    normalize_stimulus,
    periodicity_score,
)

MODEL_NAME = "encoding_compressibility"

DEFAULT_PARAMS: Dict[str, float] = {
    # Stick-breaking feature weights:
    # longrun_weight = longrun_weight
    # periodic_weight = (1 - longrun_weight) * periodic_share
    # imbalance_weight = (1 - longrun_weight) * (1 - periodic_share)
    "longrun_weight": 0.40,
    "periodic_share": 0.50,
    "beta": 4.0,
    "side_bias": 0.0,
}

PARAM_BOUNDS: Dict[str, tuple[float, float]] = {
    "longrun_weight": (0.01, 0.99),
    "periodic_share": (0.01, 0.99),
    "beta": (0.2, 12.0),
    "side_bias": (-2.0, 2.0),
}


def feature_weights(params: Mapping[str, float]) -> Dict[str, float]:
    longrun = max(
        0.0,
        min(1.0, float(params.get("longrun_weight", DEFAULT_PARAMS["longrun_weight"]))),
    )
    periodic_share = max(
        0.0,
        min(1.0, float(params.get("periodic_share", DEFAULT_PARAMS["periodic_share"]))),
    )
    remaining = 1.0 - longrun
    return {
        "longrun": longrun,
        "periodic": remaining * periodic_share,
        "imbalance": remaining * (1.0 - periodic_share),
    }


def score_sequence(seq: str, params: Mapping[str, float] | None = None) -> float:
    p = merge_params(DEFAULT_PARAMS, params)
    weights = feature_weights(p)
    compressibility_penalty = (
        weights["longrun"] * max_run_norm(seq)
        + weights["periodic"] * periodicity_score(seq)
        + weights["imbalance"] * imbalance(seq)
    )
    return -compressibility_penalty


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
