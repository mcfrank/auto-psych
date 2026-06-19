"""Statistical-inference model of subjective randomness.

Pure-Python twin of the ``statistical_inference`` PyMC seed model. A sequence's
randomness is the log-likelihood ratio between a fair coin and a regular
process (Griffiths, Daniels, Austerweil & Tenenbaum, 2018):

    randomness(x) = log P(x | random) - log P(x | regular)
                  = n*log(0.5) - log P(x | regular)

with the regular process approximated at the canonical motif parse (n1
repetition motifs, n2 alternation motifs) by the closed form of Section 6.1:

    log P(x | regular) = (n - n1 - n2)*log δ + (n1 + n2)*log C + (n1 + 2*n2)*log α

where C = (1-δ)/(2α + 2α²), δ is motif persistence, and α penalizes motif
complexity. Falk & Konold's Difficulty Predictor DP = n1 + 2*n2 is the special
case carried by the α exponent. The score is *not* length-normalized — length
sensitivity is a property of the model.
"""

from __future__ import annotations

import math
from typing import Dict, Mapping, Sequence

from .common import (
    Stimulus,
    choice_probability,
    clipped,
    distribution,
    merge_params,
    normalize_stimulus,
    parse_motifs,
)

MODEL_NAME = "statistical_inference"

DEFAULT_PARAMS: Dict[str, float] = {
    "delta": 0.5,
    "alpha": 0.35,  # near (sqrt(3)-1)/2, the DP-equivalent value from the paper
    "beta": 4.0,
    "side_bias": 0.0,
}

PARAM_BOUNDS: Dict[str, tuple[float, float]] = {
    "delta": (0.01, 0.99),
    "alpha": (0.01, 0.99),
    "beta": (0.2, 12.0),
    "side_bias": (-2.0, 2.0),
}


def score_sequence(seq: str, params: Mapping[str, float] | None = None) -> float:
    """Randomness = log P(seq | fair coin) - log P(seq | regular process)."""
    p = merge_params(DEFAULT_PARAMS, params)
    delta = clipped(float(p["delta"]))
    alpha = clipped(float(p["alpha"]))

    n = len(seq)
    rep_motifs, alt_motifs = parse_motifs(seq)
    stays = n - rep_motifs - alt_motifs  # within-motif continuations; >= 0

    log_c = math.log(1.0 - delta) - math.log(2.0 * alpha + 2.0 * alpha**2)
    log_regular = (
        stays * math.log(delta)
        + (rep_motifs + alt_motifs) * log_c
        + (rep_motifs + 2 * alt_motifs) * math.log(alpha)
    )
    return n * math.log(0.5) - log_regular


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
