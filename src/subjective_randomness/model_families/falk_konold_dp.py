"""Falk & Konold (1997) Difficulty Predictor model of subjective randomness.

"Making sense of randomness: Implicit encoding as a basis for judgment"
(Psychological Review, 104(2), 301-318). A sequence seems random to the extent
that it is hard to encode mentally. Their Difficulty Predictor (p. 308) scores
a sequence by partitioning it into pure runs (weight 1) and alternating runs
(weight 2), choosing the partition that "achieve[s] the lowest possible
number":

    DP = (number of pure runs) + 2 * (number of alternating runs)
    score = DP  (higher DP = harder encoding = more random-seeming)

DP is deliberately *not* length-normalised — Falk & Konold never divide by n.
In their data mean DP correlates .95 with mean apparent-randomness ratings and
.96-.99 with memorization difficulty. The parse comes from ``parse_motifs``
(minimal-DP partition; ties broken toward fewest chunks).

Free parameters: only the choice-rule ``beta`` and ``side_bias`` — the theory
itself has no free cognitive parameters. ``beta`` defaults to 1.0 because DP
differences are integer-scale; larger defaults saturate the choice rule.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

from .common import (
    Stimulus,
    choice_probability,
    distribution,
    merge_params,
    normalize_stimulus,
    parse_motifs,
)

MODEL_NAME = "falk_konold_dp"

DEFAULT_PARAMS: Dict[str, float] = {
    "beta": 1.0,
    "side_bias": 0.0,
}

PARAM_BOUNDS: Dict[str, tuple[float, float]] = {
    "beta": (0.2, 12.0),
    "side_bias": (-2.0, 2.0),
}


def score_sequence(seq: str, params: Mapping[str, float] | None = None) -> float:
    rep_motifs, alt_motifs = parse_motifs(seq)
    return float(rep_motifs + 2 * alt_motifs)


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
