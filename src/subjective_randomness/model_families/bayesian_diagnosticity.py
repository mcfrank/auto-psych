"""Bayesian diagnosticity model for subjective randomness.

A unified "randomness as statistical inference" account (Griffiths & Tenenbaum,
2001/2003; Griffiths, Daniels, Austerweil & Tenenbaum, 2018): a sequence looks
random to the extent it is better evidence for a fair coin than for a *regular*
(non-random) generator,

    randomness(x) = log P(x | fair) - log P(x | regular)

with P(x | fair) = (1/2)^n. This model merges the two Bayesian seeds that used
to be separate. The regular hypothesis is a mixture of two complementary
non-random generators:

  * a **motif-complexity process** (Griffiths et al. 2018, §6.1), evaluated at
    the canonical minimal-description parse (n1 repetition motifs, n2 alternation
    motifs). For that parse,

        log P(x | motif) = (n - n1 - n2)*log δ + (n1 + n2)*log C + (n1 + 2*n2)*log α

    with C = (1-δ)/(2α + 2α²); δ is motif persistence and α penalises motif
    complexity. This single process subsumes the old "alternating" and "streaky"
    Markov alternatives — long runs are explained by high persistence, regular
    alternation by alternation motifs — and carries Falk & Konold's Difficulty
    Predictor DP = n1 + 2*n2 in the α exponent.
  * a **biased-coin** generator (head- or tail-heavy), which captures the H/T
    imbalance the motif process is blind to.

``bias_share`` is the mixture weight on the biased-coin alternative; the rest
goes to the motif process. The score is *not* length-normalised — accumulating
evidence with sequence length is a property of the Bayesian account.

Free cognitive parameters: δ (motif persistence), α (complexity penalty),
bias_share (weight on the biased-coin alternative), β (choice sensitivity), and
a left/right side bias.
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
    normalize_stimulus,
    parse_motifs,
)

MODEL_NAME = "bayesian_diagnosticity"

DEFAULT_PARAMS: Dict[str, float] = {
    "delta": 0.5,
    "alpha": 0.35,  # near (sqrt(3)-1)/2, the DP-equivalent value from the paper
    "bias_share": 0.30,
    "beta": 4.0,
    "side_bias": 0.0,
}

PARAM_BOUNDS: Dict[str, tuple[float, float]] = {
    "delta": (0.01, 0.99),
    "alpha": (0.01, 0.99),
    "bias_share": (0.01, 0.99),
    "beta": (0.2, 12.0),
    "side_bias": (-2.0, 2.0),
}

_BIAS_HEAD_PROB = 0.85


def _log_motif(n: int, rep_motifs: int, alt_motifs: int, delta: float, alpha: float) -> float:
    """Log P(x | motif process) at the canonical parse (Griffiths et al. 2018)."""
    stays = n - rep_motifs - alt_motifs  # within-motif continuations; >= 0
    log_c = math.log(1.0 - delta) - math.log(2.0 * alpha + 2.0 * alpha**2)
    return (
        stays * math.log(delta)
        + (rep_motifs + alt_motifs) * log_c
        + (rep_motifs + 2 * alt_motifs) * math.log(alpha)
    )


def _log_biased(seq: str) -> float:
    """Log P(x | biased coin), marginalising over head- vs tail-heavy bias."""
    heads = sum(1 for c in seq if c == "H")
    tails = len(seq) - heads
    head_heavy = bernoulli_log_prob(heads, tails, _BIAS_HEAD_PROB)
    tail_heavy = bernoulli_log_prob(heads, tails, 1.0 - _BIAS_HEAD_PROB)
    return logsumexp([math.log(0.5) + head_heavy, math.log(0.5) + tail_heavy])


def score_sequence(seq: str, params: Mapping[str, float] | None = None) -> float:
    """Diagnosticity for the fair coin over the regular-process mixture."""
    p = merge_params(DEFAULT_PARAMS, params)
    delta = clipped(float(p["delta"]))
    alpha = clipped(float(p["alpha"]))
    bias_share = clipped(float(p["bias_share"]))

    n = len(seq)
    rep_motifs, alt_motifs = parse_motifs(seq)

    log_fair = n * math.log(0.5)
    log_motif = _log_motif(n, rep_motifs, alt_motifs, delta, alpha)
    log_biased = _log_biased(seq)
    log_regular = logsumexp(
        [
            math.log(1.0 - bias_share) + log_motif,
            math.log(bias_share) + log_biased,
        ]
    )
    return log_fair - log_regular


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
