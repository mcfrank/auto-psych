"""Griffiths et al. (2018) motif-HMM model of subjective randomness.

"Subjective randomness as statistical inference" (Griffiths, Daniels,
Austerweil & Tenenbaum, Cognitive Psychology 103:85-109). A sequence looks
random to the extent it is better evidence for a fair coin than for a regular
generator:

    randomness(x) = log P(x | fair) - log P(x | regular),  P(x | fair) = (1/2)^n

P(x | regular) is their six-state motif HMM (their Eq. 7-9): four motifs —
repeating heads, repeating tails, repeating heads-tails, repeating tails-heads
— where each state emits one symbol, the current motif continues with
per-symbol probability δ, and switching to a new motif has probability
proportional to α^k with k the motif length (so complex motifs are
penalised). Faithful to the published model in the two respects the older
``bayesian_diagnosticity`` seed approximates away:

  * P(x | regular) **marginalises over all parses** (their Eq. 8) via the HMM
    forward pass — no fixed best parse.
  * The transition matrix is **row-normalised** (their footnote 11, the form
    used from their Experiment 1 onward) rather than carrying the improper
    C = (1-δ)/(2α+2α²) "null state" mass.

States (0-indexed): 0=H-repeat, 1=T-repeat, 2=HT-motif emitting H,
3=HT-motif emitting T, 4=TH-motif emitting H, 5=TH-motif emitting T; even
states emit H, odd states emit T. Initial state distribution is proportional
to (α, α, α², 0, 0, α²) — motif entries only.

The score is not length-normalised (evidence accumulates with length; the
paper defends this against raw DP). Free parameters: δ, α, choice ``beta``
(default 1.0 — randomness differences are in nats and larger defaults
saturate the choice rule), and ``side_bias``.
"""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence

from .common import (
    Stimulus,
    choice_probability,
    clean_sequence,
    clipped,
    distribution,
    merge_params,
    normalize_stimulus,
)

MODEL_NAME = "motif_hmm"

N_STATES = 6
# Even states emit H, odd states emit T.
_EMITS = "HTHTHT"

DEFAULT_PARAMS: Dict[str, float] = {
    # Near the paper's Experiment 2 best fit (delta=0.549, alpha=0.207).
    "delta": 0.55,
    "alpha": 0.21,
    "beta": 1.0,
    "side_bias": 0.0,
}

PARAM_BOUNDS: Dict[str, tuple[float, float]] = {
    "delta": (0.01, 0.99),
    "alpha": (0.01, 0.99),
    "beta": (0.2, 12.0),
    "side_bias": (-2.0, 2.0),
}


def _matrices(delta: float, alpha: float) -> tuple[List[float], List[List[float]]]:
    """Row-normalised transition matrix and initial vector (Eq. 9 + fn. 11)."""
    a, a2, d = alpha, alpha * alpha, delta
    rows = [
        [d, a, a2, 0.0, 0.0, a2],
        [a, d, a2, 0.0, 0.0, a2],
        [a, a, 0.0, d, 0.0, a2],
        [a, a, d, 0.0, 0.0, a2],
        [a, a, a2, 0.0, 0.0, d],
        [a, a, a2, 0.0, d, 0.0],
    ]
    transition = [[v / sum(row) for v in row] for row in rows]
    init_raw = [a, a, a2, 0.0, 0.0, a2]
    z = sum(init_raw)
    init = [v / z for v in init_raw]
    return init, transition


def log_p_regular(seq: str, delta: float, alpha: float) -> float:
    """log P(seq | motif HMM), marginalised over hidden state paths (Eq. 8)."""
    s = clean_sequence(seq)
    delta = clipped(delta)
    alpha = clipped(alpha)
    init, transition = _matrices(delta, alpha)

    forward = [init[j] if _EMITS[j] == s[0] else 0.0 for j in range(N_STATES)]
    for c in s[1:]:
        forward = [
            sum(forward[i] * transition[i][j] for i in range(N_STATES))
            if _EMITS[j] == c
            else 0.0
            for j in range(N_STATES)
        ]
    total = sum(forward)
    if total <= 0.0:
        raise ValueError(
            f"motif HMM assigned zero probability to {s!r} "
            f"(delta={delta}, alpha={alpha}); this should be impossible"
        )
    return math.log(total)


def score_sequence(seq: str, params: Mapping[str, float] | None = None) -> float:
    """randomness(x) = log P(x | fair) - log P(x | motif HMM)."""
    p = merge_params(DEFAULT_PARAMS, params)
    s = clean_sequence(seq)
    log_fair = len(s) * math.log(0.5)
    return log_fair - log_p_regular(s, float(p["delta"]), float(p["alpha"]))


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
