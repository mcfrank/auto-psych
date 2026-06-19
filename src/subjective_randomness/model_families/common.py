"""Shared sequence features and choice helpers for subjective-randomness models."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

Stimulus = Tuple[str, str]

_EPS = 1e-9


def normalize_stimulus(stimulus: object) -> Stimulus:
    """Accept a tuple/list or JSON-like dict and return (sequence_a, sequence_b)."""
    if isinstance(stimulus, Mapping):
        return clean_sequence(str(stimulus["sequence_a"])), clean_sequence(
            str(stimulus["sequence_b"])
        )
    if (
        isinstance(stimulus, Sequence)
        and not isinstance(stimulus, (str, bytes))
        and len(stimulus) >= 2
    ):
        return clean_sequence(str(stimulus[0])), clean_sequence(str(stimulus[1]))
    raise ValueError(
        f"Stimulus must be (sequence_a, sequence_b) or a dict; got {type(stimulus)!r}"
    )


def clean_sequence(seq: str) -> str:
    """Uppercase an H/T sequence and reject other symbols."""
    out = "".join(c.upper() for c in seq.strip() if not c.isspace())
    bad = sorted({c for c in out if c not in {"H", "T"}})
    if bad:
        raise ValueError(f"Sequence contains non-H/T symbols: {bad}")
    if not out:
        raise ValueError("Sequence must not be empty")
    return out


def sigmoid(x: float) -> float:
    """Numerically stable logistic function."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def clipped(p: float) -> float:
    return max(_EPS, min(1.0 - _EPS, p))


def distribution(p_left: float, response_options: Sequence[str]) -> Dict[str, float]:
    """Return a response distribution for the current pipeline API."""
    left = response_options[0] if response_options else "left"
    right = response_options[1] if len(response_options) > 1 else "right"
    p = clipped(p_left)
    return {left: p, right: 1.0 - p}


def choice_probability(
    score_left: float, score_right: float, params: Mapping[str, float]
) -> float:
    """Softmax/logistic choice rule for a left-vs-right forced choice."""
    beta = float(params.get("beta", 1.0))
    side_bias = float(params.get("side_bias", 0.0))
    return sigmoid(beta * (score_left - score_right) + side_bias)


def merge_params(
    defaults: Mapping[str, float], params: Mapping[str, float] | None
) -> Dict[str, float]:
    merged = dict(defaults)
    if params:
        merged.update({k: float(v) for k, v in params.items()})
    return merged


def prop_heads(seq: str) -> float:
    seq = clean_sequence(seq)
    return sum(1 for c in seq if c == "H") / len(seq)


def imbalance(seq: str) -> float:
    """Distance from 50/50 heads/tails, scaled to [0, 1]."""
    return 2.0 * abs(prop_heads(seq) - 0.5)


def n_switches(seq: str) -> int:
    seq = clean_sequence(seq)
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def alternation_rate(seq: str) -> float:
    seq = clean_sequence(seq)
    if len(seq) <= 1:
        return 0.0
    return n_switches(seq) / (len(seq) - 1)


def max_run_length(seq: str) -> int:
    seq = clean_sequence(seq)
    best = 1
    cur = 1
    for a, b in zip(seq, seq[1:]):
        if a == b:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def max_run_norm(seq: str) -> float:
    """Maximum run length scaled so alternating sequences are 0 and solid runs are 1."""
    seq = clean_sequence(seq)
    if len(seq) <= 1:
        return 0.0
    return (max_run_length(seq) - 1) / (len(seq) - 1)


def parse_motifs(seq: str) -> Tuple[int, int]:
    """Parse an H/T sequence into Falk & Konold (1997) motifs.

    Returns ``(rep_motifs, alt_motifs)`` — n1 (repetition motifs: maximal
    constant runs) and n2 (alternation motifs: maximal alternating sub-sequences
    of length >= 2) of the canonical minimal-description parse used by the
    statistical-inference model (Griffiths et al. 2018). Mirrors the featurizer
    helper of the same name; DP = n1 + 2*n2.
    """
    s = clean_sequence(seq)
    run_lengths = []
    cur = 1
    for a, b in zip(s, s[1:]):
        if a == b:
            cur += 1
        else:
            run_lengths.append(cur)
            cur = 1
    run_lengths.append(cur)

    rep_motifs = 0
    alt_motifs = 0
    i = 0
    n_runs = len(run_lengths)
    while i < n_runs:
        if run_lengths[i] == 1:
            j = i
            while j < n_runs and run_lengths[j] == 1:
                j += 1
            if j - i >= 2:
                alt_motifs += 1
            else:
                rep_motifs += 1
            i = j
        else:
            rep_motifs += 1
            i += 1
    return rep_motifs, alt_motifs


def periodicity_score(seq: str) -> float:
    """
    Degree to which the sequence can be described by a short repeating template.

    Returns 0 for weak periodicity and approaches 1 for obvious patterns like
    HHHHHHHH or HTHTHTHT.
    """
    seq = clean_sequence(seq)
    n = len(seq)
    if n <= 2:
        return 0.0
    best_match = 0.5
    for period in range(1, (n // 2) + 1):
        template = seq[:period]
        matches = sum(1 for i, c in enumerate(seq) if c == template[i % period])
        best_match = max(best_match, matches / n)
    return max(0.0, min(1.0, 2.0 * (best_match - 0.5)))


def logsumexp(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return -math.inf
    m = max(vals)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in vals))


def bernoulli_log_prob(successes: int, failures: int, p: float) -> float:
    p = clipped(p)
    return successes * math.log(p) + failures * math.log(1.0 - p)
