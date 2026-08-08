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


def softplus(x: float) -> float:
    """Numerically stable ``log(1 + exp(x))``."""
    if x > 30.0:
        return x
    return math.log1p(math.exp(x))


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

    Returns ``(rep_motifs, alt_motifs)`` — n1 (repetition motifs: constant-run
    chunks) and n2 (alternation motifs: strictly alternating chunks of length
    >= 2) of the Difficulty Predictor parse, for which DP = n1 + 2*n2. Falk &
    Konold define the parse as the partition minimising DP (boundaries may
    split runs, e.g. HTHHTH -> HTH|HTH); DP ties are broken toward the fewest
    chunks, making (n1, n2) unique. Mirrors the featurizer helper of the same
    name in ``features.py``.
    """
    s = clean_sequence(seq)
    n = len(s)

    # best[i] = lexicographically minimal (DP cost, chunk count) over all
    # partitions of s[:i] into constant-run chunks (cost 1) and strictly
    # alternating chunks of length >= 2 (cost 2).
    unreachable = (n * 2 + 1, n + 1)
    best = [(0, 0)] + [unreachable] * n
    for i in range(1, n + 1):
        for j in range(i - 1, -1, -1):
            chunk = s[j:i]
            if all(c == chunk[0] for c in chunk):
                cost = 1
            elif all(a != b for a, b in zip(chunk, chunk[1:])):
                cost = 2
            else:
                continue
            candidate = (best[j][0] + cost, best[j][1] + 1)
            if candidate < best[i]:
                best[i] = candidate
    dp, chunks = best[n]
    rep_motifs = 2 * chunks - dp
    alt_motifs = dp - chunks
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


LOCAL_WINDOW = 4


def local_imbalance(seq: str) -> float:
    """Worst H/T imbalance over sliding windows of length min(n, LOCAL_WINDOW).

    2*|prop_heads - 0.5| of the most imbalanced window (Kahneman & Tversky
    1972: representativeness holds "locally in each of its parts"). Mirrors
    the featurizer helper of the same name in ``features.py``.
    """
    s = clean_sequence(seq)
    n = len(s)
    window = min(n, LOCAL_WINDOW)
    worst = 0.0
    for start in range(n - window + 1):
        chunk = s[start : start + window]
        heads = sum(1 for c in chunk if c == "H")
        worst = max(worst, 2.0 * abs(heads / window - 0.5))
    return worst


def occurrence_probability(pattern: str, n_global: int) -> float:
    """P(``pattern`` occurs as a contiguous substring of ``n_global`` fair flips).

    The quantity of Hahn & Warren (2009): the probability that a length-k
    string appears at least once within a finite global sequence of fair coin
    flips. Computed exactly by evolving the distribution over KMP
    prefix-automaton states (state = length of the longest pattern prefix
    matching the current suffix; reaching state k absorbs).
    """
    p = clean_sequence(pattern)
    k = len(p)
    if n_global < 0:
        raise ValueError(f"n_global must be >= 0, got {n_global}")
    if n_global < k:
        return 0.0

    # next_state[state][symbol] for states 0..k-1 via KMP failure links.
    failure = [0] * k
    for i in range(1, k):
        j = failure[i - 1]
        while j > 0 and p[i] != p[j]:
            j = failure[j - 1]
        failure[i] = j + 1 if p[i] == p[j] else 0

    def next_state(state: int, symbol: str) -> int:
        while True:
            if symbol == p[state]:
                return state + 1
            if state == 0:
                return 0
            state = failure[state - 1]

    transitions = [
        {symbol: next_state(state, symbol) for symbol in "HT"} for state in range(k)
    ]

    dist = [0.0] * k
    dist[0] = 1.0
    absorbed = 0.0
    for _ in range(n_global):
        new_dist = [0.0] * k
        for state, mass in enumerate(dist):
            if mass == 0.0:
                continue
            for symbol in "HT":
                target = transitions[state][symbol]
                if target == k:
                    absorbed += 0.5 * mass
                else:
                    new_dist[target] += 0.5 * mass
        dist = new_dist
    return absorbed


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
