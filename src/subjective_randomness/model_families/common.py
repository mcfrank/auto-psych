"""Shared sequence features and choice helpers for subjective-randomness models.

The sequence statistics themselves live in ``src/subjective_randomness/features.py``
— the featurizer that turns raw H/T pairs into the numeric columns the PyMC
models read — and this module wraps them for the pure-Python model families:
each wrapper cleans its input once (``clean_sequence``) and caches on the raw
string. Keeping one implementation means a PyMC model fitted on featurizer
columns and its pure-Python twin can never disagree about what "periodicity" or
"local imbalance" means.
"""

from __future__ import annotations

import functools
import math
from typing import Dict, Iterable, Mapping, Sequence, Tuple

from .. import features
from ..features import LOCAL_WINDOW  # noqa: F401  re-exported for model families

Stimulus = Tuple[str, str]

_EPS = 1e-9

# Every public str -> scalar helper below is pure (same input, same output,
# forever) and is called repeatedly on the same handful of distinct sequences
# across a design run (once per model x per parameter draw). Caching turns that
# fan-out into a cache hit after the first call; measured ~4x on encoding_
# compressibility, the most call-heavy family. lru_cache does not cache
# exceptions, so clean_sequence's loud rejection of non-H/T input still raises
# every time it is given bad input.
_CACHE_SIZE = 1 << 18


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


@functools.lru_cache(maxsize=_CACHE_SIZE)
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


# --- Private helpers below assume an already-cleaned (upper, H/T-only, non-
# empty) string and do not call clean_sequence themselves. Every public
# function cleans exactly once and delegates, instead of each stat re-cleaning
# (e.g. max_run_norm calling max_run_length used to clean twice).


def _head_count(seq: str) -> int:
    return sum(1 for c in seq if c == "H")


def _switch_count(seq: str) -> int:
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b)


def _max_run(seq: str) -> int:
    best = 1
    cur = 1
    for a, b in zip(seq, seq[1:]):
        if a == b:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


@functools.lru_cache(maxsize=_CACHE_SIZE)
def prop_heads(seq: str) -> float:
    seq = clean_sequence(seq)
    return _head_count(seq) / len(seq)


@functools.lru_cache(maxsize=_CACHE_SIZE)
def imbalance(seq: str) -> float:
    """Distance from 50/50 heads/tails, scaled to [0, 1]."""
    seq = clean_sequence(seq)
    return 2.0 * abs(_head_count(seq) / len(seq) - 0.5)


@functools.lru_cache(maxsize=_CACHE_SIZE)
def n_switches(seq: str) -> int:
    seq = clean_sequence(seq)
    return _switch_count(seq)


@functools.lru_cache(maxsize=_CACHE_SIZE)
def alternation_rate(seq: str) -> float:
    seq = clean_sequence(seq)
    if len(seq) <= 1:
        return 0.0
    return _switch_count(seq) / (len(seq) - 1)


@functools.lru_cache(maxsize=_CACHE_SIZE)
def max_run_length(seq: str) -> int:
    seq = clean_sequence(seq)
    return _max_run(seq)


@functools.lru_cache(maxsize=_CACHE_SIZE)
def max_run_norm(seq: str) -> float:
    """Maximum run length scaled so alternating sequences are 0 and solid runs are 1."""
    seq = clean_sequence(seq)
    if len(seq) <= 1:
        return 0.0
    return (_max_run(seq) - 1) / (len(seq) - 1)


@functools.lru_cache(maxsize=_CACHE_SIZE)
def parse_motifs(seq: str) -> Tuple[int, int]:
    """Parse an H/T sequence into Falk & Konold (1997) motifs.

    Returns ``(rep_motifs, alt_motifs)`` — n1 (repetition motifs: constant-run
    chunks) and n2 (alternation motifs: strictly alternating chunks of length
    >= 2) of the Difficulty Predictor parse, for which DP = n1 + 2*n2. Falk &
    Konold (1997, p. 308) define the parse as the partition of the sequence
    into such chunks that "achieve[s] the lowest possible number" — chunk
    boundaries need not respect run boundaries (their example: XXXOXO ->
    XX|XOXO, DP 3). DP ties are broken toward the fewest chunks (the most
    compressed description), which makes (n1, n2) unique. For example
    HHTTHTHT -> {HH, TT} repetition + {HTHT} alternation -> (2, 1), DP = 4;
    HTHHTH -> {HTH, HTH} -> (0, 2), DP = 4. Implemented in ``features.py``.
    """
    return features.parse_motifs(clean_sequence(seq))


@functools.lru_cache(maxsize=_CACHE_SIZE)
def periodicity_score(seq: str) -> float:
    """
    Degree to which the sequence can be described by a short repeating template.

    Returns 0 for weak periodicity and approaches 1 for obvious patterns like
    HHHHHHHH or HTHTHTHT. Implemented in ``features.py``.
    """
    return features.periodicity_score(clean_sequence(seq))


@functools.lru_cache(maxsize=_CACHE_SIZE)
def local_imbalance(seq: str) -> float:
    """Worst H/T imbalance over sliding windows of length min(n, LOCAL_WINDOW).

    2*|prop_heads - 0.5| of the most imbalanced window (Kahneman & Tversky
    1972: representativeness holds "locally in each of its parts").
    Implemented in ``features.py``.
    """
    return features.local_imbalance(clean_sequence(seq))


@functools.lru_cache(maxsize=_CACHE_SIZE)
def multiscale_local_imbalance(seq: str) -> float:
    """Mean H/T imbalance across global and short local descriptions.

    The global sequence and each sliding-window scale from two through four
    receive equal weight. Within a scale, every window receives equal weight.
    This makes the operationalization explicit and avoids allowing a single
    worst window to determine the entire score. Implemented in ``features.py``.
    """
    return features.multiscale_local_imbalance(clean_sequence(seq))


def occurrence_probability(pattern: str, n_global: int) -> float:
    """P(``pattern`` occurs as a contiguous substring of ``n_global`` fair flips).

    The quantity of Hahn & Warren (2009): the probability that a length-k
    string appears at least once within a finite global sequence of fair coin
    flips. Implemented in ``features.py``.
    """
    return features.occurrence_probability(clean_sequence(pattern), n_global)


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
