"""Refinement of motif_stack: shared four-motif stack automaton plus
person-specific alternation preference.

Hypothesis: people judge randomness as the log-likelihood ratio of a fair coin
against Griffiths et al. (2018)'s four-motif stack automaton (a regularity
detector shared by everyone), but individuals differ in how strongly they
additionally treat frequent alternation itself as a sign of randomness. The
single change from motif_stack is a person-specific weight (non-centred
population distribution, 400 slots so a new participant is a draw from it) on
the difference in alternation rate between the two sequences.

The automaton's internal settings (delta, alpha, production-method weights)
are held at motif_stack's posterior means on these data, so each sequence's
randomness score is computed once, in numpy, by compute_features; the graph is
a vectorised logistic choice rule. (Those settings were weakly identified in
motif_stack's own fit, and the max-over-paths likelihood made NUTS slow.)
"""

import numpy as np
import pymc as pm

N_PARTICIPANT_SLOTS = 400
_EMITS = "HTHTHT"

# motif_stack's posterior means on experiment-1 data.
_DELTA = 0.471
_ALPHA = 0.176
_REPETITION_WEIGHT = 0.979
_MIRROR_SHARE = 0.288
_COMPLEMENT_SHARE = 0.372


def _clean(seq):
    out = "".join(c.upper() for c in str(seq) if not c.isspace())
    if not out or any(c not in "HT" for c in out):
        raise ValueError(f"not an H/T sequence: {seq!r}")
    return out


def _automaton():
    a, a2, d = _ALPHA, _ALPHA ** 2, _DELTA
    rows = np.array(
        [
            [d, a, a2, 0, 0, a2],
            [a, d, a2, 0, 0, a2],
            [a, a, 0, d, 0, a2],
            [a, a, d, 0, 0, a2],
            [a, a, a2, 0, 0, d],
            [a, a, a2, 0, d, 0],
        ],
        dtype="float64",
    )
    transition = rows / rows.sum(axis=1, keepdims=True)
    init = np.array([a, a, a2, 0, 0, a2], dtype="float64")
    return init / init.sum(), transition


_INIT, _TRANSITION = _automaton()
_REM = 1.0 - _REPETITION_WEIGHT
_METHOD_WEIGHTS = np.array(
    [
        _REPETITION_WEIGHT,
        _REM * _MIRROR_SHARE,
        _REM * (1 - _MIRROR_SHARE) * _COMPLEMENT_SHARE,
        _REM * (1 - _MIRROR_SHARE) * (1 - _COMPLEMENT_SHARE),
    ]
)


def _viterbi_log(seq):
    mask = lambda ch: np.array([1.0 if e == ch else 0.0 for e in _EMITS])
    best = _INIT * mask(seq[0])
    for ch in seq[1:]:
        best = (best[:, None] * _TRANSITION).max(axis=0) * mask(ch)
    return float(np.log(best.max()))


def _randomness_score(seq):
    """log P(seq | fair) - log P(seq | regular), max over production methods."""
    n = len(seq)
    half_len = (n + 1) // 2
    prefix = seq[:half_len]
    source = prefix[:-1] if n % 2 else prefix
    suffix = seq[half_len:]
    comp = {"H": "T", "T": "H"}
    logs = [np.log(_METHOD_WEIGHTS[0]) + _viterbi_log(seq)]
    half = _viterbi_log(prefix)
    if suffix == source[::-1]:
        logs.append(np.log(_METHOD_WEIGHTS[1]) + half)
    if suffix == "".join(comp[c] for c in source[::-1]):
        logs.append(np.log(_METHOD_WEIGHTS[2]) + half)
    if n % 2 == 0 and suffix == prefix:
        logs.append(np.log(_METHOD_WEIGHTS[3]) + half)
    return n * np.log(0.5) - max(logs)


def _alternation_rate(seq):
    if len(seq) < 2:
        return 0.0
    return sum(1 for x, y in zip(seq, seq[1:]) if x != y) / (len(seq) - 1)


def compute_features(sequence_a, sequence_b):
    a, b = _clean(sequence_a), _clean(sequence_b)
    if len(a) != len(b):
        raise ValueError(
            f"motif-stack scores are only comparable at equal lengths: {a!r} vs {b!r}"
        )
    return {
        "motif_score_diff": _randomness_score(a) - _randomness_score(b),
        "alt_diff": _alternation_rate(a) - _alternation_rate(b),
    }


with pm.Model() as model:
    motif_score_diff = pm.Data("motif_score_diff", np.zeros(1, dtype="float64"))
    alt_diff = pm.Data("alt_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Sensitivity to the shared automaton's randomness score.
    beta = pm.HalfNormal("beta", sigma=2.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=0.5)

    # Person-specific alternation preference (non-centred).
    alt_mu = pm.Normal("alt_mu", mu=0.0, sigma=2.0)
    alt_sigma = pm.HalfNormal("alt_sigma", sigma=1.5)
    alt_z = pm.Normal("alt_z", mu=0.0, sigma=1.0, shape=N_PARTICIPANT_SLOTS)
    alt_weight = alt_mu + alt_sigma * alt_z

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(
            beta * motif_score_diff
            + alt_weight[participant_id] * alt_diff
            + side_bias
        ),
    )
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
