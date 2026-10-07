"""Refinement of motif_stack: per-person sensitivity on the motif-stack randomness score.

Hypothesis: people judge a sequence random to the extent it is poorly explained
by Griffiths et al. (2018)'s four-motif stack automaton (six-state motif
process with mirror, complement and duplication production methods, max-path /
max-method) relative to a fair coin, as in ``motif_stack``.  The one change:
the sensitivity mapping the score difference to choice varies across
participants (log-normal population, non-centred, 400 slots so new people can
be predicted) instead of being one shared value.

The automaton's own parameters are fixed at motif_stack's posterior means on
experiment 1's data (they are weakly identified and their max-ridged
likelihood made a joint fit with person effects unsampleable), so the score is
computed once per sequence in numpy and only the sensitivities and side bias
are inferred.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

N_PARTICIPANT_SLOTS = 400

# motif_stack posterior means (experiment 1).
_DELTA = 0.564
_ALPHA = 0.293
_REPETITION_WEIGHT = 0.978
_MIRROR_SHARE = 0.543
_COMPLEMENT_SHARE = 0.430
_EMITS = "HTHTHT"


def _automaton():
    a, a2, d = _ALPHA, _ALPHA**2, _DELTA
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
_REMAINING = 1.0 - _REPETITION_WEIGHT
_METHOD_WEIGHTS = np.array(
    [
        _REPETITION_WEIGHT,
        _REMAINING * _MIRROR_SHARE,
        _REMAINING * (1 - _MIRROR_SHARE) * _COMPLEMENT_SHARE,
        _REMAINING * (1 - _MIRROR_SHARE) * (1 - _COMPLEMENT_SHARE),
    ]
)


def _clean(seq):
    out = "".join(c for c in str(seq).strip().upper() if not c.isspace())
    if not out or set(out) - {"H", "T"}:
        raise ValueError(f"not an H/T sequence: {seq!r}")
    return out


def _viterbi_prefix_logs(seq):
    """Log max-path probability of every prefix of ``seq``."""
    mask = lambda ch: np.array([float(e == ch) for e in _EMITS])
    best = _INIT * mask(seq[0])
    logs = [np.log(best.max())]
    for ch in seq[1:]:
        best = (best[:, None] * _TRANSITION).max(axis=0) * mask(ch)
        logs.append(np.log(best.max()))
    return logs


def _randomness_score(seq):
    seq = _clean(seq)
    n = len(seq)
    logs = _viterbi_prefix_logs(seq)
    full, half = logs[n - 1], logs[(n - 1) // 2]
    prefix = seq[: (n + 1) // 2]
    source = prefix[:-1] if n % 2 else prefix
    suffix = seq[(n + 1) // 2 :]
    comp = {"H": "T", "T": "H"}
    candidates = [np.log(_METHOD_WEIGHTS[0]) + full]
    if suffix == source[::-1]:
        candidates.append(np.log(_METHOD_WEIGHTS[1]) + half)
    if suffix == "".join(comp[c] for c in source[::-1]):
        candidates.append(np.log(_METHOD_WEIGHTS[2]) + half)
    if n % 2 == 0 and suffix == prefix:
        candidates.append(np.log(_METHOD_WEIGHTS[3]) + half)
    return n * np.log(0.5) - max(candidates)


def compute_features(sequence_a, sequence_b):
    a, b = _clean(sequence_a), _clean(sequence_b)
    if len(a) != len(b):
        raise ValueError(f"same-length pairs only: {a!r} vs {b!r}")
    return {"motif_score_diff": float(_randomness_score(a) - _randomness_score(b))}


with pm.Model() as model:
    motif_score_diff = pm.Data("motif_score_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Per-person sensitivity: log-normal population, non-centred.
    log_beta_mu = pm.Normal("log_beta_mu", mu=0.0, sigma=1.0)
    log_beta_sigma = pm.HalfNormal("log_beta_sigma", sigma=0.5)
    log_beta_z = pm.Normal("log_beta_z", mu=0.0, sigma=1.0, shape=N_PARTICIPANT_SLOTS)
    beta_person = pt.exp(log_beta_mu + log_beta_sigma * log_beta_z)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=0.5)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta_person[participant_id] * motif_score_diff + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
