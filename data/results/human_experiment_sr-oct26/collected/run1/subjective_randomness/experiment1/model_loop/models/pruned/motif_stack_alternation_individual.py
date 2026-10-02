"""Refinement of motif_stack: the four-motif stack automaton randomness score
plus a participant-level random preference for alternation.

Hypothesis: people judge randomness as the log-likelihood ratio of a fair coin
versus Griffiths et al.'s four-motif stack automaton, but individuals differ in
how strongly they additionally favour (or disfavour) the sequence that
alternates more often; each participant's alternation preference is drawn from
a population distribution (non-centred). The single change from motif_stack is
the per-participant alternation slope. To keep the fit fast, the automaton's
parameters are held at motif_stack's posterior means on this data, so the
max-path/max-method score is computed once per sequence in numpy rather than
inside the sampled graph (its ridges made NUTS very slow).
"""

import numpy as np
import pymc as pm

MAX_SEQ_LEN = 8
MAX_PARTICIPANTS = 600

# motif_stack posterior means (experiment 1 fit).
DELTA = 0.640
ALPHA = 0.358
REPETITION_WEIGHT = 0.977
MIRROR_SHARE = 0.578
COMPLEMENT_SHARE = 0.473

_EMITS = "HTHTHT"


def _clean(seq):
    out = "".join(c.upper() for c in str(seq).strip() if not c.isspace())
    if not out or any(c not in "HT" for c in out) or len(out) > MAX_SEQ_LEN:
        raise ValueError(f"invalid H/T sequence {seq!r}")
    return out


def _automaton():
    a, a2, d = ALPHA, ALPHA**2, DELTA
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


_INIT, _TRANS = _automaton()
_REM = 1.0 - REPETITION_WEIGHT
_METHOD_W = np.array(
    [
        REPETITION_WEIGHT,
        _REM * MIRROR_SHARE,
        _REM * (1 - MIRROR_SHARE) * COMPLEMENT_SHARE,
        _REM * (1 - MIRROR_SHARE) * (1 - COMPLEMENT_SHARE),
    ]
)


def _viterbi_prefix_logs(seq):
    """log max-path probability of every prefix of seq."""
    masks = [np.array([float(e == ch) for e in _EMITS]) for ch in seq]
    best = _INIT * masks[0]
    logs = [np.log(best.max())]
    for m in masks[1:]:
        best = (best[:, None] * _TRANS).max(axis=0) * m
        logs.append(np.log(best.max()))
    return logs


def _score(seq):
    n = len(seq)
    logs = _viterbi_prefix_logs(seq)
    full, half = logs[n - 1], logs[(n - 1) // 2]
    k = (n + 1) // 2
    prefix, suffix = seq[:k], seq[k:]
    src = prefix[:-1] if n % 2 else prefix
    comp = {"H": "T", "T": "H"}
    cands = [np.log(_METHOD_W[0]) + full]
    if suffix == src[::-1]:
        cands.append(np.log(_METHOD_W[1]) + half)
    if suffix == "".join(comp[c] for c in src[::-1]):
        cands.append(np.log(_METHOD_W[2]) + half)
    if n % 2 == 0 and suffix == prefix:
        cands.append(np.log(_METHOD_W[3]) + half)
    return n * np.log(0.5) - max(cands)


def _alternation(seq):
    if len(seq) < 2:
        return 0.0
    return sum(1 for x, y in zip(seq, seq[1:]) if x != y) / (len(seq) - 1)


def prepare_observed(rows):
    rows = list(rows)
    if not rows:
        raise ValueError("prepare_observed requires at least one row.")
    score_diff, alt_diff, pid = [], [], []
    for i, row in enumerate(rows):
        a, b = _clean(row["sequence_a"]), _clean(row["sequence_b"])
        if len(a) != len(b):
            raise ValueError(f"row {i}: unequal lengths {a!r} vs {b!r}")
        score_diff.append(_score(a) - _score(b))
        alt_diff.append(_alternation(a) - _alternation(b))
        pid.append(int(float(row["participant_id"])))
    pid = np.array(pid, dtype="int64")
    if pid.min() < 0 or pid.max() >= MAX_PARTICIPANTS:
        raise ValueError(f"participant_id must lie in [0, {MAX_PARTICIPANTS})")
    has = ["chose_left" in r for r in rows]
    if any(has) and not all(has):
        raise ValueError("chose_left present on some rows but not others")
    chose = (
        np.array([int(float(r["chose_left"])) for r in rows], dtype="int64")
        if all(has)
        else np.zeros(len(rows), dtype="int64")
    )
    return {
        "score_diff": np.array(score_diff, dtype="float64"),
        "alt_diff": np.array(alt_diff, dtype="float64"),
        "participant_id": pid,
        "chose_left": chose,
    }


with pm.Model() as model:
    score_diff = pm.Data("score_diff", np.zeros(1, dtype="float64"))
    alt_diff = pm.Data("alt_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    beta = pm.HalfNormal("beta", sigma=2.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    # Participant-level alternation preference (non-centred).
    alt_mu = pm.Normal("alt_mu", mu=0.0, sigma=2.0)
    alt_sigma = pm.HalfNormal("alt_sigma", sigma=2.0)
    alt_z = pm.Normal("alt_z", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    alt_slope = alt_mu + alt_sigma * alt_z

    eta = beta * score_diff + alt_slope[participant_id] * alt_diff + side_bias
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(eta))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
