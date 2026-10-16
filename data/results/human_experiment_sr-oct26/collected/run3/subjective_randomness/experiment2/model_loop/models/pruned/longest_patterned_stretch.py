"""Longest patterned stretch.

People judge randomness by the single most striking patterned stretch in a
sequence: the longest contiguous stretch that keeps copying one short motif
(a streak, strict alternation, or a repeated 3- or 4-flip motif). The number of
flips that one stretch "predicts" (its length beyond the motif itself), weighed
by a motif-kind salience, sets how non-random the sequence looks; the sequence
whose most striking patterned stretch is weaker is chosen as more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

PERIODS = (1, 2, 3, 4)


def _primitive(motif):
    p = len(motif)
    for q in range(1, p):
        if p % q == 0 and motif == motif[:q] * (p // q):
            return False
    return True


def _max_excess(seq, p):
    """Longest stretch copying a primitive period-p motif: flips beyond the motif."""
    n = len(seq)
    best = 0
    for i in range(n - p + 1):
        if not _primitive(seq[i:i + p]):
            continue
        j = i + p
        while j < n and seq[j] == seq[j - p]:
            j += 1
        best = max(best, j - i - p)
    return float(best)


def compute_features(sequence_a, sequence_b):
    out = {}
    for tag, seq in (("a", sequence_a), ("b", sequence_b)):
        s = seq.strip().upper()
        for p in PERIODS:
            out[f"ex{p}_{tag}"] = _max_excess(s, p)
    return out


with pm.Model() as model:
    ex_a = pt.stack([pm.Data(f"ex{p}_a", np.zeros(1, dtype="float64")) for p in PERIODS])
    ex_b = pt.stack([pm.Data(f"ex{p}_b", np.zeros(1, dtype="float64")) for p in PERIODS])

    # Salience of each motif kind per predicted flip (streak's fixed at 1).
    log_w = pm.Normal("log_w", mu=0.0, sigma=1.0, shape=3)
    w = pt.concatenate([pt.ones(1), pt.exp(log_w)])[:, None]
    # Sensitivity of the choice to the difference in striking-stretch strength.
    beta = pm.HalfNormal("beta", sigma=2.0)

    strength_a = pt.max(w * ex_a, axis=0)
    strength_b = pt.max(w * ex_b, axis=0)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * (strength_b - strength_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
