"""Recency-weighted local repetition hypothesis.

People judge how random a sequence looks by tracking whether each toss
repeats the toss that came right before it, and weight recent tosses more
heavily. A sequence reads as random when its recent tosses keep alternating
(few repeats, near 50-50) and as suspiciously structured when recent tosses
keep repeating the last one. The cognitive log-odds favor the option whose
recency-weighted repetition rate is lower (cleaner, more alternating recent
tail) as the more random one. A free parameter infers how much participants
rely on recency versus the whole sequence.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

DECAY = 0.5  # exponential recency kernel rate (fixed modelling choice)


def _repetition_rates(seq):
    """Return (unweighted, recency-weighted) rate of run-continuations."""
    seq = seq.strip().upper()
    n = len(seq)
    if n < 2:
        return (0.5, 0.5)
    cents = []
    for i in range(1, n):
        cents.append(1.0 if seq[i] == seq[i - 1] else 0.0)
    cents = np.array(cents, dtype="float64")
    unweighted = cents.mean()
    dist_from_end = np.arange(len(cents) - 1, -1, -1, dtype="float64")
    weights = np.exp(-DECAY * dist_from_end)
    weighted = (cents * weights).sum() / weights.sum()
    return (float(unweighted), float(weighted))


def compute_features(sequence_a, sequence_b):
    """Return recency-agnostic and recency-weighted repetition rates."""
    ua, wa = _repetition_rates(sequence_a)
    ub, wb = _repetition_rates(sequence_b)
    return {
        "rep_plain_a": ua,
        "rep_rec_a": wa,
        "rep_plain_b": ub,
        "rep_rec_b": wb,
    }


with pm.Model() as model:
    rep_plain_a = pm.Data("rep_plain_a", np.zeros(1, dtype="float64"))
    rep_rec_a = pm.Data("rep_rec_a", np.zeros(1, dtype="float64"))
    rep_plain_b = pm.Data("rep_plain_b", np.zeros(1, dtype="float64"))
    rep_rec_b = pm.Data("rep_rec_b", np.zeros(1, dtype="float64"))

    alpha = pm.Beta("alpha", 2.0, 2.0)
    beta = pm.HalfNormal("beta", sigma=2.0)

    rep_eff_a = (1 - alpha) * rep_plain_a + alpha * rep_rec_a
    rep_eff_b = (1 - alpha) * rep_plain_b + alpha * rep_rec_b

    p_chose_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta * (rep_eff_b - rep_eff_a))
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_chose_left, observed=chose_left)
