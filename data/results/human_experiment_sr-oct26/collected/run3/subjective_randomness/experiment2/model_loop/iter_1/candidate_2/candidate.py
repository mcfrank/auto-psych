"""Tally-reverting chance coin.

People read a sequence flip by flip while keeping a running tally of how far
heads lead tails, and they believe a fair coin keeps that tally in check:
whenever one side is ahead, the next flip is expected to favour the side that
is behind, more strongly the bigger the lead. A sequence looks random to the
extent its tally path is probable under this self-correcting coin, and people
differ in how strongly (or in which direction) they expect the pull back.

For each step taken while the tally is unbalanced (|lead| = L >= 1), the
believed log-probability is log sigmoid(g * L) for a step back towards balance
and log sigmoid(-g * L) for a step away from it; steps from balance are 50/50
and carry no evidence. The choice is the log-probability ratio of the two
tally paths (fixed unit weight: the person's own likelihood ratio), with a
person-specific reversal strength g drawn from a population.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_LEAD = 7


def _tally_counts(seq):
    seq = seq.strip().upper()
    toward = np.zeros(MAX_LEAD)
    away = np.zeros(MAX_LEAD)
    lead = 0
    for c in seq:
        step = 1 if c == "H" else -1
        if lead != 0:
            L = abs(lead)
            if np.sign(step) == -np.sign(lead):
                toward[L - 1] += 1
            else:
                away[L - 1] += 1
        lead += step
    return toward, away


def compute_features(sequence_a, sequence_b):
    ta, aa = _tally_counts(sequence_a)
    tb, ab = _tally_counts(sequence_b)
    out = {}
    for i in range(MAX_LEAD):
        out[f"d_toward_{i + 1}"] = float(ta[i] - tb[i])
        out[f"d_away_{i + 1}"] = float(aa[i] - ab[i])
    return out


with pm.Model() as model:
    d_toward = pt.stack(
        [pm.Data(f"d_toward_{i + 1}", np.zeros(1, dtype="float64")) for i in range(MAX_LEAD)],
        axis=1,
    )
    d_away = pt.stack(
        [pm.Data(f"d_away_{i + 1}", np.zeros(1, dtype="float64")) for i in range(MAX_LEAD)],
        axis=1,
    )
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific belief in how strongly chance pulls the tally back to
    # balance (g > 0: gambler's-fallacy correction; g < 0: drifting tally).
    mu_g = pm.Normal("mu_g", mu=0.0, sigma=0.5)
    sigma_g = pm.HalfNormal("sigma_g", sigma=0.3)
    z_g = pm.Normal("z_g", mu=0.0, sigma=1.0, shape=400)
    g = mu_g + sigma_g * z_g

    leads = pt.as_tensor_variable(np.arange(1, MAX_LEAD + 1, dtype="float64"))
    gl = g[participant_id][:, None] * leads[None, :]
    log_toward = -pt.softplus(-gl)
    log_away = -pt.softplus(gl)
    evidence = pt.sum(d_toward * log_toward + d_away * log_away, axis=1)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(evidence))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
