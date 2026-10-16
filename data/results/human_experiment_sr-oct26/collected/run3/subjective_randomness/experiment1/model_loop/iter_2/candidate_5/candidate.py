"""Refinement of iter0_candidate3: representativeness with a fitted balance grain.

People judge randomness by Kahneman & Tversky local representativeness (H/T
balance plus irregularity relative to an over-alternating prototype and
periodic templates) with person-specific decision sensitivity. The single
change: the balance judgment is a fitted mix of whole-sequence H/T balance and
short-window (2-4 flip) local balance, instead of a fixed equal-weight average
of the scales, so people may weight overall balance more than local balance.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

N_SLOTS = 400
LOCAL_WINDOW = 4


def _clean(seq):
    s = seq.strip().upper()
    if not s:
        raise ValueError("Sequence must not be empty")
    return s


def _periodicity(s):
    n = len(s)
    if n <= 2:
        return 0.0
    best = 0.5
    for period in range(1, n // 2 + 1):
        t = s[:period]
        m = sum(1 for i, c in enumerate(s) if c == t[i % period])
        best = max(best, m / n)
    return max(0.0, min(1.0, 2.0 * (best - 0.5)))


def _global_and_local_imbalance(s):
    n = len(s)
    glob = 2.0 * abs(s.count("H") / n - 0.5)
    scales = []
    for w in range(2, min(LOCAL_WINDOW, n - 1) + 1):
        vals = [2.0 * abs(s[i:i + w].count("H") / w - 0.5) for i in range(n - w + 1)]
        scales.append(sum(vals) / len(vals))
    # Length-2 sequences have no proper sub-window: local balance = global.
    local = sum(scales) / len(scales) if scales else glob
    return glob, local


def compute_features(sequence_a, sequence_b):
    out = {}
    for seq, sfx in ((sequence_a, "a"), (sequence_b, "b")):
        s = _clean(seq)
        n = len(s)
        alts = sum(1 for i in range(1, n) if s[i] != s[i - 1])
        g, l = _global_and_local_imbalance(s)
        out[f"p_alts_{sfx}"] = alts / (n - 1) if n > 1 else 0.0
        out[f"periodicity_{sfx}"] = _periodicity(s)
        out[f"global_imb_{sfx}"] = g
        out[f"local_imb_{sfx}"] = l
    return out


with pm.Model() as model:
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    global_imb_a = pm.Data("global_imb_a", np.zeros(1, dtype="float64"))
    global_imb_b = pm.Data("global_imb_b", np.zeros(1, dtype="float64"))
    local_imb_a = pm.Data("local_imb_a", np.zeros(1, dtype="float64"))
    local_imb_b = pm.Data("local_imb_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    theta_alt = pm.Uniform("theta_alt", lower=0.5001, upper=0.95)
    alt_weight = pm.Uniform("alt_weight", lower=0.01, upper=0.99)
    periodic_share = pm.Uniform("periodic_share", lower=0.01, upper=0.99)
    # The one change: fitted share of balance judged on the whole sequence.
    global_share = pm.Beta("global_share", alpha=2.0, beta=2.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=1.0)

    log_beta_mu = pm.Normal("log_beta_mu", mu=1.0, sigma=1.0)
    log_beta_sigma = pm.HalfNormal("log_beta_sigma", sigma=0.7)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta_person = pm.math.exp(log_beta_mu + log_beta_sigma * z_beta)

    imb_a = global_share * global_imb_a + (1.0 - global_share) * local_imb_a
    imb_b = global_share * global_imb_b + (1.0 - global_share) * local_imb_b
    irr_a = (1.0 - periodic_share) * pt.abs(p_alts_a - theta_alt) + periodic_share * periodicity_a
    irr_b = (1.0 - periodic_share) * pt.abs(p_alts_b - theta_alt) + periodic_share * periodicity_b
    balance_weight = 1.0 - alt_weight
    score_a = -(balance_weight * imb_a + alt_weight * irr_a)
    score_b = -(balance_weight * imb_b + alt_weight * irr_b)

    p_left = pm.Deterministic(
        "p_left",
        pm.math.sigmoid(beta_person[participant_id] * (score_a - score_b) + side_bias),
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
