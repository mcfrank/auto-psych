"""Worst-window switching deviation.

People judge a sequence by its single most off-pattern stretch of switching:
they scan it in four-flip stretches and register how far each stretch's
switching rate is from their own ideal switching rate for a random coin; the
one stretch that departs most from that ideal (a local streak or a local run
of strict alternation) alone decides how non-random the sequence looks. Each
person has their own ideal switching rate and decisiveness, plus a small
habitual left/right lean.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

WINDOW = 4  # flips per stretch (three transitions)


def compute_features(sequence_a, sequence_b):
    def window_switch_extremes(seq):
        seq = seq.strip().upper()
        sw = [1.0 if x != y else 0.0 for x, y in zip(seq, seq[1:])]
        k = min(WINDOW - 1, len(sw))
        rates = [sum(sw[i:i + k]) / k for i in range(len(sw) - k + 1)]
        return min(rates), max(rates)

    lo_a, hi_a = window_switch_extremes(sequence_a)
    lo_b, hi_b = window_switch_extremes(sequence_b)
    return {"wlo_a": lo_a, "whi_a": hi_a, "wlo_b": lo_b, "whi_b": hi_b}


N_SLOTS = 400
TAU = 0.03  # sharpness of the max over stretches (near-hard maximum)

with pm.Model() as model:
    wlo_a = pm.Data("wlo_a", np.zeros(1, dtype="float64"))
    whi_a = pm.Data("whi_a", np.zeros(1, dtype="float64"))
    wlo_b = pm.Data("wlo_b", np.zeros(1, dtype="float64"))
    whi_b = pm.Data("whi_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific ideal switching rate (logit scale, non-centred).
    mu_sw = pm.Normal("mu_sw", mu=0.3, sigma=0.7)
    sigma_sw = pm.HalfNormal("sigma_sw", sigma=0.7)
    z_sw = pm.Normal("z_sw", mu=0.0, sigma=1.0, shape=N_SLOTS)
    sw_ideal = pm.math.sigmoid(mu_sw + sigma_sw * z_sw)

    # Person-specific decisiveness (log scale, non-centred).
    mu_log_beta = pm.Normal("mu_log_beta", mu=1.5, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.7)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pm.math.exp(mu_log_beta + sigma_log_beta * z_beta)

    # Person-specific left/right lean.
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = sigma_side * z_side

    s = sw_ideal[participant_id]

    def worst(lo, hi):
        # Deviation is convex in the window rate, so the worst stretch is the
        # least- or the most-switching one; smooth maximum of the two.
        d_lo = (lo - s) ** 2
        d_hi = (hi - s) ** 2
        return TAU * pt.logaddexp(d_lo / TAU, d_hi / TAU)

    logit = beta[participant_id] * (worst(wlo_b, whi_b) - worst(wlo_a, whi_a)) + side[participant_id]
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(logit))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
