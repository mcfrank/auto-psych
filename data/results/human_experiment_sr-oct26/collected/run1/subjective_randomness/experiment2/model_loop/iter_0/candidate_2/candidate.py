"""Shared-stretch cancellation in side-by-side comparison.

People compare the two sequences side by side and cancel what they share: the
flips both sequences have in common at their start and at their end are
discounted, and each sequence is judged only on the stretch where the two
differ (plus the flip on either side, so the switches into and out of it
count). On that stretch each person judges randomness by the distance of its
switching rate from their own ideal, its heads/tails imbalance and its longest
streak (personal weights), and guesses on some trials at a personal lapse rate.
The same sequence is therefore judged differently beside a different partner.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400


def _common_prefix(a, b):
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def compute_features(sequence_a, sequence_b):
    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    if len(a) != len(b) or len(a) < 2:
        raise ValueError(f"need equal-length sequences of >= 2 flips: {a!r}, {b!r}")
    L = len(a)
    if a == b:
        lo, hi = 0, L
    else:
        p = _common_prefix(a, b)
        s = _common_prefix(a[::-1], b[::-1])
        lo = max(p - 1, 0)
        hi = min(L - s + 1, L)
        if hi - lo < 2:
            lo, hi = max(hi - 2, 0), max(hi, 2)

    def feats(seq):
        w = seq[lo:hi]
        n = len(w)
        alt = sum(1 for x, y in zip(w, w[1:]) if x != y) / (n - 1)
        imb = abs(w.count("H") - w.count("T")) / n
        best = cur = 1
        for x, y in zip(w, w[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        return alt, imb, (best - 1) / (n - 1)

    alt_a, imb_a, run_a = feats(a)
    alt_b, imb_b, run_b = feats(b)
    return {
        "dalt_a": alt_a,
        "dalt_b": alt_b,
        "dimb_a": imb_a,
        "dimb_b": imb_b,
        "drun_a": run_a,
        "drun_b": run_b,
    }


with pm.Model() as model:
    dalt_a = pm.Data("dalt_a", np.zeros(1, dtype="float64"))
    dalt_b = pm.Data("dalt_b", np.zeros(1, dtype="float64"))
    dimb_a = pm.Data("dimb_a", np.zeros(1, dtype="float64"))
    dimb_b = pm.Data("dimb_b", np.zeros(1, dtype="float64"))
    drun_a = pm.Data("drun_a", np.zeros(1, dtype="float64"))
    drun_b = pm.Data("drun_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal ideal switching rate (non-centred logit-normal population).
    mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))
    beta = pm.LogNormal("beta", mu=2.5, sigma=0.5)

    # Personal weight on heads/tails imbalance of the differing stretch.
    mu_gamma = pm.Normal("mu_gamma", mu=0.0, sigma=2.0)
    sigma_gamma = pm.HalfNormal("sigma_gamma", sigma=1.5)
    z_gamma = pm.Normal("z_gamma", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    gamma = mu_gamma + sigma_gamma * z_gamma

    # Personal streak aversion on the differing stretch.
    mu_delta = pm.Normal("mu_delta", mu=0.0, sigma=2.0)
    sigma_delta = pm.HalfNormal("sigma_delta", sigma=1.5)
    z_delta = pm.Normal("z_delta", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    delta = mu_delta + sigma_delta * z_delta

    # Personal lapse (guessing) rate.
    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))

    theta = ideal[participant_id]
    g = gamma[participant_id]
    d = delta[participant_id]
    score_a = -beta * pt.sqr(dalt_a - theta) - g * dimb_a - d * drun_a
    score_b = -beta * pt.sqr(dalt_b - theta) - g * dimb_b - d * drun_b
    p_engaged = pm.math.sigmoid(score_a - score_b)
    lam = lapse[participant_id]
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
