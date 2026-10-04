"""Tally span ideal, inverted-U switching ideal, plus a repeated-unit penalty.

Refinement of `tally_span_switch_rate_ideal`: people keep a running
heads-minus-tails tally and pick the sequence whose tally span (share of the
length) is closer to their own expected span, judge switching by closeness to a
shared ideal rate (person-specific weight), and have a personal left/right lean.
The one change: a sequence visibly built by repeating a short unit at least
twice (HTHTHTHT, HHTHHT, HHTTHHTT) looks designed, and is penalised by a shared
weight regardless of its span or switching rate.
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def span_share(seq):
        tally, hi, lo = 0, 0, 0
        for c in seq:
            tally += 1 if c == "H" else -1
            hi = max(hi, tally)
            lo = min(lo, tally)
        return (hi - lo) / len(seq)

    def switch_rate(seq):
        return sum(1 for x, y in zip(seq, seq[1:]) if x != y) / (len(seq) - 1)

    def periodic(seq):
        n = len(seq)
        for p in range(1, n // 2 + 1):
            if all(seq[i] == seq[i + p] for i in range(n - p)):
                return 1.0
        return 0.0

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "span_a": span_share(a),
        "span_b": span_share(b),
        "sw_a": switch_rate(a),
        "sw_b": switch_rate(b),
        "periodic_diff_ba": periodic(b) - periodic(a),
        "log_len": float(np.log(len(a) / 6.0)),
    }


N_SLOTS = 400

with pm.Model() as model:
    span_a = pm.Data("span_a", np.zeros(1, dtype="float64"))
    span_b = pm.Data("span_b", np.zeros(1, dtype="float64"))
    sw_a = pm.Data("sw_a", np.zeros(1, dtype="float64"))
    sw_b = pm.Data("sw_b", np.zeros(1, dtype="float64"))
    per_diff = pm.Data("periodic_diff_ba", np.zeros(1, dtype="float64"))
    log_len = pm.Data("log_len", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific expected tally span (share of the length), in (0, 1).
    mu_ideal = pm.Normal("mu_ideal", mu=-0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Person-specific span sensitivity, scaled by a power of the length.
    mu_log_beta = pm.Normal("mu_log_beta", mu=2.0, sigma=1.5)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.7)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pm.Deterministic("beta", pm.math.exp(mu_log_beta + sigma_log_beta * z_beta))
    lam = pm.Normal("lam", mu=0.0, sigma=1.0)

    # Closeness to a shared ideal switching rate, person-specific weight.
    sw_ideal = pm.Beta("sw_ideal", alpha=6.0, beta=3.0)
    mu_log_w = pm.Normal("mu_log_w", mu=1.0, sigma=1.0)
    sigma_log_w = pm.HalfNormal("sigma_log_w", sigma=0.7)
    z_w = pm.Normal("z_w", mu=0.0, sigma=1.0, shape=N_SLOTS)
    w = pm.math.exp(mu_log_w + sigma_log_w * z_w)

    # The change: shared penalty on a visibly repeated short unit.
    eta = pm.Normal("eta", mu=0.0, sigma=1.5)

    # Person-specific left/right response bias (decision stage, not a cue).
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = sigma_side * z_side

    theta = ideal[participant_id]
    dist_a = (span_a - theta) ** 2
    dist_b = (span_b - theta) ** 2
    sens = beta[participant_id] * pm.math.exp(lam * log_len)
    sw_dist_a = (sw_a - sw_ideal) ** 2
    sw_dist_b = (sw_b - sw_ideal) ** 2
    logit = (
        sens * (dist_b - dist_a)
        + w[participant_id] * (sw_dist_b - sw_dist_a)
        + eta * per_diff
        + side[participant_id]
    )
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(logit))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
