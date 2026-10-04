"""Tally span + ideal switching rate + graded penalty for near-periodicity.

Refinement of `tally_span_switch_ideal_periodic_unit`: people keep a running
heads-minus-tails tally and judge a sequence random by how close the tally's
span (share of the length) is to their own expected span, with a personal
left/right lean, and judge switching by closeness to a shared ideal switching
rate (person-specific weight). The one change: the penalty for a visibly
repeating unit is graded, not all-or-nothing -- it grows with the squared
share of flips that match the flip one period earlier, at the best-fitting
period of two or more flips, so near-periodic sequences with a slip also look
designed.
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

    def near_periodicity(seq):
        # Best share of flips matching the flip one period back, periods 2..n/2.
        n = len(seq)
        best = 0.0
        for p in range(2, n // 2 + 1):
            matches = sum(1 for i in range(n - p) if seq[i] == seq[i + p])
            best = max(best, matches / (n - p))
        return best ** 2

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "span_a": span_share(a),
        "span_b": span_share(b),
        "sw_a": switch_rate(a),
        "sw_b": switch_rate(b),
        "nper_a": near_periodicity(a),
        "nper_b": near_periodicity(b),
        "log_len": float(np.log(len(a) / 6.0)),
    }


N_SLOTS = 400

with pm.Model() as model:
    span_a = pm.Data("span_a", np.zeros(1, dtype="float64"))
    span_b = pm.Data("span_b", np.zeros(1, dtype="float64"))
    sw_a = pm.Data("sw_a", np.zeros(1, dtype="float64"))
    sw_b = pm.Data("sw_b", np.zeros(1, dtype="float64"))
    nper_a = pm.Data("nper_a", np.zeros(1, dtype="float64"))
    nper_b = pm.Data("nper_b", np.zeros(1, dtype="float64"))
    log_len = pm.Data("log_len", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific expected tally span (share of the length), in (0, 1).
    mu_ideal = pm.Normal("mu_ideal", mu=-0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Person-specific sensitivity, scaled by a power of the length.
    mu_log_beta = pm.Normal("mu_log_beta", mu=2.0, sigma=1.5)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.7)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pm.Deterministic("beta", pm.math.exp(mu_log_beta + sigma_log_beta * z_beta))
    lam = pm.Normal("lam", mu=0.0, sigma=1.0)

    # Switching judged by closeness to a shared ideal switch rate.
    sw_ideal = pm.Beta("sw_ideal", alpha=6.0, beta=4.0)
    mu_kappa = pm.Normal("mu_kappa", mu=0.0, sigma=1.5)
    sigma_kappa = pm.HalfNormal("sigma_kappa", sigma=1.0)
    z_kappa = pm.Normal("z_kappa", mu=0.0, sigma=1.0, shape=N_SLOTS)
    kappa = mu_kappa + sigma_kappa * z_kappa

    # The change: shared graded penalty on how nearly a short unit repeats.
    rho = pm.Normal("rho", mu=0.0, sigma=1.0)

    # Person-specific left/right response bias (decision stage, not a cue).
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = sigma_side * z_side

    theta = ideal[participant_id]
    dist_a = (span_a - theta) ** 2
    dist_b = (span_b - theta) ** 2
    sens = beta[participant_id] * pm.math.exp(lam * log_len)
    sw_dist = (sw_b - sw_ideal) ** 2 - (sw_a - sw_ideal) ** 2
    logit = (
        sens * (dist_b - dist_a)
        + 4.0 * kappa[participant_id] * sw_dist
        + rho * (nper_b - nper_a)
        + side[participant_id]
    )
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(logit))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
