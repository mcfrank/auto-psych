"""Tally span + ideal switching + periodic unit + triplet variety, person decisiveness.

Refinement of `iter0_candidate4`: people judge a sequence random by how close
its running heads-minus-tails tally span is to their own expected span
(length-scaled sensitivity), by how close its switching rate is to a shared
ideal (person-specific weight), by whether it repeats a short unit, and by the
variety of its three-flip chunks, with a personal left/right lean. The one
change: each person has one overall decisiveness that scales every cue of the
randomness impression together, so people differ in how consistently they act
on it (replacing the person-specific sensitivity on the tally span alone).
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

    def repeats_unit(seq):
        n = len(seq)
        if len(set(seq)) < 2:
            return 0.0
        for p in range(2, n // 2 + 1):
            if all(seq[i] == seq[i + p] for i in range(n - p)):
                return 1.0
        return 0.0

    def triplet_variety(seq):
        n_windows = len(seq) - 2
        if n_windows < 1:
            return 0.0
        distinct = len({seq[i:i + 3] for i in range(n_windows)})
        return distinct / min(8, n_windows)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "span_a": span_share(a),
        "span_b": span_share(b),
        "sw_a": switch_rate(a),
        "sw_b": switch_rate(b),
        "per_a": repeats_unit(a),
        "per_b": repeats_unit(b),
        "trip_diff": triplet_variety(a) - triplet_variety(b),
        "log_len": float(np.log(len(a) / 6.0)),
    }


N_SLOTS = 400

with pm.Model() as model:
    span_a = pm.Data("span_a", np.zeros(1, dtype="float64"))
    span_b = pm.Data("span_b", np.zeros(1, dtype="float64"))
    sw_a = pm.Data("sw_a", np.zeros(1, dtype="float64"))
    sw_b = pm.Data("sw_b", np.zeros(1, dtype="float64"))
    per_a = pm.Data("per_a", np.zeros(1, dtype="float64"))
    per_b = pm.Data("per_b", np.zeros(1, dtype="float64"))
    trip_diff = pm.Data("trip_diff", np.zeros(1, dtype="float64"))
    log_len = pm.Data("log_len", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific expected tally span (share of the length), in (0, 1).
    mu_ideal = pm.Normal("mu_ideal", mu=-0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Shared tally-span sensitivity, scaled by a power of the length.
    log_beta = pm.Normal("log_beta", mu=2.0, sigma=1.5)
    lam = pm.Normal("lam", mu=0.0, sigma=1.0)

    # Switching judged by closeness to a shared ideal switch rate.
    sw_ideal = pm.Beta("sw_ideal", alpha=6.0, beta=4.0)
    mu_kappa = pm.Normal("mu_kappa", mu=0.0, sigma=1.5)
    sigma_kappa = pm.HalfNormal("sigma_kappa", sigma=1.0)
    z_kappa = pm.Normal("z_kappa", mu=0.0, sigma=1.0, shape=N_SLOTS)
    kappa = mu_kappa + sigma_kappa * z_kappa

    rho = pm.Normal("rho", mu=0.0, sigma=1.0)
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)

    # The change: one person-specific decisiveness scaling the whole impression
    # (population mean fixed at 1 on the log scale so cue weights stay identified).
    sigma_gain = pm.HalfNormal("sigma_gain", sigma=0.7)
    z_gain = pm.Normal("z_gain", mu=0.0, sigma=1.0, shape=N_SLOTS)
    gain = pm.math.exp(sigma_gain * z_gain)

    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = sigma_side * z_side

    theta = ideal[participant_id]
    dist_a = (span_a - theta) ** 2
    dist_b = (span_b - theta) ** 2
    sens = pm.math.exp(log_beta + lam * log_len)
    sw_dist = (sw_b - sw_ideal) ** 2 - (sw_a - sw_ideal) ** 2
    impression = (
        sens * (dist_b - dist_a)
        + 4.0 * kappa[participant_id] * sw_dist
        + rho * (per_b - per_a)
        + gamma * trip_diff
    )
    logit = gain[participant_id] * impression + side[participant_id]
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(logit))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
