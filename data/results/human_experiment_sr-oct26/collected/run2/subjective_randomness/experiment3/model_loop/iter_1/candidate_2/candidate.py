"""Shared randomness impression, person-specific decisiveness.

Everyone shares one impression of what makes a coin sequence look random (a
running heads-minus-tails tally whose span is close to an expected span,
switching close to an ideal rate, varied three-flip chunks, no visible
construction rule), but people differ in how consistently they act on it: each
person has their own decisiveness, a single gain on the whole felt difference
in randomness between the two sequences. Individual differences lie in decision
noise, not in what randomness looks like.
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

    def triplet_variety(seq):
        n_windows = len(seq) - 2
        if n_windows < 1:
            return 0.0
        distinct = len({seq[i:i + 3] for i in range(n_windows)})
        return distinct / min(8, n_windows)

    def rule_built(seq):
        n = len(seq)
        if len(set(seq)) < 2:
            return 0.0
        for p in range(2, n // 2 + 1):
            if all(seq[i] == seq[i + p] for i in range(n - p)):
                return 1.0
        swapped = seq.translate(str.maketrans("HT", "TH"))
        return 1.0 if seq == swapped[::-1] else 0.0

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "span_a": span_share(a),
        "span_b": span_share(b),
        "sw_a": switch_rate(a),
        "sw_b": switch_rate(b),
        "trip_diff": triplet_variety(a) - triplet_variety(b),
        "rule_diff": rule_built(a) - rule_built(b),
        "log_len": float(np.log(len(a) / 6.0)),
    }


N_SLOTS = 400

with pm.Model() as model:
    span_a = pm.Data("span_a", np.zeros(1, dtype="float64"))
    span_b = pm.Data("span_b", np.zeros(1, dtype="float64"))
    sw_a = pm.Data("sw_a", np.zeros(1, dtype="float64"))
    sw_b = pm.Data("sw_b", np.zeros(1, dtype="float64"))
    trip_diff = pm.Data("trip_diff", np.zeros(1, dtype="float64"))
    rule_diff = pm.Data("rule_diff", np.zeros(1, dtype="float64"))
    log_len = pm.Data("log_len", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Shared impression: expected tally span, its length-scaled sensitivity,
    # ideal switching rate and its weight, triplet variety, rule-built penalty.
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(pm.Normal("logit_ideal", mu=-0.5, sigma=1.0)))
    log_beta = pm.Normal("log_beta", mu=2.0, sigma=1.5)
    lam = pm.Normal("lam", mu=0.0, sigma=1.0)
    sw_ideal = pm.Beta("sw_ideal", alpha=6.0, beta=4.0)
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)
    rho = pm.Normal("rho", mu=0.0, sigma=1.0)

    # The claim: person-specific decisiveness, a gain on the whole impression
    # (log-gain centred at zero so the shared weights carry the scale).
    sigma_gain = pm.HalfNormal("sigma_gain", sigma=1.0)
    z_gain = pm.Normal("z_gain", mu=0.0, sigma=1.0, shape=N_SLOTS)
    gain = pm.Deterministic("gain", pm.math.exp(sigma_gain * z_gain))

    # Person-specific left/right response bias.
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = sigma_side * z_side

    dist_a = (span_a - ideal) ** 2
    dist_b = (span_b - ideal) ** 2
    sens = pm.math.exp(log_beta + lam * log_len)
    sw_dist = (sw_b - sw_ideal) ** 2 - (sw_a - sw_ideal) ** 2
    impression = (
        sens * (dist_b - dist_a)
        + 4.0 * kappa * sw_dist
        + gamma * trip_diff
        - rho * rule_diff
    )
    logit = gain[participant_id] * impression + side[participant_id]
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(logit))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
