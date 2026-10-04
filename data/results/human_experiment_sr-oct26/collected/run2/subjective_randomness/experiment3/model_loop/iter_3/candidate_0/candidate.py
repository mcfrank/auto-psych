"""First-read anchoring of the randomness impression.

People read the left sequence first, and its impression of randomness anchors
the comparison: the first-read sequence's impression counts by a different
weight than the second's (a shared primacy weight), rather than the choice
resting on the plain difference of the two impressions. The impression is the
current best model's (tally span against a personal expected span, switching
against a personal ideal rate, chunk variety and a rule-built penalty scaled
by a personal pattern sensitivity, personal left/right lean). The same pair
is judged differently depending on which sequence is read first.
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
        "trip_a": triplet_variety(a),
        "trip_b": triplet_variety(b),
        "rule_a": rule_built(a),
        "rule_b": rule_built(b),
        "log_len": float(np.log(len(a) / 6.0)),
    }


N_SLOTS = 400

with pm.Model() as model:
    span_a = pm.Data("span_a", np.zeros(1, dtype="float64"))
    span_b = pm.Data("span_b", np.zeros(1, dtype="float64"))
    sw_a = pm.Data("sw_a", np.zeros(1, dtype="float64"))
    sw_b = pm.Data("sw_b", np.zeros(1, dtype="float64"))
    trip_a = pm.Data("trip_a", np.zeros(1, dtype="float64"))
    trip_b = pm.Data("trip_b", np.zeros(1, dtype="float64"))
    rule_a = pm.Data("rule_a", np.zeros(1, dtype="float64"))
    rule_b = pm.Data("rule_b", np.zeros(1, dtype="float64"))
    log_len = pm.Data("log_len", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific expected tally span (share of the length).
    mu_ideal = pm.Normal("mu_ideal", mu=-0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal)

    # Person-specific span sensitivity, scaled by a power of the length.
    mu_log_beta = pm.Normal("mu_log_beta", mu=2.0, sigma=1.5)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.7)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pm.math.exp(mu_log_beta + sigma_log_beta * z_beta)
    lam = pm.Normal("lam", mu=0.0, sigma=1.0)

    # Person-specific ideal switching rate and switching weight.
    mu_sw = pm.Normal("mu_sw", mu=0.4, sigma=0.7)
    sigma_sw = pm.HalfNormal("sigma_sw", sigma=0.7)
    z_sw = pm.Normal("z_sw", mu=0.0, sigma=1.0, shape=N_SLOTS)
    sw_ideal = pm.math.sigmoid(mu_sw + sigma_sw * z_sw)
    mu_kappa = pm.Normal("mu_kappa", mu=0.0, sigma=1.5)
    sigma_kappa = pm.HalfNormal("sigma_kappa", sigma=1.0)
    z_kappa = pm.Normal("z_kappa", mu=0.0, sigma=1.0, shape=N_SLOTS)
    kappa = mu_kappa + sigma_kappa * z_kappa

    # Chunk variety reward and rule-built penalty, personal pattern sensitivity.
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)
    rho = pm.Normal("rho", mu=0.0, sigma=1.0)
    sigma_pat = pm.HalfNormal("sigma_pat", sigma=0.7)
    z_pat = pm.Normal("z_pat", mu=0.0, sigma=1.0, shape=N_SLOTS)
    pat = pm.math.exp(sigma_pat * z_pat)

    # Personal left/right response bias.
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = sigma_side * z_side

    # The claim: a shared primacy weight on the first-read (left) impression
    # (delta > 0: the first sequence counts more; delta < 0: less).
    delta = pm.Normal("delta", mu=0.0, sigma=0.5)

    theta = ideal[participant_id]
    sens = beta[participant_id] * pm.math.exp(lam * log_len)
    sw_p = sw_ideal[participant_id]
    k = 4.0 * kappa[participant_id]
    pp = pat[participant_id]

    def impression(span, sw, trip, rule):
        return (
            -sens * (span - theta) ** 2
            - k * (sw - sw_p) ** 2
            + pp * (gamma * trip - rho * rule)
        )

    s_first = impression(span_a, sw_a, trip_a, rule_a)
    s_second = impression(span_b, sw_b, trip_b, rule_b)
    logit = pm.math.exp(delta) * s_first - pm.math.exp(-delta) * s_second + side[participant_id]
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(logit))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
