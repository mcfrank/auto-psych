"""First-read anchoring of departures from randomness.

The sequence read first (the left one) sets the anchor: people register how
far each sequence departs from what a fair coin should do (tally span away
from their expected span, switching rate away from an ideal rate), but a
departure in the first-read sequence weighs by a different amount than the
same departure in the second-read one (a shared asymmetry w). The choice thus
depends on the pair's overall non-randomness and the reading order, not only
on the gap. Chunk variety, a rule-built penalty and a personal left/right lean
enter as usual.
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

    # Person-specific expected tally span (share of the length).
    mu_ideal = pm.Normal("mu_ideal", mu=-0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=0.7)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal)

    # Person-specific span sensitivity, scaled by a power of the length.
    mu_log_beta = pm.Normal("mu_log_beta", mu=2.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pm.math.exp(mu_log_beta + sigma_log_beta * z_beta)
    lam = pm.Normal("lam", mu=0.0, sigma=0.5)

    # Shared ideal switching rate, person-specific (non-negative) weight.
    sw_ideal = pm.Beta("sw_ideal", alpha=4.0, beta=3.0)
    mu_log_kappa = pm.Normal("mu_log_kappa", mu=1.0, sigma=1.0)
    sigma_log_kappa = pm.HalfNormal("sigma_log_kappa", sigma=0.5)
    z_kappa = pm.Normal("z_kappa", mu=0.0, sigma=1.0, shape=N_SLOTS)
    kappa = pm.math.exp(mu_log_kappa + sigma_log_kappa * z_kappa)

    # Shared chunk-variety reward and rule-built penalty.
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)
    rho = pm.Normal("rho", mu=0.0, sigma=1.0)

    # Personal left/right lean.
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = sigma_side * z_side

    # The claim: first-read departures weigh (1 + w), second-read (1 - w).
    w_raw = pm.Normal("w_raw", mu=0.0, sigma=0.5)
    w = pm.Deterministic("w", 0.9 * pm.math.tanh(w_raw))

    pid = participant_id
    theta = ideal[pid]
    sens = beta[pid] * pm.math.exp(lam * log_len)
    k = kappa[pid]
    dep_a = sens * (span_a - theta) ** 2 + k * (sw_a - sw_ideal) ** 2
    dep_b = sens * (span_b - theta) ** 2 + k * (sw_b - sw_ideal) ** 2

    logit = (
        (1.0 - w) * dep_b
        - (1.0 + w) * dep_a
        + gamma * trip_diff
        - rho * rule_diff
        + side[pid]
    )
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(logit))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
