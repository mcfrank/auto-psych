"""Memory-span-limited comparison of two coin sequences.

People compare the two sequences in working memory, and each person has their
own memory span: a sequence within the span is judged sharply, but as its
length exceeds the span the impression of which sequence is more random fades
toward a guess. The impression is the tally-span / switching / three-flip
chunk variety / rule-built judgement of the current models; the personal
left/right lean does not fade. The claim is the person-specific memory span.
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
        "seq_len": float(len(a)),
    }


N_SLOTS = 400

with pm.Model() as model:
    span_a = pm.Data("span_a", np.zeros(1, dtype="float64"))
    span_b = pm.Data("span_b", np.zeros(1, dtype="float64"))
    sw_a = pm.Data("sw_a", np.zeros(1, dtype="float64"))
    sw_b = pm.Data("sw_b", np.zeros(1, dtype="float64"))
    trip_diff = pm.Data("trip_diff", np.zeros(1, dtype="float64"))
    rule_diff = pm.Data("rule_diff", np.zeros(1, dtype="float64"))
    seq_len = pm.Data("seq_len", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # The impression (shared structure, person-specific expected span and switch weight).
    mu_ideal = pm.Normal("mu_ideal", mu=-0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal)

    log_beta = pm.Normal("log_beta", mu=2.0, sigma=1.5)
    sw_ideal = pm.Beta("sw_ideal", alpha=6.0, beta=4.0)
    mu_kappa = pm.Normal("mu_kappa", mu=0.0, sigma=1.5)
    sigma_kappa = pm.HalfNormal("sigma_kappa", sigma=1.0)
    z_kappa = pm.Normal("z_kappa", mu=0.0, sigma=1.0, shape=N_SLOTS)
    kappa = mu_kappa + sigma_kappa * z_kappa
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)
    rho = pm.Normal("rho", mu=0.0, sigma=1.0)

    # The claim: a person-specific memory span (in flips); the impression fades
    # logistically as the length overshoots it (one flip of overshoot ~ halves the odds scale).
    mu_span = pm.Normal("mu_span", mu=8.0, sigma=2.0)
    sigma_span = pm.HalfNormal("sigma_span", sigma=2.0)
    z_span = pm.Normal("z_span", mu=0.0, sigma=1.0, shape=N_SLOTS)
    mem_span = mu_span + sigma_span * z_span
    retain = pm.math.sigmoid(mem_span[participant_id] - seq_len)

    # Person-specific left/right lean (decision stage, unaffected by memory).
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = sigma_side * z_side

    theta = ideal[participant_id]
    impression = (
        pm.math.exp(log_beta) * ((span_b - theta) ** 2 - (span_a - theta) ** 2)
        + 4.0 * kappa[participant_id] * ((sw_b - sw_ideal) ** 2 - (sw_a - sw_ideal) ** 2)
        + gamma * trip_diff
        - rho * rule_diff
    )
    logit = 2.0 * retain * impression + side[participant_id]
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(logit))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
