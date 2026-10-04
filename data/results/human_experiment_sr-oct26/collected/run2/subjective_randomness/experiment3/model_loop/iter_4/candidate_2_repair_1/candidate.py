"""Alignable-difference comparison: decisiveness depends on how alignable the pair is.

People compare the two sequences by aligning them flip by flip; how sharply
they act on their impression of which looks more random depends on the share of
positions at which the two strings differ (a shared exponential gain, direction
free). The impression is the current best model's (tally span against a personal
expected span, switching against a personal ideal rate, chunk variety and a
rule-built penalty scaled by personal pattern sensitivity, a terminal-streak
penalty); the personal left/right lean is not scaled. The closeness-to-an-ideal
terms are written in expanded quadratic form, so each person's weight and ideal
enter linearly (one posterior mode rather than a sign-flipped second one).
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
        # A unit of two or more flips repeated at least twice, or a sequence
        # equal to its own reversed complement.
        n = len(seq)
        if len(set(seq)) < 2:
            return 0.0
        for p in range(2, n // 2 + 1):
            if all(seq[i] == seq[i + p] for i in range(n - p)):
                return 1.0
        swapped = seq.translate(str.maketrans("HT", "TH"))
        return 1.0 if seq == swapped[::-1] else 0.0

    def terminal_excess(seq):
        # Flips of the final run beyond the first (0 when it ends on a switch).
        k = 1
        while k < len(seq) and seq[-1 - k] == seq[-1]:
            k += 1
        return float(k - 1)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "span_a": span_share(a),
        "span_b": span_share(b),
        "sw_a": switch_rate(a),
        "sw_b": switch_rate(b),
        "trip_diff": triplet_variety(a) - triplet_variety(b),
        "rule_diff": rule_built(a) - rule_built(b),
        "term_diff": terminal_excess(a) - terminal_excess(b),
        "log_len": float(np.log(len(a) / 6.0)),
        # Share of aligned positions at which the two strings differ, centred.
        "mismatch": float(np.mean([x != y for x, y in zip(a, b)])) - 0.5,
    }


N_SLOTS = 400

with pm.Model() as model:
    span_a = pm.Data("span_a", np.zeros(1, dtype="float64"))
    span_b = pm.Data("span_b", np.zeros(1, dtype="float64"))
    sw_a = pm.Data("sw_a", np.zeros(1, dtype="float64"))
    sw_b = pm.Data("sw_b", np.zeros(1, dtype="float64"))
    trip_diff = pm.Data("trip_diff", np.zeros(1, dtype="float64"))
    rule_diff = pm.Data("rule_diff", np.zeros(1, dtype="float64"))
    term_diff = pm.Data("term_diff", np.zeros(1, dtype="float64"))
    log_len = pm.Data("log_len", np.zeros(1, dtype="float64"))
    mismatch = pm.Data("mismatch", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific expected tally span (share of the length), centred near
    # 0.4; unbounded so the span term stays linear in it (no saturation).
    mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=0.3)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=0.2)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = mu_ideal + sigma_ideal * z_ideal

    # Person-specific sensitivity, scaled by a power of the length.
    mu_log_beta = pm.Normal("mu_log_beta", mu=2.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pm.math.exp(mu_log_beta + sigma_log_beta * z_beta)
    lam = pm.Normal("lam", mu=0.0, sigma=0.5)

    # Switching judged by closeness to a personal ideal rate. Written in its
    # expanded quadratic form, w*(sw^2) + v*sw per person (ideal = -v / 2w),
    # so the personal weight and ideal enter linearly and a negative weight
    # cannot trade off against an ideal pushed to an edge (a second mode).
    mu_w = pm.Normal("mu_w", mu=0.0, sigma=2.0)
    sigma_w = pm.HalfNormal("sigma_w", sigma=1.0)
    z_w = pm.Normal("z_w", mu=0.0, sigma=1.0, shape=N_SLOTS)
    w_sw = mu_w + sigma_w * z_w
    mu_v = pm.Normal("mu_v", mu=0.0, sigma=2.0)
    sigma_v = pm.HalfNormal("sigma_v", sigma=1.0)
    z_v = pm.Normal("z_v", mu=0.0, sigma=1.0, shape=N_SLOTS)
    v_sw = mu_v + sigma_v * z_v

    # Shared reward on the share of distinct three-flip chunks.
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)

    # Shared penalty on a rule-built (copied or anti-symmetric) sequence.
    rho = pm.Normal("rho", mu=0.0, sigma=1.0)

    # Person-specific pattern sensitivity scaling both structural cues
    # (median 1, so gamma and rho remain the typical person's weights).
    sigma_pat = pm.HalfNormal("sigma_pat", sigma=0.7)
    z_pat = pm.Normal("z_pat", mu=0.0, sigma=1.0, shape=N_SLOTS)
    pat = pm.math.exp(sigma_pat * z_pat)

    # The change: shared penalty per flip of the final run beyond the first.
    tau = pm.Normal("tau", mu=0.0, sigma=1.0)

    # The claim: decisiveness scales with the pair's position-wise dissimilarity.
    delta = pm.Normal("delta", mu=0.0, sigma=1.5)
    gain = pm.math.exp(delta * mismatch)

    # Person-specific left/right response bias (decision stage, not a cue).
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS)
    side = sigma_side * z_side

    theta = ideal[participant_id]
    dist_a = (span_a - theta) ** 2
    dist_b = (span_b - theta) ** 2
    sens = beta[participant_id] * pm.math.exp(lam * log_len)
    # Closeness of a's switching to the personal ideal minus b's.
    sw_close = -w_sw[participant_id] * (sw_a ** 2 - sw_b ** 2) - v_sw[participant_id] * (sw_a - sw_b)
    impression = (
        sens * (dist_b - dist_a)
        + sw_close
        + pat[participant_id] * (gamma * trip_diff - rho * rule_diff)
        - tau * term_diff
    )
    logit = gain * impression + side[participant_id]
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(logit))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
