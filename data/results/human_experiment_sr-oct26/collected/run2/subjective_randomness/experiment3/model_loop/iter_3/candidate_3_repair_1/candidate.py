"""Run-length variety added to the person-specific switching-ideal pattern model.

Refinement of `person_switch_ideal_pattern_rule_built`: people keep a running
heads-minus-tails tally and judge a sequence random by how close its span is
to their own expected span (person-specific, length-scaled sensitivity),
judge switching by closeness to their own ideal switching rate (person-specific
weight), weigh three-flip chunk variety and a rule-built penalty by a personal
pattern sensitivity, and have a personal left/right lean. The one change:
people expect a random coin's streaks to come in mixed lengths, so a sequence
whose runs take many different lengths (relative to the most its length
allows) looks more random, by a shared reward.
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

    def run_length_variety(seq):
        # Distinct run lengths beyond one, as a share of the most possible
        # for this length (k distinct lengths need at least 1 + 2 + ... + k flips).
        runs, count = [], 1
        for x, y in zip(seq, seq[1:]):
            if x == y:
                count += 1
            else:
                runs.append(count)
                count = 1
        runs.append(count)
        max_k = 1
        while (max_k + 1) * (max_k + 2) // 2 <= len(seq):
            max_k += 1
        if max_k < 2:
            return 0.0
        return (len(set(runs)) - 1) / (max_k - 1)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "span_a": span_share(a),
        "span_b": span_share(b),
        "sw_a": switch_rate(a),
        "sw_b": switch_rate(b),
        "trip_diff": triplet_variety(a) - triplet_variety(b),
        "rule_diff": rule_built(a) - rule_built(b),
        "runvar_diff": run_length_variety(a) - run_length_variety(b),
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
    runvar_diff = pm.Data("runvar_diff", np.zeros(1, dtype="float64"))
    log_len = pm.Data("log_len", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Population means and spreads of the six person-level traits, in order:
    # expected tally span (logit), log span sensitivity, ideal switching rate
    # (logit), switching weight, log pattern sensitivity, left/right lean.
    mu_ideal = pm.Normal("mu_ideal", mu=-0.5, sigma=1.0)
    mu_log_beta = pm.Normal("mu_log_beta", mu=2.0, sigma=1.5)
    mu_sw = pm.Normal("mu_sw", mu=0.4, sigma=0.7)
    mu_kappa = pm.Normal("mu_kappa", mu=0.0, sigma=1.5)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.7)
    sigma_sw = pm.HalfNormal("sigma_sw", sigma=0.7)
    sigma_kappa = pm.HalfNormal("sigma_kappa", sigma=1.0)
    sigma_pat = pm.HalfNormal("sigma_pat", sigma=0.7)
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)

    # Non-centred person effects, one matrix so each trial gathers once.
    z = pm.Normal("z", mu=0.0, sigma=1.0, shape=(N_SLOTS, 6))
    zp = z[participant_id]

    lam = pm.Normal("lam", mu=0.0, sigma=1.0)
    # Shared reward on chunk variety and penalty on rule-built sequences.
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)
    rho = pm.Normal("rho", mu=0.0, sigma=1.0)
    # The change: shared reward on the variety of run lengths.
    eta = pm.Normal("eta", mu=0.0, sigma=1.0)

    theta = pm.math.sigmoid(mu_ideal + sigma_ideal * zp[:, 0])
    sens = pm.math.exp(mu_log_beta + sigma_log_beta * zp[:, 1] + lam * log_len)
    sw_p = pm.math.sigmoid(mu_sw + sigma_sw * zp[:, 2])
    kappa = mu_kappa + sigma_kappa * zp[:, 3]
    pat = pm.math.exp(sigma_pat * zp[:, 4])
    side = sigma_side * zp[:, 5]

    dist_a = (span_a - theta) ** 2
    dist_b = (span_b - theta) ** 2
    sw_dist = (sw_b - sw_p) ** 2 - (sw_a - sw_p) ** 2
    logit = (
        sens * (dist_b - dist_a)
        + 4.0 * kappa * sw_dist
        + pat * (gamma * trip_diff - rho * rule_diff)
        + eta * runvar_diff
        + side
    )
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(logit))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
