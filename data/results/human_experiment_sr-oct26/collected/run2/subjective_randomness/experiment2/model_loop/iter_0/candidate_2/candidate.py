"""Lexicographic semiorder: switching first, H/T balance as the tie-breaker.

People compare the two sequences first on how close each one's switching rate
is to their own ideal for a random coin (person-specific ideal and sensitivity,
sensitivity scaling with the number of transitions), and act on that
difference when it is noticeable. Only when the pair is close on switching does
a second comparison take over: the sequence with the more even heads/tails
count looks more random. The balance cue's weight is gated by the size of the
switching difference within the pair, so the same sequence's balance matters
beside one partner and not beside another. The claim tested is the gated
balance comparison (`kappa`, `tau`).
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def alternation_rate(seq):
        switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        return switches / (len(seq) - 1)

    def imbalance(seq):
        # |#H - #T| as a share of the sequence length.
        return abs(seq.count("H") - seq.count("T")) / len(seq)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "alt_rate_a": alternation_rate(a),
        "alt_rate_b": alternation_rate(b),
        "imbalance_diff": imbalance(b) - imbalance(a),
        "log_trans": float(np.log((len(a) - 1) / 5.0)),
    }


N_SLOTS = 400

with pm.Model() as model:
    alt_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    imb_diff = pm.Data("imbalance_diff", np.zeros(1, dtype="float64"))
    log_trans = pm.Data("log_trans", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal ideal switching rates (logit scale, population distribution).
    mu_ideal = pm.Normal("mu_ideal", mu=0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Personal sensitivities to the switching comparison (log scale).
    mu_log_beta = pm.Normal("mu_log_beta", mu=1.5, sigma=1.5)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.7)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pm.Deterministic("beta", pm.math.exp(mu_log_beta + sigma_log_beta * z_beta))
    lam = pm.Normal("lam", mu=0.0, sigma=1.0)

    # Second-stage balance comparison and the noticeability scale of the
    # switching difference that gates it.
    kappa = pm.Normal("kappa", mu=0.0, sigma=3.0)
    log_tau = pm.Normal("log_tau", mu=0.0, sigma=1.0)
    tau = pm.Deterministic("tau", pm.math.exp(log_tau))

    # Response-level nuisance: a shared lean toward the Left button (not a
    # randomness cue; keeps mirror-image pairs from a constant 0.5).
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=0.5)

    theta = ideal[participant_id]
    sens = beta[participant_id] * pm.math.exp(lam * log_trans)
    switch_diff = sens * ((alt_b - theta) ** 2 - (alt_a - theta) ** 2)
    gate = pm.math.exp(-((switch_diff / tau) ** 2))
    score = switch_diff + gate * kappa * imb_diff + side_bias
    p_left = pm.Deterministic("p_left", pm.math.clip(pm.math.sigmoid(score), 1e-6, 1 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
