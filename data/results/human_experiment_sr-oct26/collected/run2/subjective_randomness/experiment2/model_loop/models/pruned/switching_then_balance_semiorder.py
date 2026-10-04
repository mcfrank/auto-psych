"""Lexicographic semiorder: switching first, H/T balance as the tie-breaker.

People compare the two sequences first on how close each one's switching rate
is to their own ideal for a random coin (person-specific ideal; shared,
length-scaled sensitivity). Only when the two sequences switch about equally
often does a second comparison take over: the sequence with the more even
heads/tails count looks more random. The balance cue is gated by the pair's
difference in switch counts (full weight at equal counts, halving with each
switch of difference), so the same sequence's balance matters beside one
partner and not beside another. The gate is computed from the stimulus, once,
so the model stays cheap and well identified.
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def switches(seq):
        return sum(1 for x, y in zip(seq, seq[1:]) if x != y)

    def imbalance(seq):
        # |#H - #T| as a share of the sequence length.
        return abs(seq.count("H") - seq.count("T")) / len(seq)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    n_trans = len(a) - 1
    gate = 0.5 ** abs(switches(a) - switches(b))
    return {
        "alt_rate_a": switches(a) / n_trans,
        "alt_rate_b": switches(b) / n_trans,
        "gated_balance_diff": gate * (imbalance(b) - imbalance(a)),
        "log_trans": float(np.log(n_trans / 5.0)),
    }


N_SLOTS = 400

with pm.Model() as model:
    alt_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    gated_bal = pm.Data("gated_balance_diff", np.zeros(1, dtype="float64"))
    log_trans = pm.Data("log_trans", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal ideal switching rates (logit scale, non-centred population).
    mu_ideal = pm.Normal("mu_ideal", mu=0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal)

    # Shared sensitivity to the switching comparison, scaled with length.
    log_beta = pm.Normal("log_beta", mu=1.5, sigma=1.0)
    lam = pm.Normal("lam", mu=0.0, sigma=1.0)

    # Second-stage balance comparison (consulted through the stimulus gate).
    kappa = pm.Normal("kappa", mu=0.0, sigma=3.0)

    # Response-level nuisance: a shared lean toward the Left button.
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=0.5)

    theta = ideal[participant_id]
    sens = pm.math.exp(log_beta + lam * log_trans)
    switch_diff = sens * ((alt_b - theta) ** 2 - (alt_a - theta) ** 2)
    score = switch_diff + kappa * gated_bal + side_bias
    p_left = pm.Deterministic("p_left", pm.math.clip(pm.math.sigmoid(score), 1e-6, 1 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
