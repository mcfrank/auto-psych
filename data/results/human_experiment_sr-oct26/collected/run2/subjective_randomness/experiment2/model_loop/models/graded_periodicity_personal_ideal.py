"""Personal alternation ideal with a graded near-periodicity penalty.

Refinement of `person_sensitivity_length_scaled_ideal`: each person judges a
sequence as random by how close its proportion of H/T switches is to their own
ideal switching rate (person-specific sensitivity, scaled as a fitted power of
the number of transitions). The single change: the shared penalty for visibly
periodic sequences is graded — a sequence that repeats a short unit (period
1-4) with m flips out of place is penalised by gamma * exp(-kappa * m), so
almost-repeating patterns look non-random too, less so the more they break.
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def alternation_rate(seq):
        switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        return switches / (len(seq) - 1)

    def min_mismatches(seq):
        # Fewest flips that break "each flip copies the one p places back",
        # over short periods p (1-4) that repeat at least twice.
        n = len(seq)
        best = n
        for p in range(1, min(4, n // 2) + 1):
            m = sum(1 for i in range(n - p) if seq[i] != seq[i + p])
            best = min(best, m)
        return float(best)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "alt_rate_a": alternation_rate(a),
        "alt_rate_b": alternation_rate(b),
        "mismatch_a": min_mismatches(a),
        "mismatch_b": min_mismatches(b),
        "log_trans": float(np.log((len(a) - 1) / 5.0)),
    }


N_SLOTS = 400

with pm.Model() as model:
    alt_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    mis_a = pm.Data("mismatch_a", np.zeros(1, dtype="float64"))
    mis_b = pm.Data("mismatch_b", np.zeros(1, dtype="float64"))
    log_trans = pm.Data("log_trans", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Population distribution of personal ideal switching rates (logit scale).
    mu_ideal = pm.Normal("mu_ideal", mu=0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Population distribution of personal sensitivities (log scale, at 5 transitions).
    mu_log_beta = pm.Normal("mu_log_beta", mu=1.5, sigma=1.5)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.7)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pm.Deterministic("beta", pm.math.exp(mu_log_beta + sigma_log_beta * z_beta))

    # Power-law exponent: how sensitivity scales with number of transitions.
    lam = pm.Normal("lam", mu=0.0, sigma=1.0)
    # Shared pattern penalty and its fall-off per flip that breaks the pattern.
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)
    kappa = pm.LogNormal("kappa", mu=0.0, sigma=0.75)

    pat_a = pm.math.exp(-kappa * mis_a)
    pat_b = pm.math.exp(-kappa * mis_b)

    theta = ideal[participant_id]
    dist_a = (alt_a - theta) ** 2
    dist_b = (alt_b - theta) ** 2
    sens = beta[participant_id] * pm.math.exp(lam * log_trans)
    score = sens * (dist_b - dist_a) + gamma * (pat_b - pat_a)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
