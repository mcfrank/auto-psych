"""Length-scaled personal alternation ideal with periodic and H/T-imbalance penalties.

Refinement of `length_scaled_alternation_ideal`: each person judges a sequence
as random by how close its proportion of H/T switches is to their own ideal
switching rate (sensitivity scaling as a power of the number of transitions),
with a shared penalty for visibly periodic sequences. The single change: a
shared penalty on how lopsided the sequence's heads/tails count is (squared
deviation of the share of heads from one half), since people expect a random
coin to give about equal numbers of each.
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def alternation_rate(seq):
        switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        return switches / (len(seq) - 1)

    def periodic(seq):
        n = len(seq)
        for p in range(1, n // 2 + 1):
            if all(seq[i] == seq[i + p] for i in range(n - p)):
                return 1.0
        return 0.0

    def imbalance(seq):
        # squared deviation of the share of heads from 1/2, scaled to [0, 1]
        return float((2.0 * seq.count("H") / len(seq) - 1.0) ** 2)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "alt_rate_a": alternation_rate(a),
        "alt_rate_b": alternation_rate(b),
        "periodic_a": periodic(a),
        "periodic_b": periodic(b),
        "imbalance_a": imbalance(a),
        "imbalance_b": imbalance(b),
        # log of transitions relative to a 5-transition (6-flip) reference
        "log_trans": float(np.log((len(a) - 1) / 5.0)),
    }


N_SLOTS = 400

with pm.Model() as model:
    alt_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    per_a = pm.Data("periodic_a", np.zeros(1, dtype="float64"))
    per_b = pm.Data("periodic_b", np.zeros(1, dtype="float64"))
    imb_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imb_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    log_trans = pm.Data("log_trans", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Population distribution of personal ideal switching rates (logit scale).
    mu_ideal = pm.Normal("mu_ideal", mu=0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Sensitivity to squared distance from the personal ideal (at 5 transitions).
    beta = pm.HalfNormal("beta", sigma=10.0)
    # Power-law exponent: how sensitivity scales with number of transitions.
    lam = pm.Normal("lam", mu=0.0, sigma=1.0)
    # Shared penalty for a visibly periodic sequence (positive: less random).
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)
    # Shared penalty for H/T imbalance (positive: lopsided looks less random).
    kappa = pm.Normal("kappa", mu=0.0, sigma=2.0)

    theta = ideal[participant_id]
    dist_a = (alt_a - theta) ** 2
    dist_b = (alt_b - theta) ** 2
    sens = beta * pm.math.exp(lam * log_trans)
    score = (
        sens * (dist_b - dist_a)
        + gamma * (per_b - per_a)
        + kappa * (imb_b - imb_a)
    )
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
