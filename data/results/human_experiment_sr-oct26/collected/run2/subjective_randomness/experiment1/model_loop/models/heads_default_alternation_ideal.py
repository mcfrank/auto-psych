"""Heads-as-default asymmetry on a personal alternation ideal.

People do not treat the two faces of the coin symmetrically: heads is the
default, expected outcome of a coin toss, so a tails-dominated sequence reads
as a coin that is "off" and looks less random, while a heads-leaning sequence
looks like ordinary flipping. This operates on top of each person's judgement
of how close a sequence's switching rate is to their own ideal (person-specific,
length-scaled sensitivity; shared penalty for visibly periodic sequences): of
two sequences that switch equally often, people pick the one with more heads.
The claim tested is the shared heads-share weight `delta`.
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

    def heads_excess(seq):
        # Share of heads minus one half: positive = heads-leaning.
        return seq.count("H") / len(seq) - 0.5

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "alt_rate_a": alternation_rate(a),
        "alt_rate_b": alternation_rate(b),
        "periodic_a": periodic(a),
        "periodic_b": periodic(b),
        "heads_excess_diff": heads_excess(a) - heads_excess(b),
        "log_trans": float(np.log((len(a) - 1) / 5.0)),
    }


N_SLOTS = 400

with pm.Model() as model:
    alt_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    per_a = pm.Data("periodic_a", np.zeros(1, dtype="float64"))
    per_b = pm.Data("periodic_b", np.zeros(1, dtype="float64"))
    heads_diff = pm.Data("heads_excess_diff", np.zeros(1, dtype="float64"))
    log_trans = pm.Data("log_trans", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal ideal switching rates (logit scale, population distribution).
    mu_ideal = pm.Normal("mu_ideal", mu=0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Personal sensitivities (log scale, at 5 transitions).
    mu_log_beta = pm.Normal("mu_log_beta", mu=1.5, sigma=1.5)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.7)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pm.Deterministic("beta", pm.math.exp(mu_log_beta + sigma_log_beta * z_beta))

    lam = pm.Normal("lam", mu=0.0, sigma=1.0)
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)
    # Heads-as-default weight (positive: heads-leaning sequences look more random).
    delta = pm.Normal("delta", mu=0.0, sigma=2.0)

    theta = ideal[participant_id]
    dist_a = (alt_a - theta) ** 2
    dist_b = (alt_b - theta) ** 2
    sens = beta[participant_id] * pm.math.exp(lam * log_trans)
    score = sens * (dist_b - dist_a) + gamma * (per_b - per_a) + delta * heads_diff
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
