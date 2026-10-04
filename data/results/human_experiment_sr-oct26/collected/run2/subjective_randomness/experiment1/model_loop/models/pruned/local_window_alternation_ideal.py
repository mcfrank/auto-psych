"""Local-window personal alternation ideal, length-scaled, with a periodic penalty.

Refinement of `length_scaled_alternation_ideal`: each person judges a sequence
as random by how close its switching is to their own ideal switching rate
(sensitivity scaling as a power of the number of transitions), with a shared
penalty for visibly periodic sequences. The single change: the ideal is
applied locally -- every three-flip stretch (two transitions) is compared with
the personal ideal and the squared distances are averaged -- so switches
bunched together around a long streak count against randomness even when the
total number of switches is the same.
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def local_rates(seq):
        sw = [1.0 if x != y else 0.0 for x, y in zip(seq, seq[1:])]
        if len(sw) < 2:
            return sw
        return [(sw[i] + sw[i + 1]) / 2.0 for i in range(len(sw) - 1)]

    def periodic(seq):
        n = len(seq)
        for p in range(1, n // 2 + 1):
            if all(seq[i] == seq[i + p] for i in range(n - p)):
                return 1.0
        return 0.0

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    ra, rb = local_rates(a), local_rates(b)
    return {
        # mean and mean square of local (3-flip window) switch rates, so that
        # mean((r - theta)^2) = m2 - 2 theta m1 + theta^2
        "loc_m1_a": float(np.mean(ra)),
        "loc_m1_b": float(np.mean(rb)),
        "loc_m2_a": float(np.mean(np.square(ra))),
        "loc_m2_b": float(np.mean(np.square(rb))),
        "periodic_a": periodic(a),
        "periodic_b": periodic(b),
        # log of transitions relative to a 5-transition (6-flip) reference
        "log_trans": float(np.log((len(a) - 1) / 5.0)),
    }


N_SLOTS = 400

with pm.Model() as model:
    m1_a = pm.Data("loc_m1_a", np.zeros(1, dtype="float64"))
    m1_b = pm.Data("loc_m1_b", np.zeros(1, dtype="float64"))
    m2_a = pm.Data("loc_m2_a", np.zeros(1, dtype="float64"))
    m2_b = pm.Data("loc_m2_b", np.zeros(1, dtype="float64"))
    per_a = pm.Data("periodic_a", np.zeros(1, dtype="float64"))
    per_b = pm.Data("periodic_b", np.zeros(1, dtype="float64"))
    log_trans = pm.Data("log_trans", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Population distribution of personal ideal switching rates (logit scale).
    mu_ideal = pm.Normal("mu_ideal", mu=0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Sensitivity to mean squared local distance from the ideal (at 5 transitions).
    beta = pm.HalfNormal("beta", sigma=10.0)
    # Power-law exponent: how sensitivity scales with number of transitions.
    lam = pm.Normal("lam", mu=0.0, sigma=1.0)
    # Shared penalty for a visibly periodic sequence (positive: less random).
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)

    theta = ideal[participant_id]
    dist_a = m2_a - 2.0 * theta * m1_a + theta ** 2
    dist_b = m2_b - 2.0 * theta * m1_b + theta ** 2
    sens = beta * pm.math.exp(lam * log_trans)
    score = sens * (dist_b - dist_a) + gamma * (per_b - per_a)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
