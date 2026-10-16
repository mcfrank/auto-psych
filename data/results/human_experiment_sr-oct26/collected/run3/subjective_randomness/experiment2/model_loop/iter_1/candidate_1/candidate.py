"""People commit to a "more random" choice only when the two sequences differ
clearly in how often they switch between heads and tails (each person has a
signed preference for switching); inside an indifference band of switch-rate
difference they have no preference and fall back on their own habitual
response side. Side habits thus appear mainly on pairs with similar switch
counts, spreading people's Left-choice rates."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(sequence_a, sequence_b):
    def switch_rate(seq):
        seq = seq.strip().upper()
        return sum(1 for x, y in zip(seq, seq[1:]) if x != y) / (len(seq) - 1)

    return {"switch_diff": switch_rate(sequence_a) - switch_rate(sequence_b)}


BAND_SHARPNESS = 25.0  # fixed steepness of the smooth indifference band edge

with pm.Model() as model:
    switch_diff = pm.Data("switch_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific signed preference for switching (non-centred).
    mu = pm.Normal("mu", mu=0.0, sigma=3.0)
    sigma = pm.HalfNormal("sigma", sigma=3.0)
    z = pm.Normal("z", mu=0.0, sigma=1.0, shape=400)
    beta = mu + sigma * z

    # Indifference band half-width on switch-rate difference.
    tau = pm.Beta("tau", alpha=2.0, beta=8.0)

    # Person-specific habitual side when undecided (logit scale, non-centred).
    mu_side = pm.Normal("mu_side", mu=0.0, sigma=1.0)
    sigma_side = pm.HalfNormal("sigma_side", sigma=1.0)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=400)
    side = pm.math.sigmoid(mu_side + sigma_side * z_side)

    pid = participant_id
    commit = pm.math.sigmoid(BAND_SHARPNESS * (pt.abs(switch_diff) - tau))
    judged = pm.math.sigmoid(beta[pid] * switch_diff)
    p = commit * judged + (1.0 - commit) * side[pid]
    p_left = pm.Deterministic("p_left", pt.clip(p, 1e-6, 1 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
