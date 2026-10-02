"""Personal switch-rate prototype.

Each person carries their own prototype of how often a genuinely random coin
switches between heads and tails, and judges a sequence as random to the extent
its switch rate is close to that personal prototype. People differ in this
prototype, so the same pair can be judged in opposite directions by different
participants.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

# Upper bound on participant ids (ids are unique across experiments); unused
# slots are sampled from the population prior and do not touch the likelihood.
MAX_PARTICIPANTS = 512


def compute_features(sequence_a, sequence_b):
    """Each sequence's switch rate: share of adjacent flips that differ."""

    def switch_rate(seq):
        seq = seq.strip().upper()
        if len(seq) < 2:
            return 0.5
        return sum(1 for x, y in zip(seq, seq[1:]) if x != y) / (len(seq) - 1)

    return {"switch_a": switch_rate(sequence_a), "switch_b": switch_rate(sequence_b)}


with pm.Model() as model:
    switch_a = pm.Data("switch_a", np.zeros(1, dtype="float64"))
    switch_b = pm.Data("switch_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Population prototype switch rate (logit scale) and its spread across people.
    mu_proto = pm.Normal("mu_proto", mu=0.4, sigma=1.0)
    sigma_proto = pm.HalfNormal("sigma_proto", sigma=1.0)
    z_proto = pm.Normal("z_proto", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    proto = pm.math.sigmoid(mu_proto + sigma_proto * z_proto)[participant_id]

    # Sensitivity to squared distance from the personal prototype.
    beta = pm.HalfNormal("beta", sigma=10.0)

    dist_a = pt.sqr(switch_a - proto)
    dist_b = pt.sqr(switch_b - proto)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * (dist_b - dist_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
