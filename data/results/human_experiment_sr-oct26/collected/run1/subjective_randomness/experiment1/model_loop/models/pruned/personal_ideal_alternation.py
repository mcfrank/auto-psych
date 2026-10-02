"""Personal ideal alternation rate.

Each person carries their own ideal switching rate for a random coin (how often
consecutive flips should differ) and judges a sequence as more random the closer
its proportion of alternations lies to that personal ideal. Ideals differ
between people (a hierarchical, non-centred logit-normal population), so the
model predicts individual differences in alternation preference: some people
favour near-perfect alternation, others streakier sequences.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

# Upper bound on participant ids (ids are unique across a run's experiments);
# an id at or beyond it fails loudly at indexing.
MAX_PARTICIPANTS = 400


def compute_features(sequence_a, sequence_b):
    def alternation_rate(seq):
        seq = seq.strip().upper()
        if len(seq) < 2:
            raise ValueError(f"sequence too short: {seq!r}")
        return sum(1 for x, y in zip(seq, seq[1:]) if x != y) / (len(seq) - 1)

    return {"alt_rate_a": alternation_rate(sequence_a), "alt_rate_b": alternation_rate(sequence_b)}


with pm.Model() as model:
    alt_rate_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_rate_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Population ideal alternation rate (logit scale) and its spread.
    mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic(
        "ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal)
    )
    # Sensitivity to squared distance from the ideal.
    beta = pm.LogNormal("beta", mu=2.0, sigma=1.0)

    theta = ideal[participant_id]
    score_a = -pt.sqr(alt_rate_a - theta)
    score_b = -pt.sqr(alt_rate_b - theta)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * (score_a - score_b)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
