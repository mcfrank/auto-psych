"""Personal ideal alternation rate plus a balance expectation.

Refinement of `personal_ideal_alternation`: each person judges a sequence as
more random the closer its proportion of alternations lies to their own ideal
switching rate (hierarchical, non-centred logit-normal population), and people
additionally penalise H/T imbalance (a shared weight on |#H - #T| / length), so
among pairs matched on alternation the more balanced sequence looks more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

# Upper bound on participant ids (ids are unique across a run's experiments);
# an id at or beyond it fails loudly at indexing.
MAX_PARTICIPANTS = 400


def compute_features(sequence_a, sequence_b):
    def alternation_rate(seq):
        if len(seq) < 2:
            raise ValueError(f"sequence too short: {seq!r}")
        return sum(1 for x, y in zip(seq, seq[1:]) if x != y) / (len(seq) - 1)

    def imbalance(seq):
        return abs(seq.count("H") - seq.count("T")) / len(seq)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "alt_rate_a": alternation_rate(a),
        "alt_rate_b": alternation_rate(b),
        "imbalance_a": imbalance(a),
        "imbalance_b": imbalance(b),
    }


with pm.Model() as model:
    alt_rate_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_rate_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
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
    # Shared penalty on H/T imbalance (positive: lopsided counts look less random).
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)

    theta = ideal[participant_id]
    score_a = -beta * pt.sqr(alt_rate_a - theta) - gamma * imbalance_a
    score_b = -beta * pt.sqr(alt_rate_b - theta) - gamma * imbalance_b
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score_a - score_b))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
