"""Personal ideal alternation plus an H/T balance expectation.

Refinement of `personal_ideal_alternation`: each person judges a sequence as
more random the closer its proportion of alternations lies to their own ideal
switching rate (hierarchical, non-centred logit-normal population of ideals),
and additionally everyone expects a random coin to give roughly equal numbers
of heads and tails, so lopsided H/T counts reduce perceived randomness. The one
change is the shared balance penalty on |#H - #T| / length.
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

    def imbalance(seq):
        seq = seq.strip().upper()
        return abs(seq.count("H") - seq.count("T")) / len(seq)

    return {
        "alt_rate_a": alternation_rate(sequence_a),
        "alt_rate_b": alternation_rate(sequence_b),
        "imbalance_a": imbalance(sequence_a),
        "imbalance_b": imbalance(sequence_b),
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
    # Shared penalty on H/T imbalance (positive: lopsided looks less random).
    gamma = pm.Normal("gamma", mu=0.0, sigma=3.0)

    theta = ideal[participant_id]
    score_a = -beta * pt.sqr(alt_rate_a - theta) - gamma * imbalance_a
    score_b = -beta * pt.sqr(alt_rate_b - theta) - gamma * imbalance_b
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score_a - score_b))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
