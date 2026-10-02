"""Length-scaled personal ideal alternation plus a balance expectation.

Refinement of `ideal_alternation_with_balance`: each person judges a sequence
as more random the closer its proportion of alternations lies to their own
ideal switching rate, and a shared weight penalises H/T imbalance. The one
change: a departure from the ideal rate counts as evidence that grows with the
number of transitions it is observed over, so alternation sensitivity scales
as ((length - 1) / 7) ** kappa, with kappa fitted (kappa = 0 is the incumbent).
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
    if len(a) != len(b):
        raise ValueError(f"sequences differ in length: {a!r} vs {b!r}")
    return {
        "alt_rate_a": alternation_rate(a),
        "alt_rate_b": alternation_rate(b),
        "imbalance_a": imbalance(a),
        "imbalance_b": imbalance(b),
        # Log of transitions relative to a length-8 sequence (0 at length 8).
        "log_rel_transitions": float(np.log((len(a) - 1) / 7.0)),
    }


with pm.Model() as model:
    alt_rate_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
    alt_rate_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
    imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
    imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
    log_rel_transitions = pm.Data("log_rel_transitions", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic(
        "ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal)
    )
    # Sensitivity to squared distance from the ideal at length 8.
    beta = pm.LogNormal("beta", mu=2.0, sigma=1.0)
    # How sensitivity grows with the number of transitions (1: binomial evidence).
    kappa = pm.Normal("kappa", mu=1.0, sigma=0.5)
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)

    beta_len = beta * pt.exp(kappa * log_rel_transitions)
    theta = ideal[participant_id]
    score_a = -beta_len * pt.sqr(alt_rate_a - theta) - gamma * imbalance_a
    score_b = -beta_len * pt.sqr(alt_rate_b - theta) - gamma * imbalance_b
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score_a - score_b))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
