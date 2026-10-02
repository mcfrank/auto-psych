"""Divisive pairwise contrast of non-randomness.

People judge the two sequences against each other by divisive contrast: the
difference in their non-randomness (departure from the person's own ideal
switching rate plus H/T imbalance) is scaled by the pair's total
non-randomness, so a given difference decides the choice strongly when the
partner is nearly ideal and weakly when both look clearly non-random. The same
sequence is thus judged differently beside a different partner.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

# Upper bound on participant ids (ids are unique across a run's experiments).
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

    # Personal ideal alternation rate (non-centred logit-normal population).
    mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Weight of H/T imbalance in a sequence's non-randomness (non-negative).
    gamma = pm.HalfNormal("gamma", sigma=1.0)
    # Contrast gain and the semi-saturation constant of the divisive normalisation
    # (small: strongly relative judgments; large: nearly absolute differences).
    beta = pm.LogNormal("beta", mu=1.0, sigma=1.0)
    kappa = pm.LogNormal("kappa", mu=np.log(0.1), sigma=1.0)

    theta = ideal[participant_id]
    dev_a = pt.sqr(alt_rate_a - theta) + gamma * imbalance_a
    dev_b = pt.sqr(alt_rate_b - theta) + gamma * imbalance_b
    contrast = (dev_b - dev_a) / (kappa + dev_a + dev_b)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * contrast))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
