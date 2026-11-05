"""Place-value exemplar similarity (GCM-style).

People do not induce a rule; they judge a target by its summed similarity to
the remembered examples, where similarity between two numbers is computed in a
two-dimensional psychological space of overall magnitude and final digit.
Similarity decays exponentially with numeric distance and drops by a fixed
factor when final digits differ; greater summed similarity -> more likely yes.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4


def compute_features(number_set, target):
    ex = [int(x) for x in number_set.split(",")][:MAXK]
    t = int(target)
    out = {}
    for k in range(MAXK):
        if k < len(ex):
            out[f"dist_{k}"] = abs(t - ex[k]) / 10.0
            out[f"unit_diff_{k}"] = 0.0 if t % 10 == ex[k] % 10 else 1.0
            out[f"present_{k}"] = 1.0
        else:
            out[f"dist_{k}"] = 0.0
            out[f"unit_diff_{k}"] = 0.0
            out[f"present_{k}"] = 0.0
    return out


with pm.Model() as model:
    dist = pt.stack([pm.Data(f"dist_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    udiff = pt.stack([pm.Data(f"unit_diff_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    present = pt.stack([pm.Data(f"present_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)

    # Generalization gradient on magnitude (per 10 units) and the log-penalty
    # for a final-digit mismatch.
    c_mag = pm.HalfNormal("c_mag", sigma=2.0)
    c_unit = pm.HalfNormal("c_unit", sigma=3.0)
    # Response mapping: baseline and sensitivity to log summed similarity.
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=2.0)

    log_sim = -c_mag * dist - c_unit * udiff
    log_sim = pt.switch(present > 0.5, log_sim, -1e3)
    log_summed = pm.math.logsumexp(log_sim, axis=1).flatten()

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(alpha + beta * log_summed))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
