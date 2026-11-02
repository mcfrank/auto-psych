"""Exemplar digit-feature similarity.

People judge membership by surface-feature similarity to the individual
examples, not by inducing a rule shared by the whole set: they compare the
target with each example on a few salient features (last digit, tens digit,
parity, remainder mod 3); similarity to an example falls off exponentially
with the number of mismatching features, and the yes probability grows with
the target's average similarity to the examples.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EX = 4


def _mismatches(t, x):
    return float((t % 10 != x % 10) + ((t // 10) != (x // 10))
                 + (t % 2 != x % 2) + (t % 3 != x % 3))


def prepare_observed(rows):
    n = len(rows)
    mis = np.zeros((n, MAX_EX), dtype="float64")
    mask = np.zeros((n, MAX_EX), dtype="float64")
    rating = np.zeros(n, dtype="int64")
    for i, r in enumerate(rows):
        ex = [int(v) for v in str(r["number_set"]).split(",") if v.strip()]
        t = int(float(r["target"]))
        for j, x in enumerate(ex[:MAX_EX]):
            mis[i, j] = _mismatches(t, x)
            mask[i, j] = 1.0
        rating[i] = int(float(r["rating"]))
    return {"mismatch_counts": mis, "example_mask": mask, "rating": rating}


with pm.Model() as model:
    mis = pm.Data("mismatch_counts", np.zeros((1, MAX_EX), dtype="float64"))
    mask = pm.Data("example_mask", np.ones((1, MAX_EX), dtype="float64"))

    # How fast similarity decays per mismatching feature.
    decay = pm.HalfNormal("decay", sigma=2.0)
    # Baseline log-odds of yes at zero similarity, and gain on similarity.
    intercept = pm.Normal("intercept", mu=-2.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    sim = pt.exp(-decay * mis) * mask
    mean_sim = pt.sum(sim, axis=1) / pt.maximum(pt.sum(mask, axis=1), 1.0)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(intercept + gain * mean_sim), 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
