"""Average digit-and-magnitude similarity to the examples.

People do not infer a rule; they judge a target by its average similarity to
the examples, where two numbers are similar if they are close in magnitude and
share their final digit. Similarity decays exponentially with numerical
distance and drops by a fixed factor when final digits differ. Similarity is
averaged (not summed) over examples, so "yes" depends on the proportion of the
examples the target resembles, not on how many were shown.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EX = 4


def prepare_observed(rows):
    n = len(rows)
    digit_mismatch = np.zeros((n, MAX_EX))
    distance = np.zeros((n, MAX_EX))
    example_mask = np.zeros((n, MAX_EX))
    rating = np.zeros(n, dtype="int64")
    for i, r in enumerate(rows):
        t = int(r["target"])
        for j, e in enumerate(int(x) for x in str(r["number_set"]).split(",")):
            digit_mismatch[i, j] = float(t % 10 != e % 10)
            distance[i, j] = abs(t - e) / 100.0
            example_mask[i, j] = 1.0
        rating[i] = int(float(r["rating"]))
    return {
        "digit_mismatch": digit_mismatch,
        "distance": distance,
        "example_mask": example_mask,
        "rating": rating,
    }


with pm.Model() as model:
    digit_mismatch = pm.Data("digit_mismatch", np.zeros((1, MAX_EX)))
    distance = pm.Data("distance", np.zeros((1, MAX_EX)))
    example_mask = pm.Data("example_mask", np.ones((1, MAX_EX)))

    # Baseline log-odds of "yes", gain on average similarity, the cost of a
    # final-digit mismatch and the decay rate along the number line.
    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    digit_cost = pm.HalfNormal("digit_cost", sigma=3.0)
    decay = pm.HalfNormal("decay", sigma=5.0)

    sim = pt.exp(-digit_cost * digit_mismatch - decay * distance) * example_mask
    mean_sim = sim.sum(axis=1) / pt.maximum(example_mask.sum(axis=1), 1.0)

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
