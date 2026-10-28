"""Weber-compressed nearest-example similarity.

People do not infer a rule; they judge the target by how close it is on a
compressed (logarithmic, Weber-like) mental number line to the nearest example,
so a gap of 5 feels large between 3 and 8 but small between 83 and 88.
Endorsement falls off exponentially with this compressed distance, and the
generalization gradient narrows as more examples are shown; apart from
occasional blind guesses, "yes" follows this ratio-scale similarity to the
closest example.
"""
import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(number_set, target):
    examples = [int(x) for x in str(number_set).split(",") if x.strip()]
    t = int(target)
    d = min(abs(math.log(t) - math.log(x)) for x in examples)
    return {"log_ratio_dist": float(d), "log_set_size": math.log(len(examples))}


with pm.Model() as model:
    log_ratio_dist = pm.Data("log_ratio_dist", np.zeros(1, dtype="float64"))
    log_set_size = pm.Data("log_set_size", np.zeros(1, dtype="float64"))

    # Gradient width (in log-ratio units) for a single example, and how fast it
    # narrows with set size.
    log_width = pm.Normal("log_width", mu=math.log(0.2), sigma=1.0)
    narrowing = pm.Normal("narrowing", mu=0.5, sigma=0.5)
    # Blind guesses: how often, and how often a guess is "yes".
    guess_rate = pm.Beta("guess_rate", alpha=2.0, beta=8.0)
    guess_yes = pm.Beta("guess_yes", alpha=2.0, beta=2.0)

    width = pt.exp(log_width - narrowing * log_set_size)
    sim = pt.exp(-log_ratio_dist / width)
    p = guess_rate * guess_yes + (1.0 - guess_rate) * sim
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
