"""Exemplar similarity on the number line (GCM-style).

People do not induce any rule or interval: they judge a target by its summed
similarity to the shown examples, similarity decaying exponentially with
numeric distance, and say "yes" with probability S / (S + criterion), where S
is the summed similarity; a small lapse rate gives random answers. This
disagrees most with a rule/interval learner in gaps between widely spaced
examples (similarity says no, a spanning interval says yes) and just outside
the examples' range (similarity says yes, the interval says no).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EXAMPLES = 4


def prepare_observed(rows):
    dist = np.zeros((len(rows), MAX_EXAMPLES))
    mask = np.zeros((len(rows), MAX_EXAMPLES))
    for i, r in enumerate(rows):
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        t = int(r["target"])
        for j, x in enumerate(ex[:MAX_EXAMPLES]):
            dist[i, j] = abs(t - x) / 10.0
            mask[i, j] = 1.0
    return {
        "example_dist": dist,
        "example_mask": mask,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    example_dist = pm.Data("example_dist", np.zeros((1, MAX_EXAMPLES)))  # |target - example| / 10
    example_mask = pm.Data("example_mask", np.zeros((1, MAX_EXAMPLES)))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Generalization gradient width (in units of 10 on the number line).
    log_scale = pm.Normal("log_scale", mu=0.0, sigma=1.0)
    # Response criterion against which summed similarity is compared.
    log_criterion = pm.Normal("log_criterion", mu=0.0, sigma=1.5)
    lapse = pm.Beta("lapse", 1, 9)

    scale = pt.exp(log_scale)
    summed = (pt.exp(-example_dist / scale) * example_mask).sum(axis=1)
    p_sim = summed / (summed + pt.exp(log_criterion))
    p = (1 - lapse) * p_sim + lapse * 0.5
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
