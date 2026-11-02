"""Digit-wise exemplar similarity (GCM-style) hypothesis.

People do not induce a rule; they judge a target by its summed similarity to
the remembered examples, representing each number by its two decimal digits.
Distance between target and example is an attention-weighted city-block
distance: the gap between tens digits (magnitude) and whether the ones digits
differ. Similarity decays exponentially with distance, and the probability of
"yes" rises with the target's mean similarity to the examples.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4


def _digits(n):
    return n // 10, n % 10  # 100 -> tens 10, ones 0


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    tt, to = _digits(t)
    out = {}
    for i in range(MAXK):
        if i < len(xs):
            xt, xo = _digits(xs[i])
            out[f"dtens_{i}"] = abs(tt - xt) / 9.0
            out[f"dones_{i}"] = 0.0 if to == xo else 1.0
            out[f"mask_{i}"] = 1.0
        else:
            out[f"dtens_{i}"] = 0.0
            out[f"dones_{i}"] = 0.0
            out[f"mask_{i}"] = 0.0
    return out


with pm.Model() as model:
    dtens = [pm.Data(f"dtens_{i}", np.zeros(1, dtype="float64")) for i in range(MAXK)]
    dones = [pm.Data(f"dones_{i}", np.zeros(1, dtype="float64")) for i in range(MAXK)]
    mask = [pm.Data(f"mask_{i}", np.zeros(1, dtype="float64")) for i in range(MAXK)]

    # attention to the tens (magnitude) dimension vs the ones (final digit) one
    w_tens = pm.Beta("w_tens", alpha=2.0, beta=2.0)
    # similarity gradient (specificity)
    c = pm.HalfNormal("c", sigma=5.0)
    # response mapping: gain and criterion on mean similarity
    gain = pm.HalfNormal("gain", sigma=5.0)
    crit = pm.Normal("crit", mu=0.3, sigma=0.5)

    D = pt.stack([w_tens * dtens[i] + (1.0 - w_tens) * dones[i] for i in range(MAXK)])
    M = pt.stack(mask)
    sim = pt.exp(-c * D) * M
    mean_sim = pt.sum(sim, axis=0) / pt.maximum(pt.sum(M, axis=0), 1.0)

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(gain * (mean_sim - crit)), 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
