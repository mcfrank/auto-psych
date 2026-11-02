"""Digit-overlap exemplar similarity.

People do not induce mathematical rules or magnitude intervals; they judge a
target by its surface resemblance to the examples, treating each number as the
set of decimal digits it is written with, and say "yes" in proportion to the
average digit-overlap similarity between target and examples.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EX = 4


def _digits(n):
    return set(str(int(n)))


def compute_features(number_set, target):
    examples = [int(x) for x in number_set.split(",")]
    td = _digits(target)
    out = {}
    for i in range(MAX_EX):
        if i < len(examples):
            out[f"dd_{i}"] = float(len(td ^ _digits(examples[i])))
            out[f"mask_{i}"] = 1.0
        else:
            out[f"dd_{i}"] = 0.0
            out[f"mask_{i}"] = 0.0
    return out


with pm.Model() as model:
    dd = [pm.Data(f"dd_{i}", np.zeros(1, dtype="float64")) for i in range(MAX_EX)]
    mask = [pm.Data(f"mask_{i}", np.zeros(1, dtype="float64")) for i in range(MAX_EX)]

    # Similarity decay per mismatched digit, baseline yes tendency, gain.
    decay = pm.HalfNormal("decay", sigma=2.0)
    bias = pm.Normal("bias", mu=-1.5, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)

    D = pt.stack(dd, axis=0)
    M = pt.stack(mask, axis=0)
    sim = (M * pt.exp(-decay * D)).sum(axis=0) / pt.maximum(M.sum(axis=0), 1.0)

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(bias + gain * sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
