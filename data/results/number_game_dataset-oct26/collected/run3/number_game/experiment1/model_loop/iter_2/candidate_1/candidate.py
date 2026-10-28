"""Digit-place similarity gradient.

People judge a target by its similarity to the examples in their written
decimal form rather than on the number line: they compare the target with each
example place by place, and every mismatch in the units digit or in the tens
digit multiplies similarity down by its own inferred factor. The more similar
the target is on average to the examples under this digit-place metric, the
more likely people are to say "yes".
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EX = 4


def _digits(n):
    """(tens, units) of n; 100 has tens 10, single digits have tens 0."""
    return n // 10, n % 10


def compute_features(number_set, target):
    examples = [int(x) for x in str(number_set).split(",")][:MAX_EX]
    tt, tu = _digits(int(target))
    out = {}
    for k in range(MAX_EX):
        if k < len(examples):
            et, eu = _digits(examples[k])
            out[f"units_miss_{k}"] = float(eu != tu)
            out[f"tens_miss_{k}"] = float(et != tt)
            out[f"present_{k}"] = 1.0
        else:
            out[f"units_miss_{k}"] = 0.0
            out[f"tens_miss_{k}"] = 0.0
            out[f"present_{k}"] = 0.0
    return out


with pm.Model() as model:
    units_miss = [pm.Data(f"units_miss_{k}", np.zeros(1, dtype="float64")) for k in range(MAX_EX)]
    tens_miss = [pm.Data(f"tens_miss_{k}", np.zeros(1, dtype="float64")) for k in range(MAX_EX)]
    present = [pm.Data(f"present_{k}", np.zeros(1, dtype="float64")) for k in range(MAX_EX)]

    # Penalty (in log-similarity) for a mismatched units / tens digit.
    w_units = pm.HalfNormal("w_units", sigma=3.0)
    w_tens = pm.HalfNormal("w_tens", sigma=3.0)
    # Log-odds of "yes" for a target sharing no digit place with any example,
    # and how strongly mean digit similarity raises it.
    intercept = pm.Normal("intercept", mu=-1.0, sigma=2.0)
    slope = pm.HalfNormal("slope", sigma=5.0)

    sim_sum = 0.0
    n_present = 0.0
    for k in range(MAX_EX):
        sim_k = pt.exp(-(w_units * units_miss[k] + w_tens * tens_miss[k]))
        sim_sum = sim_sum + present[k] * sim_k
        n_present = n_present + present[k]
    mean_sim = sim_sum / pt.maximum(n_present, 1.0)

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
