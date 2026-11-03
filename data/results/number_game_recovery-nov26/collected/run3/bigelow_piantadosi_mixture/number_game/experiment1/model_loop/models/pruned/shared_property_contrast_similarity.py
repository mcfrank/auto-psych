"""Shared-property (Tversky-style) similarity to the example set.

People do not infer a single hidden concept; they judge a target by its
feature-based similarity to the examples as a group. Each number is
represented by its mathematical properties (parity, primality, squares,
cubes, powers, multiples, final digit, digits contained) and by the magnitude
bands it falls in (sliding windows of width 5, 10, 20 and 40). A property's
salience grows with the fraction of examples that share it (raised to an
inferred commonality exponent) and with its distinctiveness (its size in 1-100
raised to an inferred negative exponent). The similarity of the target is the
salience-weighted share of properties the target also has, and the
probability of "yes" rises logistically with it.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


def _features():
    f = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "even": list(range(2, 101, 2)),
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
    }
    f.update({f"mult{m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    f.update({f"pow{m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    f.update({f"end{d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    f.update({f"has{d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(1, 10)})
    for w in (5, 10, 20, 40):
        f.update({f"band{w}_{lo}": list(range(lo, lo + w)) for lo in range(1, 102 - w)})
    return np.array([np.isin(NUMBERS, c) for c in f.values()], dtype="float64")  # (F, 100)


MEMBERSHIP = _features()
LOG_SIZE = np.log(MEMBERSHIP.sum(axis=1))


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    share = np.stack([MEMBERSHIP[:, np.array(s) - 1].mean(axis=1) for s in unique])  # (sets, F)
    present = (share > 0).astype("float64")
    log_share = np.where(share > 0, np.log(np.maximum(share, 1e-12)), 0.0)
    return {
        "log_share": log_share,
        "present": present,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    log_share = pm.Data("log_share", np.zeros((1, MEMBERSHIP.shape[0])))
    present = pm.Data("present", np.ones((1, MEMBERSHIP.shape[0])))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    commonality = pm.LogNormal("commonality", mu=np.log(8.0), sigma=0.75)
    distinctiveness = pm.LogNormal("distinctiveness", mu=np.log(2.0), sigma=0.75)
    intercept = pm.Normal("intercept", mu=-2.0, sigma=2.0)
    slope = pm.LogNormal("slope", mu=np.log(5.0), sigma=1.0)

    log_sal = commonality * log_share - distinctiveness * pt.as_tensor(LOG_SIZE)[None, :]
    salience = present * pt.exp(log_sal)  # (sets, F)
    sim = (salience @ pt.as_tensor(MEMBERSHIP)) / pt.sum(salience, axis=1, keepdims=True)  # (sets, 100)

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * sim.reshape((-1,))[row_cell]))
    pm.Bernoulli("response", p=p_yes, observed=rating)
