"""Tversky feature-contrast similarity.

People do not infer a hidden rule or range; they judge a target by Tversky-style
feature similarity to the examples. Each number is represented by its salient
properties (parity, primality, squares, cubes, powers, multiples, final digit,
digits it contains, and which band of ten it falls in), each weighted by its
distinctiveness (rarer properties count more, (1/|feature|)^gamma). The examples
are summarised as a feature profile (the share of examples having each feature);
similarity is Tversky's ratio model
    S = common / (common + alpha * set_only + beta * target_only),
and the probability of "yes" is a logistic function of S.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


def _features():
    f = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "even": [n for n in NUMBERS if n % 2 == 0],
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
    }
    f.update({f"mult{m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    f.update({f"pow{m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    f.update({f"end{d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    f.update({f"has{d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(1, 10)})
    f.update({f"band{k}": list(range(10 * k + 1, 10 * k + 11)) for k in range(10)})
    f.update({f"sband{k}": list(range(10 * k + 6, 10 * k + 16)) for k in range(9)})
    return list(f.values())


FEATS = _features()
MEMB = np.array([np.isin(NUMBERS, c) for c in FEATS], dtype="float64")  # (F, 100)
N_FEAT = MEMB.shape[0]
LOG_SIZE = np.log(MEMB.sum(axis=1)) - np.log(100.0)  # <= 0


def prepare_observed(rows):
    prof, tgt, y = [], [], []
    for r in rows:
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        prof.append(MEMB[:, [e - 1 for e in ex]].mean(axis=1))
        tgt.append(MEMB[:, int(float(r["target"])) - 1])
        y.append(int(float(r["rating"])))
    return {
        "set_profile": np.array(prof, dtype="float64"),
        "target_feats": np.array(tgt, dtype="float64"),
        "rating": np.array(y, dtype="int64"),
    }


with pm.Model() as model:
    set_profile = pm.Data("set_profile", np.zeros((1, N_FEAT)))
    target_feats = pm.Data("target_feats", np.zeros((1, N_FEAT)))

    gamma = pm.HalfNormal("gamma", sigma=1.0)  # distinctiveness weighting
    alpha = pm.HalfNormal("alpha", sigma=2.0)  # weight on examples' features the target lacks
    beta = pm.HalfNormal("beta", sigma=2.0)  # weight on target's features the examples lack
    intercept = pm.Normal("intercept", mu=-2.0, sigma=2.0)
    slope = pm.HalfNormal("slope", sigma=10.0)

    w = pt.exp(-gamma * pt.as_tensor_variable(LOG_SIZE))  # (F,)
    common = pt.dot(set_profile * target_feats, w)
    set_only = pt.dot(set_profile * (1.0 - target_feats), w)
    target_only = pt.dot((1.0 - set_profile) * target_feats, w)
    sim = common / (common + alpha * set_only + beta * target_only + 1e-6)

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
