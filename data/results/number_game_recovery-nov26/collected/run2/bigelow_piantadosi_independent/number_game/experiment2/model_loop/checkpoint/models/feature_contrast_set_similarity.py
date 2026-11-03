"""Feature-contrast similarity to what the examples have in common.

People do not infer a hidden concept; they judge the target by its similarity
to the example set on Tversky's ratio contrast metric. The examples' common
features are the familiar number properties (parity, primes, squares/cubes,
powers, multiples, final digit, contained digit) that every example has, each
counting by its distinctiveness (the rarer, the more salient). Featural
similarity is the salience of the common features the target shares, relative
to that plus the salience of the common features it lacks. Magnitude is one more dimension of the
same similarity: closeness to the nearest example, falling off exponentially
(a Shepard gradient). "Yes" is a logistic function of the total similarity.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


def _features():
    primes = [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))]
    feats = [
        [n for n in NUMBERS if n % 2 == 1],
        list(range(2, 101, 2)),
        primes,
        [n * n for n in range(1, 11)],
        [n**3 for n in range(1, 5)],
    ]
    feats += [list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    feats += [[m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)]
    feats += [[n for n in NUMBERS if n % 10 == d] for d in range(10)]
    feats += [[n for n in NUMBERS if str(d) in str(n)] for d in range(10)]
    member = np.array([np.isin(NUMBERS, f) for f in feats], dtype=bool)
    return member, -np.log(member.sum(axis=1) / 100.0)


MEMBER, SALIENCE = _features()  # (features, 100), (features,)


def compute_features(number_set, target):
    examples = np.array([int(x) for x in number_set.split(",")])
    t = int(target)
    shared = MEMBER[:, examples - 1].all(axis=1)
    has = MEMBER[:, t - 1]
    return {
        "common_shared": float(SALIENCE[shared & has].sum()),
        "common_missing": float(SALIENCE[shared & ~has].sum()),
        "nearest_distance": float(np.abs(examples - t).min()),
    }


with pm.Model() as model:
    common_shared = pm.Data("common_shared", np.zeros(1))
    common_missing = pm.Data("common_missing", np.zeros(1))
    nearest_distance = pm.Data("nearest_distance", np.zeros(1))

    bias = pm.Normal("bias", mu=0.0, sigma=2.0)
    theta = pm.HalfNormal("theta", sigma=5.0)  # weight of featural similarity
    alpha = pm.LogNormal("alpha", mu=0.0, sigma=1.0)  # weight of common features the target lacks
    kappa = pm.HalfNormal("kappa", sigma=5.0)  # weight of magnitude similarity
    width = pm.LogNormal("width", mu=1.0, sigma=1.0)  # Shepard gradient width on the number line

    featural = common_shared / (common_shared + alpha * common_missing + 1e-3)
    magnitude = pt.exp(-nearest_distance / width)
    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(bias + theta * featural + kappa * magnitude), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
