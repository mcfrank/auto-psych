"""Tightest shared description similarity.

People judge the target by its similarity to the examples, measured by the
tightest description they share: the most specific concept containing all the
examples (one of 45 arithmetic/digit rules, 18 digit-sum concepts, or the
narrowest interval of 1..100 covering them), and how much broader that shared
description must become to include the target too. The metric is the
broadening d = log(|tightest concept with target| / |tightest concept without|)
and similarity falls off as exp(-lam * d^shape); p(yes) runs from a floor to a
ceiling with similarity. No weighing of alternative hypotheses.
"""

import functools

import numpy as np
import pymc as pm

NUMBERS = np.arange(1, 101)


def _concepts():
    rules = [
        [n for n in NUMBERS if n % 2 == 1],
        [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        [n * n for n in range(1, 11)],
        [n**3 for n in range(1, 5)],
        [2, 4, 8, 16, 64],
        [2, 4, 8, 16, 32, 37, 64],
        list(range(2, 101, 2)),
    ]
    rules += [list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rules += [[m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)]
    rules += [[n for n in NUMBERS if n % 10 == d] for d in range(10)]
    rules += [[n for n in NUMBERS if str(d) in str(n)] for d in range(10)]
    dsum = np.array([sum(int(c) for c in str(n)) for n in NUMBERS])
    rules += [list(NUMBERS[dsum == s]) for s in range(1, 19)]
    return np.array([np.isin(NUMBERS, c) for c in rules], dtype=bool)


RULE_MEMBERS = _concepts()
RULE_SIZE = RULE_MEMBERS.sum(axis=1).astype("float64")


@functools.lru_cache(maxsize=None)
def _tightest(members):
    """Size of the most specific rule or interval containing every number in `members`."""
    idx = np.array(members) - 1
    fits = RULE_MEMBERS[:, idx].all(axis=1)
    rule_size = RULE_SIZE[fits].min() if fits.any() else 100.0
    return min(rule_size, float(max(members) - min(members) + 1))


def compute_features(number_set, target):
    examples = tuple(sorted({int(x) for x in number_set.split(",")}))
    with_target = tuple(sorted(set(examples) | {int(target)}))
    return {"broadening": float(np.log(_tightest(with_target) / _tightest(examples)))}


with pm.Model() as model:
    broadening = pm.Data("broadening", np.zeros(1, dtype="float64"))

    lam = pm.LogNormal("lam", 0.0, 1.0)  # steepness of the similarity gradient
    shape = pm.LogNormal("shape", -0.5, 0.75)  # curvature: <1 means steep then flat
    ceiling = pm.Beta("ceiling", 4, 2)  # p(yes) for a target losing no specificity
    floor = pm.Beta("floor", 1, 20)  # p(yes) for a maximally dissimilar target

    similarity = pm.math.exp(-lam * (broadening + 1e-6) ** shape)
    p_yes = pm.Deterministic("p_yes", floor + (ceiling - floor) * similarity)

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
