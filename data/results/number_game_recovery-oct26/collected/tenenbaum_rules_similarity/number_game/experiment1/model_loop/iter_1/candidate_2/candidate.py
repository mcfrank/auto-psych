"""Minimal consistent concept (MAP-only, smallest-hypothesis) generalization.

People do not average over many candidate concepts; they commit to the single
smallest concept that contains every example -- the tightest arithmetic rule
(powers, multiples, final digit, contained digit, primes, even, odd, squares,
cubes) or, if no rule is tighter, the exact range spanned by the examples --
and say "yes" to a target essentially only if it belongs to that one concept.
Apart from a constant rate of lapses in either direction, how far an excluded
target lies from the examples does not matter.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = list(range(1, 101))


def _rules():
    rules = [
        [2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36],
        [7, 49], [8, 64], [9, 81], [10, 100],
    ]
    rules += [[i for i in NUMBERS if i % n == 0] for n in range(3, 13)]
    rules += [[i for i in NUMBERS if i % 10 == d] for d in range(1, 10)]
    rules += [[i for i in NUMBERS if str(d) in str(i)] for d in range(1, 10)]
    rules.append([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    rules.append([i for i in NUMBERS if i % 2 == 0])
    rules.append([i for i in NUMBERS if i % 2 == 1])
    rules.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    rules.append([1, 8, 27, 64])
    return [set(r) for r in rules]


RULES = _rules()


def compute_features(number_set, target):
    """1 if the target is in the smallest concept containing all examples."""
    examples = [int(x) for x in number_set.split(",")]
    t = int(target)
    span = set(range(min(examples), max(examples) + 1))
    best = span
    for r in RULES:
        if all(x in r for x in examples) and len(r) <= len(best):
            if len(r) < len(best) or best is span:
                best = r
    return {"in_minimal_concept": 1.0 if t in best else 0.0}


with pm.Model() as model:
    in_minimal_concept = pm.Data("in_minimal_concept", np.zeros(1, dtype="float64"))

    # Probability of "yes" for targets inside / outside the adopted concept.
    p_in = pm.Beta("p_in", 4.0, 1.5)
    p_out = pm.Beta("p_out", 1.5, 4.0)

    p = p_out + (p_in - p_out) * in_minimal_concept
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
