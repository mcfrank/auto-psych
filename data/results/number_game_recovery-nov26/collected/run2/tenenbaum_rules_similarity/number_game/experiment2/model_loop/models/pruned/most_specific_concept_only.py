"""People adopt the single most specific concept that covers all the examples --
the smallest of the arithmetic/digit rules or the tightest range spanning the
examples -- and answer "yes" when the target belongs to it and "no" otherwise,
apart from constant response noise (Tenenbaum's MIN strategy). There is no
graded averaging over concepts: a single example generalizes to almost nothing,
and "yes" stops sharply at the edge of the chosen concept.
"""

import numpy as np
import pymc as pm

NUMBERS = list(range(1, 101))


def _is_prime(n):
    return n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1))


def _rules():
    rules = [
        {n for n in NUMBERS if n % 2 == 1},
        {n for n in NUMBERS if n % 2 == 0},
        {n for n in NUMBERS if _is_prime(n)},
        {n * n for n in range(1, 11)},
        {n ** 3 for n in range(1, 5)},
        {2, 4, 8, 16, 64},
        {2, 4, 8, 16, 32, 37, 64},
    ]
    rules += [set(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rules += [{m ** k for k in range(1, 8) if m ** k <= 100} for m in range(2, 11)]
    rules += [{n for n in NUMBERS if n % 10 == d} for d in range(10)]
    rules += [{n for n in NUMBERS if str(d) in str(n)} for d in range(10)]
    return rules


RULES = _rules()


def compute_features(number_set, target):
    """Share of the smallest concepts covering every example that contain the target."""
    examples = [int(x) for x in str(number_set).split(",")]
    t = int(target)
    concepts = [r for r in RULES if all(x in r for x in examples)]
    concepts.append(set(range(min(examples), max(examples) + 1)))
    smallest = min(len(c) for c in concepts)
    chosen = {frozenset(c) for c in concepts if len(c) == smallest}
    return {"in_min_concept": float(np.mean([t in c for c in chosen]))}


with pm.Model() as model:
    in_min_concept = pm.Data("in_min_concept", np.zeros(1, dtype="float64"))

    # Probability of "yes" to a target inside / outside the chosen concept.
    p_in = pm.Beta("p_in", 4.0, 2.0)
    p_out = pm.Beta("p_out", 2.0, 4.0)

    p_yes = pm.Deterministic("p_yes", p_out + (p_in - p_out) * in_min_concept)

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
