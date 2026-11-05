"""Most-specific-rule, winner-take-all hypothesis.

People settle on a single concept: the smallest mathematical rule that every
example satisfies, and answer "yes" essentially only to targets that obey that
rule (with noise). If no rule fits all examples they fall back on the range
spanned by the examples. No graded averaging over hypotheses.
"""
import numpy as np
import pymc as pm


def _rules():
    N = range(1, 101)
    R = [
        {n for n in N if n % 2 == 1},
        {n for n in N if n % 2 == 0},
        {n for n in N if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))},
        {k * k for k in range(1, 11)},
        {k ** 3 for k in range(1, 5)},
    ]
    for b in range(2, 11):
        R.append({b ** k for k in range(1, 8) if b ** k <= 100})
    for m in range(3, 13):
        R.append({n for n in N if n % m == 0})
    for d in range(10):
        R.append({n for n in N if n % 10 == d})
    for d in range(10):
        R.append({n for n in N if str(d) in str(n)})
    return R


_RULES = _rules()


def _concept_membership(examples, target):
    consistent = [r for r in _RULES if all(x in r for x in examples)]
    if not consistent:
        return 1.0 if min(examples) <= target <= max(examples) else 0.0
    smallest = min(len(r) for r in consistent)
    best = [r for r in consistent if len(r) == smallest]
    return float(np.mean([target in r for r in best]))


def compute_features(number_set, target):
    examples = [int(x) for x in str(number_set).split(",")]
    return {"in_best_rule": _concept_membership(examples, int(target))}


with pm.Model() as model:
    in_best_rule = pm.Data("in_best_rule", np.zeros(1, dtype="float64"))

    # probability of "yes" for a target outside / inside the adopted concept
    p_out = pm.Beta("p_out", alpha=1.0, beta=3.0)
    p_in = pm.Beta("p_in", alpha=3.0, beta=1.0)

    p = p_out + (p_in - p_out) * in_best_rule
    p_yes = pm.Deterministic("p_yes", pm.math.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
