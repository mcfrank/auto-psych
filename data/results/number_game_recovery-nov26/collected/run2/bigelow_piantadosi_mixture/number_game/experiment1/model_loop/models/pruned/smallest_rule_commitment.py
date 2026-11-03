"""Smallest-rule commitment (a deterministic MAP-style rule learner).

People do not weigh many candidate concepts at once; they search for the
single most specific rule that covers every example (the smallest of the
arithmetic and digit rules such as "powers of 2", "multiples of 7", "ends in
3" or "contains an 8"), commit to it, and answer yes to a target if and only if
it obeys that rule, falling back to the range spanned by the examples only when
no rule fits them all. Apart from response noise (a rate of yes inside the
chosen concept and a rate of yes outside it), generalization is all-or-none:
no graded similarity and no averaging over alternative concepts. When several
rules tie for smallest, the learner picks one of them at random.
"""

import numpy as np
import pymc as pm

NUMBERS = np.arange(1, 101)


def is_prime(n):
    return n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))


def rule_concepts():
    rules = [
        [n for n in NUMBERS if n % 2 == 1],
        [n for n in NUMBERS if n % 2 == 0],
        [n * n for n in range(1, 11)],
        [n**3 for n in range(1, 5)],
        [n for n in NUMBERS if is_prime(n)],
    ]
    rules += [list(range(m, 101, m)) for m in range(3, 13)]
    rules += [[m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)]
    rules += [[n for n in NUMBERS if n % 10 == d] for d in range(10)]
    rules += [[n for n in NUMBERS if str(d) in str(n)] for d in range(10)]
    return rules


RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, r) for r in rule_concepts()], dtype="float64")
RULE_SIZES = RULE_MEMBERSHIP.sum(axis=1)


def compute_features(number_set, target):
    """Probability that the target lies in the committed (smallest consistent) concept."""
    examples = np.array([int(x) for x in str(number_set).split(",")])
    t = int(target)
    consistent = RULE_MEMBERSHIP[:, examples - 1].all(axis=1)
    if consistent.any():
        smallest = consistent & (RULE_SIZES == RULE_SIZES[consistent].min())
        in_rule = float(RULE_MEMBERSHIP[smallest, t - 1].mean())
    else:
        in_rule = float(examples.min() <= t <= examples.max())
    return {"in_committed_rule": in_rule}


with pm.Model() as model:
    in_committed_rule = pm.Data("in_committed_rule", np.zeros(1, dtype="float64"))

    # Rate of "yes" for targets that obey the committed rule, and for those that don't.
    p_yes_in = pm.Beta("p_yes_in", 2.0, 2.0)
    p_yes_out = pm.Beta("p_yes_out", 1.0, 3.0)

    p_yes = pm.Deterministic(
        "p_yes", in_committed_rule * p_yes_in + (1.0 - in_committed_rule) * p_yes_out
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
