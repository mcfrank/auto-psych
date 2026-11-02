"""Single-concept commitment (MAP hypothesis selection).

People do not average over all concepts consistent with the examples; they
commit to the single concept that best explains them -- the most probable rule
or magnitude interval under a size-principle Bayesian evaluation with a flat
prior (rules as a group as likely as intervals as a group, each rule and each
interval equally likely within its group) -- and answer "yes" exactly when the
target belongs to that one concept, apart from occasional response errors
(a false-alarm rate for targets outside it and a miss rate for targets inside).
Exactly tied best concepts (e.g. "ends in 0" and "contains digit 0") are
chosen between at random. The chosen concept depends on no free parameter, so
it is computed once per example set in prepare_observed.
"""

import numpy as np
import pymc as pm

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


def rule_concepts():
    rules = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
        "powers of 2, not 32": [2, 4, 8, 16, 64],
        "powers of 2, or 37": [2, 4, 8, 16, 32, 37, 64],
        "even": list(range(2, 101, 2)),
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(10)})
    return list(rules.values())


RULES = rule_concepts()
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
LOG_PRIOR = np.concatenate([
    np.full(len(RULES), np.log(0.5 / len(RULES))),
    np.full(len(INTERVALS), np.log(0.5 / len(INTERVALS))),
])


def map_membership(examples):
    """Membership (over 1..100) of the posterior-mode concept, ties averaged."""
    score = LOG_PRIOR + EXAMPLE_LOG_LIK[:, np.array(examples) - 1].sum(axis=1)
    best = np.flatnonzero(score >= score.max() - 1e-9)
    return MEMBERSHIP[best].mean(axis=0)


with pm.Model() as model:
    in_map_concept = pm.Data("in_map_concept", np.zeros(1, dtype="float64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    false_alarm = pm.Beta("false_alarm", 1.5, 3.0)  # yes to a target outside the chosen concept
    miss = pm.Beta("miss", 1.5, 3.0)  # no to a target inside it

    p_yes = pm.Deterministic(
        "p_yes", false_alarm + (1.0 - false_alarm - miss) * in_map_concept
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    cache = {s: map_membership(s) for s in dict.fromkeys(sets)}
    return {
        "in_map_concept": np.array([cache[s][int(r["target"]) - 1] for s, r in zip(sets, rows)]),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
