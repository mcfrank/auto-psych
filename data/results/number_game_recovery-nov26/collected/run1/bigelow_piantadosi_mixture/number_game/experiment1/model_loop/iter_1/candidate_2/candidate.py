"""Single-hypothesis (MAP) concept selection.

People do not average over all concepts consistent with the examples: they
commit to the single most probable concept (one of 45 rules or one interval
of 1-100, under a fixed prior and the size principle, with 10% of examples
treated as random) and answer "yes" exactly when the target belongs to it,
apart from response errors: a hit rate for members and a false-alarm rate
for non-members.

The prior is fixed: half the mass on the rules (shared equally), half on
the 5,050 intervals (shared equally). The chosen concept depends on no free
parameter, so it is computed once per example set.
"""

from functools import lru_cache

import numpy as np
import pymc as pm

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


def _rules():
    rules = [
        [n for n in NUMBERS if n % 2 == 1],
        list(range(2, 101, 2)),
        [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        [n * n for n in range(1, 11)],
        [n**3 for n in range(1, 5)],
        [2, 4, 8, 16, 64],
        [2, 4, 8, 16, 32, 37, 64],
    ]
    rules += [list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rules += [[m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)]
    rules += [[n for n in NUMBERS if n % 10 == d] for d in range(10)]
    rules += [[n for n in NUMBERS if str(d) in str(n)] for d in range(10)]
    return rules


RULES = _rules()
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype=bool)
SIZES = MEMBERSHIP.sum(axis=1, keepdims=True)
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / SIZES + (1 - ALPHA) / 100)
LOG_PRIOR = np.concatenate([
    np.full(len(RULES), np.log(0.5 / len(RULES))),
    np.full(len(INTERVALS), np.log(0.5 / len(INTERVALS))),
])


@lru_cache(maxsize=None)
def _map_concept(examples):
    score = LOG_PRIOR + EXAMPLE_LOG_LIK[:, np.array(examples) - 1].sum(axis=1)
    return MEMBERSHIP[int(np.argmax(score))]


def compute_features(number_set, target):
    examples = tuple(int(x) for x in str(number_set).split(","))
    return {"in_map_concept": float(_map_concept(examples)[int(target) - 1])}


with pm.Model() as model:
    in_map_concept = pm.Data("in_map_concept", np.zeros(1, dtype="float64"))

    hit_rate = pm.Beta("hit_rate", 2, 2)  # P(yes | target in the chosen concept)
    false_alarm = pm.Beta("false_alarm", 2, 2)  # P(yes | target outside it)

    p_yes = pm.Deterministic(
        "p_yes", false_alarm + (hit_rate - false_alarm) * in_map_concept
    )
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
