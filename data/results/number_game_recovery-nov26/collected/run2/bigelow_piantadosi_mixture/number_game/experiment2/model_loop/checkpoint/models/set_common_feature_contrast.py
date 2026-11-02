"""Set-level feature-contrast similarity (Tversky-style): the target is judged by
how well it matches the properties shared by every example. Each property that
all examples share (from the arithmetic and digit rules) counts toward
similarity if the target has it and against it if the target lacks it, weighted
by its squared distinctiveness log(100/size)^2, scaled to a maximum of 1 (rare
shared properties dominate). Magnitude similarity falls off linearly in log-odds with
the target's distance outside the examples' span. A lapse adds guessing.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

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


MEMBERSHIP = np.array([np.isin(NUMBERS, r) for r in rule_concepts()], dtype="float64")
INFO = np.log(100.0 / MEMBERSHIP.sum(axis=1)) ** 2
WEIGHT = INFO / INFO.max()  # squared distinctiveness of each property, max 1


def prepare_observed(rows):
    """Per trial: which set-shared properties the target keeps / lacks, and its distance outside the span."""
    kept, lacked, dist = [], [], []
    for r in rows:
        ex = np.array([int(x) for x in str(r["number_set"]).split(",")])
        t = int(r["target"])
        shared = MEMBERSHIP[:, ex - 1].min(axis=1)
        has = MEMBERSHIP[:, t - 1]
        kept.append(float(np.dot(shared * has, WEIGHT)))
        lacked.append(float(np.dot(shared * (1 - has), WEIGHT)))
        dist.append(max(ex.min() - t, t - ex.max(), 0) / 10.0)
    return {
        "shared_kept": np.array(kept, dtype="float64"),
        "shared_lacked": np.array(lacked, dtype="float64"),
        "outside_dist": np.array(dist, dtype="float64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    shared_kept = pm.Data("shared_kept", np.zeros(1))
    shared_lacked = pm.Data("shared_lacked", np.zeros(1))
    outside_dist = pm.Data("outside_dist", np.zeros(1))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    intercept = pm.Normal("intercept", 0.0, 2.0)
    w_common = pm.HalfNormal("w_common", 5.0)  # gain per unit of kept distinctive property
    w_distinct = pm.HalfNormal("w_distinct", 5.0)  # loss per unit of lacked distinctive property
    w_mag = pm.HalfNormal("w_mag", 2.0)  # magnitude gradient outside the examples' span
    lapse = pm.Beta("lapse", 2, 5)

    score = (
        intercept
        + w_common * shared_kept
        - w_distinct * shared_lacked
        - w_mag * outside_dist
    )
    p_yes = pm.Deterministic(
        "p_yes", pt.clip((1 - lapse) * pm.math.sigmoid(score) + lapse * 0.5, 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)
