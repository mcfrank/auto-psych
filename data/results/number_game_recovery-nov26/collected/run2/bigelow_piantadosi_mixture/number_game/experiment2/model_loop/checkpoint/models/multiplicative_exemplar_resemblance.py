"""Multiplicative exemplar resemblance: people store each example and compare the
target to every stored example. The resemblance of the target to one example is
how likely the two numbers are to share a number concept (one of 45 arithmetic
and digit rules, or a numeric range of 1..100) judged from that example alone,
so sharing a narrow property counts more than sharing a broad one. Resemblances
combine multiplicatively across the stored examples, as a geometric mean (the
target must resemble all of them; one poor match drags the whole judgment down),
and P(yes) grows as a power of that overall resemblance, with a guessing lapse.
A target resembling only some examples is strongly rejected.
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


RULES = np.array([np.isin(NUMBERS, c) for c in rule_concepts()], dtype="float64")
INTERVALS = np.array(
    [(NUMBERS >= m) & (NUMBERS <= n) for m in NUMBERS for n in range(m, 101)], dtype="float64"
)


def pair_terms(members):
    """Size-weighted co-membership (x, y) and membership (x) sums, averaged over the family."""
    w = members / members.sum(axis=1, keepdims=True)  # 1/|h| for members
    joint = w.T @ members / len(members)  # [x, y]: mean over h of [x in h][y in h] / |h|
    single = w.sum(axis=0) / len(members)  # [x]: mean over h of [x in h] / |h|
    return joint, single


RULE_JOINT, RULE_SINGLE = pair_terms(RULES)
INT_JOINT, INT_SINGLE = pair_terms(INTERVALS)
MAX_EX = 4

with pm.Model() as model:
    # Per trial, per stored example slot: numerator/denominator pieces of the
    # single-example resemblance, split by rules and ranges; mask marks real slots.
    rule_joint = pm.Data("rule_joint", np.zeros((1, MAX_EX)))
    rule_single = pm.Data("rule_single", np.ones((1, MAX_EX)))
    int_joint = pm.Data("int_joint", np.zeros((1, MAX_EX)))
    int_single = pm.Data("int_single", np.ones((1, MAX_EX)))
    slot_mask = pm.Data("slot_mask", np.ones((1, MAX_EX)))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)  # weight of rules vs. ranges in resemblance
    sharpness = pm.LogNormal("sharpness", 0.0, 0.75)  # exponent on overall resemblance
    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)

    num = lambda_0 * rule_joint + (1 - lambda_0) * int_joint
    den = lambda_0 * rule_single + (1 - lambda_0) * int_single
    log_sim = pt.log(pt.clip(num / den, 1e-12, 1.0))
    log_resemblance = (log_sim * slot_mask).sum(axis=1) / slot_mask.sum(axis=1)
    p_match = pt.exp(sharpness * log_resemblance)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip((1 - lapse) * p_match + lapse * guess_yes, 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    n = len(rows)
    out = {k: np.zeros((n, MAX_EX)) for k in ["rule_joint", "int_joint", "slot_mask"]}
    out["rule_single"] = np.ones((n, MAX_EX))
    out["int_single"] = np.ones((n, MAX_EX))
    for i, r in enumerate(rows):
        examples = [int(x) for x in str(r["number_set"]).split(",")]
        t = int(r["target"]) - 1
        for j, x in enumerate(examples[:MAX_EX]):
            out["rule_joint"][i, j] = RULE_JOINT[x - 1, t]
            out["rule_single"][i, j] = RULE_SINGLE[x - 1]
            out["int_joint"][i, j] = INT_JOINT[x - 1, t]
            out["int_single"][i, j] = INT_SINGLE[x - 1]
            out["slot_mask"][i, j] = 1.0
    out["rating"] = np.array([int(r["rating"]) for r in rows], dtype="int64")
    return out
