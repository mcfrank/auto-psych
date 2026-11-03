"""Feature-prototype ratio similarity.

People judge the target by its similarity to the example set's feature
prototype, on an explicit metric: the features are number properties (the 45
arithmetic and digit rules, the 18 "digits add up to s" properties, and
membership of each stretch of the number line), and the prototype holds each
feature in proportion to the share of examples that have it (raised to an
inferred sharpness). Each feature's salience falls as a power of how many
numbers share it (rare properties are distinctive), with rules as a class
weighted by an inferred share. Similarity is Tversky's ratio form: the salient
prototype mass the target shares, divided by the prototype's total salient
mass, and "yes" follows similarity, with occasional guessing. There is no
inference over hidden programs and no size principle: salience is fixed and
does not sharpen with the number of examples.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


def rule_features():
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
    digit_sum = np.array([sum(int(c) for c in str(n)) for n in NUMBERS])
    rules.update({f"digit sum {s}": list(NUMBERS[digit_sum == s]) for s in range(1, 19)})
    return list(rules.values())


RULES = rule_features()
STRETCHES = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + STRETCHES], dtype="float64")
SIZE = MEMBERSHIP.sum(axis=1).astype(int)
IS_RULE = (np.arange(len(MEMBERSHIP)) < len(RULES)).astype(int)
# Prototype shares j/n that can occur with 1-4 examples.
SHARES = sorted({j / n for n in range(1, 5) for j in range(1, n + 1)})
SHARE_ID = {round(s, 9): i for i, s in enumerate(SHARES)}
N_SHARES = len(SHARES)
# Feature groups: (size 1..100, rule or stretch, prototype share).
G_SIZE = np.repeat(np.arange(1, 101), 2 * N_SHARES).astype("float64")
G_RULE = np.tile(np.repeat([0.0, 1.0], N_SHARES), 100)
G_SHARE = np.tile(np.array(SHARES), 200)
N_GROUPS = len(G_SIZE)


def prepare_observed(rows):
    """Per (set, target) cell: counts of prototype features in each group, all and target-shared."""
    cells, cell_of, row_cell = [], {}, []
    for r in rows:
        key = (r["number_set"], int(r["target"]))
        if key not in cell_of:
            cell_of[key] = len(cells)
            cells.append(key)
        row_cell.append(cell_of[key])
    proto_counts = np.zeros((len(cells), N_GROUPS))
    shared_counts = np.zeros((len(cells), N_GROUPS))
    for i, (s, t) in enumerate(cells):
        examples = np.array([int(x) for x in s.split(",")])
        share = MEMBERSHIP[:, examples - 1].mean(axis=1)
        has = share > 0
        sid = np.array([SHARE_ID[round(v, 9)] for v in share[has]])
        group = ((SIZE[has] - 1) * 2 + IS_RULE[has]) * N_SHARES + sid
        np.add.at(proto_counts[i], group, 1.0)
        np.add.at(shared_counts[i], group, MEMBERSHIP[has, t - 1])
    return {
        "proto_counts": proto_counts,
        "shared_counts": shared_counts,
        "row_cell": np.array(row_cell, dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    proto_counts = pm.Data("proto_counts", np.zeros((1, N_GROUPS)))
    shared_counts = pm.Data("shared_counts", np.zeros((1, N_GROUPS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    gamma = pm.Normal("gamma", 2.0, 2.0)  # distinctiveness: salience ~ size^-gamma
    log_rule = pm.Normal("log_rule", 0.0, 3.0)  # salience of rules relative to stretches
    log_kappa = pm.Normal("log_kappa", 0.0, 1.0)  # sharpness of the graded prototype
    lapse = pm.Beta("lapse", 2, 5)
    guess_yes = pm.Beta("guess_yes", 2, 2)

    log_w = (
        -gamma * np.log(G_SIZE)
        + log_rule * G_RULE
        + pt.exp(log_kappa) * np.log(G_SHARE)
    )
    w = pt.exp(log_w - pt.max(log_w))
    similarity = pt.dot(shared_counts, w) / pt.dot(proto_counts, w)
    p = (1 - lapse) * similarity[row_cell] + lapse * guess_yes
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
