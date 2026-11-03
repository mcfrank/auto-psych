"""Nearest-exemplar concept resemblance.

People store each shown example and judge the target by its best match to any
single stored example (a max/nearest-exemplar rule, not summed or conjunctive
similarity). The resemblance of target t to example x is Shepard-style
consequential-region generalization from that one example: the probability that
a concept containing x also contains t, over the arithmetic/digit/digit-sum
rules (share lam) and all intervals of 1..100, each concept weighted by
size^-gamma. p(yes) runs from a floor to a ceiling with the closest match.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


def _rules():
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
    return np.array([np.isin(NUMBERS, c) for c in rules], dtype="float64")


RULE_MEMBERS = _rules()  # (63, 100)
RULE_SIZE = RULE_MEMBERS.sum(axis=1)
WIDTHS = np.arange(1, 101, dtype="float64")


def _interval_counts(lo, hi):
    """Number of intervals of each width 1..100 inside 1..100 that cover [lo, hi]."""
    w = WIDTHS
    start_min = np.maximum(1, hi - w + 1)
    start_max = np.minimum(lo, 101 - w)
    return np.maximum(0.0, start_max - start_min + 1)


def prepare_observed(rows):
    pairs, pair_index, slots = {}, [], []
    for r in rows:
        ex = [int(v) for v in str(r["number_set"]).split(",")]
        t = int(r["target"])
        idx = []
        for x in ex:
            idx.append(pairs.setdefault((x, t), len(pairs)))
        idx += [idx[0]] * (4 - len(idx))  # padding repeats an exemplar: max unchanged
        slots.append(idx)
    keys = list(pairs)
    shared_rules = np.array([RULE_MEMBERS[:, x - 1] * RULE_MEMBERS[:, t - 1] for x, t in keys])
    own_rules = np.array([RULE_MEMBERS[:, x - 1] for x, _ in keys])
    shared_int = np.array([_interval_counts(min(x, t), max(x, t)) for x, t in keys])
    own_int = np.array([_interval_counts(x, x) for x, _ in keys])
    return {
        "shared_rules": shared_rules,
        "own_rules": own_rules,
        "shared_int": shared_int,
        "own_int": own_int,
        "slot_pair": np.array(slots, dtype="int64"),
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    shared_rules = pm.Data("shared_rules", np.zeros((1, 63)))
    own_rules = pm.Data("own_rules", np.ones((1, 63)))
    shared_int = pm.Data("shared_int", np.zeros((1, 100)))
    own_int = pm.Data("own_int", np.ones((1, 100)))
    slot_pair = pm.Data("slot_pair", np.zeros((1, 4), dtype="int64"))

    lam = pm.Beta("lam", 2, 2)  # prior share of rules vs stretches of the line
    gamma = pm.LogNormal("gamma", 0.0, 0.5)  # how much narrow shared concepts count more
    ceiling = pm.Beta("ceiling", 4, 2)  # p(yes) for a perfect match to an exemplar
    floor = pm.Beta("floor", 1, 20)  # p(yes) for no resemblance at all

    w_rule = lam / 63.0 * pt.as_tensor_variable(RULE_SIZE) ** (-gamma)
    w_int = (1 - lam) / 5050.0 * pt.as_tensor_variable(WIDTHS) ** (-gamma)
    num = pt.dot(shared_rules, w_rule) + pt.dot(shared_int, w_int)
    den = pt.dot(own_rules, w_rule) + pt.dot(own_int, w_int)
    resemblance = num / den  # per (example, target) pair, in (0, 1]
    closest = pt.max(resemblance[slot_pair], axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(floor + (ceiling - floor) * closest, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
