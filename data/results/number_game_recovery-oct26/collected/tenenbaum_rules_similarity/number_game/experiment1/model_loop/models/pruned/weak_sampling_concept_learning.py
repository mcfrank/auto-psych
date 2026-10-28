"""Weak-sampling Bayesian concept learning (no size principle).

People consider the same 42 rule concepts and all intervals of 1..100, but
treat the examples only as consistent with the concept rather than sampled
from it: each example the concept contains has likelihood 1, each it does not
contain has likelihood eps (a noise rate, inferred), whatever the concept's
size. "Yes" is the posterior probability that the concept contains the
target. Unlike the strong-sampling models, adding examples does not tighten
generalization to the smallest consistent concept.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4


def concept_sets():
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
    return rules


RULES = concept_sets()
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
LO, HI = np.meshgrid(NUMBERS, NUMBERS, indexing="ij")
VALID = LO <= HI  # interval [lo, hi]
N_INTERVALS = int(VALID.sum())


def prepare_observed(rows):
    """Per distinct set: how many examples each rule misses, and for intervals
    the count of intervals missing m examples (m = 0..4), overall and among
    those containing each target. Weak-sampling likelihood is eps**m."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    rule_miss = np.zeros((len(unique), N_RULES, MAX_N + 1))
    int_count = np.zeros((len(unique), MAX_N + 1))
    int_target_count = np.zeros((len(unique), 100, MAX_N + 1))
    for i, s in enumerate(unique):
        ex = np.array(s)
        inside = RULE_MEMBERSHIP[:, ex - 1].sum(axis=1).astype(int)
        rule_miss[i, np.arange(N_RULES), len(ex) - inside] = 1.0
        contained = ((LO[..., None] <= ex) & (ex <= HI[..., None])).sum(axis=-1)
        miss = len(ex) - contained
        for m in range(MAX_N + 1):
            mask = VALID & (miss == m)
            int_count[i, m] = mask.sum()
            # intervals with lo <= t <= hi among mask
            for t in NUMBERS:
                int_target_count[i, t - 1, m] = mask[:t, t - 1:].sum()
    return {
        "rule_miss": rule_miss,
        "interval_miss_count": int_count,
        "interval_target_miss_count": int_target_count,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    rule_miss = pm.Data("rule_miss", np.zeros((1, N_RULES, MAX_N + 1)))
    interval_miss_count = pm.Data("interval_miss_count", np.zeros((1, MAX_N + 1)))
    interval_target_miss_count = pm.Data("interval_target_miss_count", np.zeros((1, 100, MAX_N + 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    eps = pm.Beta("eps", 1, 9)  # likelihood of an example the concept does not contain

    eps_pow = eps ** pt.arange(MAX_N + 1).astype("float64")  # eps**m
    rule_lik = pt.tensordot(rule_miss, eps_pow, axes=[[2], [0]])  # (sets, rules)
    rule_joint = p_mix * rule_weights[None, :] * rule_lik
    int_scale = (1 - p_mix) / N_INTERVALS
    int_joint = int_scale * pt.dot(interval_miss_count, eps_pow)  # (sets,)
    int_target_joint = int_scale * pt.tensordot(interval_target_miss_count, eps_pow, axes=[[2], [0]])  # (sets, 100)
    evidence = rule_joint.sum(axis=1) + int_joint
    in_concept = (pt.dot(rule_joint, pt.as_tensor(RULE_MEMBERSHIP)) + int_target_joint) / evidence[:, None]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
