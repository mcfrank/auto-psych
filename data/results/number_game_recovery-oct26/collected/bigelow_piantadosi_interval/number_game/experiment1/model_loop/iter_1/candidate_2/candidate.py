"""Weak-sampling hypothesis elimination: people entertain mathematical rules
and numeric intervals as candidate concepts, but treat the examples only as
positive labels (not draws from the concept), so every concept consistent with
the examples keeps its prior weight regardless of its size — no size
principle. "Yes" is the prior-weighted share of consistent concepts that
contain the target; each example inconsistent with a concept is tolerated with
probability `eps` (an error), multiplying that concept's likelihood by eps.

Rules share prior p_mix equally; the 5,050 intervals share 1 - p_mix equally.
Since a concept's likelihood is eps ** (number of examples it excludes),
hypotheses are grouped by that count (0..4) per example set, precomputed once.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_K = 5  # number of excluded examples: 0..4


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
    intervals = [list(range(m, n + 1)) for m in range(1, 101) for n in range(m, 101)]
    return rules, intervals


RULES, INTERVALS = concept_sets()
N_RULES, N_INTERVALS = len(RULES), len(INTERVALS)
RULE_MEM = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
INT_MEM = np.array([np.isin(NUMBERS, c) for c in INTERVALS], dtype="float64")


def _grouped(mem, examples):
    """Per excluded-example count k: number of concepts, and number containing each target."""
    excluded = (1.0 - mem[:, np.array(examples) - 1]).sum(axis=1).astype(int)
    onehot = np.zeros((mem.shape[0], MAX_K))
    onehot[np.arange(mem.shape[0]), excluded] = 1.0
    return onehot.sum(axis=0), onehot.T @ mem  # (K,), (K, 100)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    rc, rt, ic, it = [], [], [], []
    for s in unique:
        a, b = _grouped(RULE_MEM, s)
        c, d = _grouped(INT_MEM, s)
        rc.append(a); rt.append(b); ic.append(c); it.append(d)
    return {
        "rule_count": np.stack(rc), "rule_target": np.stack(rt),
        "int_count": np.stack(ic), "int_target": np.stack(it),
        "row_set": np.array([index[s] for s in sets], dtype="int64"),
        "row_target": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    rule_count = pm.Data("rule_count", np.zeros((1, MAX_K)))
    rule_target = pm.Data("rule_target", np.zeros((1, MAX_K, 100)))
    int_count = pm.Data("int_count", np.zeros((1, MAX_K)))
    int_target = pm.Data("int_target", np.zeros((1, MAX_K, 100)))
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    eps = pm.Beta("eps", 1, 9)

    lik = eps ** pt.arange(MAX_K)  # likelihood of a concept excluding k examples
    w_rule = p_mix / N_RULES * lik
    w_int = (1 - p_mix) / N_INTERVALS * lik
    evidence = rule_count @ w_rule + int_count @ w_int  # (sets,)
    in_target = (rule_target * w_rule[None, :, None]).sum(axis=1) + (int_target * w_int[None, :, None]).sum(axis=1)
    p_in = in_target / evidence[:, None]  # (sets, 100)

    p_yes = pm.Deterministic("p_yes", pt.clip(p_in[row_set, row_target], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
