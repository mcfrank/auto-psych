"""Refinement of bigelow_piantadosi_independent with an inferred outlier rate.

People do Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11
and 12, ending in 0-9, containing the digit 0-9) and every interval [m, n] of
1..100, with a prior weight per rule, share lambda_0 for the rules and
(1 - lambda_0) / 100^2 per interval. Each example comes from the concept
(uniformly: the size principle) with probability alpha and from 1..100
otherwise. The one change from the incumbent: alpha, the trust that an example
is drawn from the concept rather than being noise, is inferred instead of fixed
at 0.9. "Yes" is the posterior probability that the concept contains the target.

Because alpha is a parameter, likelihoods are computed in the model from
precomputed counts: for each set and hypothesis, how many examples it contains.
Intervals sharing a size and a count of contained examples share a likelihood
and are pooled per set.
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
    return rules


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
# Intervals [m, n], m <= n. Given alpha, an interval's likelihood depends only
# on its size L and on how many examples h it contains, so for each set the
# intervals are pooled into (L, h) groups: their count and, per target, the
# share of the group's intervals containing it. The posterior is exactly the one
# over all 5,050 intervals.
M_ARR = np.array([m for m in NUMBERS for n in range(m, 101)])
N_ARR = np.array([n for m in NUMBERS for n in range(m, 101)])
L_ARR = N_ARR - M_ARR + 1
MAX_H = 5
GROUP_SIZE = np.repeat(np.arange(1, 101), MAX_H).astype("float64")  # group index (L - 1) * MAX_H + h
GROUP_HITS = np.tile(np.arange(MAX_H), 100).astype("float64")
INTERVAL_MEMBERSHIP = ((M_ARR[:, None] <= NUMBERS[None, :]) & (NUMBERS[None, :] <= N_ARR[:, None])).astype("float64")

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_hits = pm.Data("rule_hits", np.zeros((1, N_RULES)))  # (sets, rules): examples in rule
    group_log_count = pm.Data("group_log_count", np.zeros((1, 100 * MAX_H)))  # (sets, groups): log #intervals
    # Distinct (set, target) cells: the set each is asked of, and the share of each
    # rule and interval group containing its target.
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_in_concept = pm.Data("cell_in_concept", np.zeros((1, N_RULES + 100 * MAX_H)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # each trial's cell
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    alpha = pm.Beta("alpha", 6, 1.5)  # trust that an example comes from the concept
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    log_noise = pt.log1p(-alpha) - np.log(100)
    k = n_examples[:, None]
    rule_ll = rule_hits * pt.log(alpha / RULE_SIZE[None, :] + (1 - alpha) / 100) + (k - rule_hits) * log_noise
    group_ll = GROUP_HITS[None, :] * pt.log(alpha / GROUP_SIZE[None, :] + (1 - alpha) / 100) + (k - GROUP_HITS[None, :]) * log_noise

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_ll
    log_groups = pt.log1p(-lambda_0) - 2 * np.log(100) + group_log_count + group_ll
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[cell_set] * cell_in_concept).sum(axis=1)[row_cell]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    rule_hits, log_count, in_concept = [], [], []
    for s in unique:
        x = np.array(s)
        rule_hits.append(RULE_MEMBERSHIP[:, x - 1].sum(axis=1))
        hits = ((M_ARR[:, None] <= x[None, :]) & (x[None, :] <= N_ARR[:, None])).sum(axis=1)
        group = (L_ARR - 1) * MAX_H + hits
        count = np.bincount(group, minlength=100 * MAX_H).astype("float64")
        member = np.zeros((100 * MAX_H, 100))
        np.add.at(member, group, INTERVAL_MEMBERSHIP)
        log_count.append(np.where(count > 0, np.log(np.maximum(count, 1)), -1e30))
        in_concept.append(member / np.maximum(count, 1)[:, None])
    cells = list(dict.fromkeys((index[s], int(r["target"])) for s, r in zip(sets, rows)))
    cell_index = {c: i for i, c in enumerate(cells)}
    return {
        "rule_hits": np.array(rule_hits),
        "group_log_count": np.array(log_count),
        "n_examples": np.array([float(len(s)) for s in unique]),
        "cell_set": np.array([c[0] for c in cells], dtype="int64"),
        "cell_in_concept": np.array([
            np.concatenate([RULE_MEMBERSHIP[:, t - 1], in_concept[i][:, t - 1]]) for i, t in cells
        ]),
        "row_cell": np.array([cell_index[(index[s], int(r["target"]))] for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
