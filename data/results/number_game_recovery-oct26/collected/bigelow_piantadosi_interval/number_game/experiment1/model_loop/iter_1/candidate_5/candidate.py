"""Refinement of bigelow_piantadosi_independent with an inferred outlier rate.

People do Bayesian concept learning over 42 weighted rules and all 5,050
intervals of 1-100 with the size principle, but the probability that an
example is a stray uniform draw from 1-100 rather than a member of the
concept (fixed at 10% in the original) is a learner property, inferred.
A high rate lets a concept that fits most but not all examples survive and
weakens how sharply examples narrow generalization; a low rate makes one
misfit example rule a concept out. "Yes" is the posterior probability that
the concept contains the target.
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
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
# Intervals [m, n]: size and membership.
IV_LO = np.array([m for m in range(1, 101) for n in range(m, 101)])
IV_HI = np.array([n for m in range(1, 101) for n in range(m, 101)])
IV_SIZE = IV_HI - IV_LO + 1
IV_MEMBERSHIP = ((NUMBERS[None, :] >= IV_LO[:, None]) & (NUMBERS[None, :] <= IV_HI[:, None])).astype("float64")
N_INTERVALS = len(IV_SIZE)
SIZES = np.arange(1, 101, dtype="float64")
COUNTS = np.arange(MAX_N + 1, dtype="float64")

# The likelihood of an example set under an interval depends only on the
# interval's size s and how many examples c it contains, so the intervals are
# grouped by (s, c): per set, the number of intervals in each group and the
# fraction of them containing each target are computed once.

with pm.Model() as model:
    rule_count = pm.Data("rule_count", np.zeros((1, N_RULES)))  # (sets, rules): examples inside rule
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    group_log_n = pm.Data("group_log_n", np.zeros((1, 100 * (MAX_N + 1))))  # (sets, s*c) log #intervals
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # set of each distinct (set, target) cell
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_group_frac = pm.Data("cell_group_frac", np.zeros((1, 100 * (MAX_N + 1))))  # (cells, s*c)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # cell of each response
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    epsilon = pm.Beta("epsilon", 2, 12)  # outlier rate, prior mean ~0.14

    log_out = pt.log(epsilon) - np.log(100.0)
    n = n_examples[:, None]
    # Rules: c * log((1-e)/|h| + e/100) + (n - c) * log(e/100)
    log_in_rule = pt.log((1 - epsilon) / RULE_SIZE + epsilon / 100.0)
    rule_log_lik = rule_count * log_in_rule[None, :] + (n - rule_count) * log_out
    # Interval groups (s, c), flattened s-major.
    log_in_size = pt.log((1 - epsilon) / SIZES + epsilon / 100.0)  # (100,)
    grid_c = np.tile(COUNTS, 100)
    group_ll = pt.repeat(log_in_size, MAX_N + 1) * grid_c  # (s*c,)
    group_ll = group_ll[None, :] + (n - grid_c[None, :]) * log_out + group_log_n
    interval_log_evidence = pt.logsumexp(group_ll, axis=1)  # (sets,)
    group_post = pt.exp(group_ll - interval_log_evidence[:, None])

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_intervals = pt.log1p(-p_mix) - np.log(N_INTERVALS) + interval_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    # Only the (set, target) cells that were asked about are evaluated.
    interval_in_concept = (group_post[cell_set] * cell_group_frac).sum(axis=1)  # (cells,)
    post_cell = posterior[cell_set]
    in_concept = (post_cell[:, :N_RULES] * cell_rule_member).sum(axis=1) + post_cell[:, N_RULES] * interval_in_concept

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_groups = 100 * (MAX_N + 1)
    rule_count = np.zeros((len(unique), N_RULES))
    log_n = np.zeros((len(unique), n_groups))
    frac = np.zeros((len(unique), n_groups, 100))
    for k, s in enumerate(unique):
        cols = np.array(s) - 1
        rule_count[k] = RULE_MEMBERSHIP[:, cols].sum(axis=1)
        c = IV_MEMBERSHIP[:, cols].sum(axis=1).astype(int)
        g = (IV_SIZE - 1) * (MAX_N + 1) + c
        cnt = np.bincount(g, minlength=n_groups).astype("float64")
        member = np.zeros((n_groups, 100))
        np.add.at(member, g, IV_MEMBERSHIP)
        log_n[k] = np.where(cnt > 0, np.log(np.maximum(cnt, 1.0)), -1e4)
        frac[k] = member / np.maximum(cnt, 1.0)[:, None]
    keys = [index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cs = np.array([c // 100 for c in cells], dtype="int64")
    ct = np.array([c % 100 for c in cells], dtype="int64")
    return {
        "rule_count": rule_count,
        "n_examples": np.array([float(len(s)) for s in unique]),
        "group_log_n": log_n,
        "cell_set": cs,
        "cell_rule_member": RULE_MEMBERSHIP[:, ct].T.copy(),
        "cell_group_frac": frac[cs, :, ct],
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
