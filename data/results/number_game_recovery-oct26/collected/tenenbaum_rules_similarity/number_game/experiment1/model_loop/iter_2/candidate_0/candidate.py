"""Primacy-anchored evidence in Bayesian concept learning: people infer a
posterior over 42 rules and every interval of 1-100 (size principle, 10% of
examples random, Erlang prior over interval width, as in
erlang_interval_size_prior), but they anchor on their first impression: the
first example shown counts fully as evidence and one shown k positions after
it counts with weight retention**k (retention inferred), as updating declines
with each new example. "Yes" is the resulting posterior probability that the
concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_EXAMPLES = 4


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
RULE_LOG_LIK = np.log(RULE_MEMBERSHIP * (1 - EPSILON) / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
WIDTHS = np.arange(1, 101, dtype="float64")
IN_LOG_LIK = np.log((1 - EPSILON) / WIDTHS + EPSILON / 100)  # example inside an interval, by width
OUT_LOG_LIK = np.log(EPSILON / 100)  # example outside it

# Per set, intervals of one width that contain the same examples share prior
# and (memory-weighted) evidence, so each such group acts as one hypothesis;
# prepare_observed pads every set to MAX_GROUPS groups.
MAX_GROUPS = 520

with pm.Model() as model:
    # Per set: each hypothesis's log-likelihood of each example slot (zero for
    # empty slots), how many positions after the first each example was shown,
    # and each interval group's width index and log interval count (-1e4 = padding).
    example_log_lik = pm.Data("example_log_lik", np.zeros((1, MAX_EXAMPLES, N_RULES + MAX_GROUPS)))
    example_position = pm.Data("example_position", np.zeros((1, MAX_EXAMPLES)))
    group_width = pm.Data("group_width", np.zeros((1, MAX_GROUPS), dtype="int64"))
    group_log_count = pm.Data("group_log_count", np.zeros((1, MAX_GROUPS)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_in_hypothesis = pm.Data("cell_in_hypothesis", np.zeros((1, N_RULES + MAX_GROUPS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    # Evidence weight retained per position after the first example.
    retention = pm.Beta("retention", 2.0, 2.0)

    # Erlang over width, per interval (101 - w intervals of width w).
    log_width = np.log(WIDTHS) - WIDTHS / width_scale
    log_width_prior = log_width - pt.logsumexp(log_width + np.log(101 - WIDTHS))
    log_rule_prior = pt.log(p_mix) + pt.log(rule_weights)
    log_group_prior = pt.log1p(-p_mix) + log_width_prior[group_width] + group_log_count
    log_prior = pt.concatenate(
        [pt.ones((group_width.shape[0], 1)) * log_rule_prior[None, :], log_group_prior], axis=1
    )

    weights = pt.exp(example_position * pt.log(retention))
    log_lik = (weights[:, :, None] * example_log_lik).sum(axis=1)
    log_joint = log_prior + log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[cell_set] * cell_in_hypothesis).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def interval_groups(examples):
    """Group every interval by (width, which examples it contains): coverage of each number."""
    ex = np.array(examples)
    groups = {}
    for lo in range(1, 101):
        for hi in range(lo, 101):
            key = (hi - lo + 1, tuple((lo <= ex) & (ex <= hi)))
            groups.setdefault(key, np.zeros(100))[lo - 1 : hi] += 1.0
    return groups


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    log_lik = np.zeros((len(unique), MAX_EXAMPLES, N_RULES + MAX_GROUPS))
    position = np.zeros((len(unique), MAX_EXAMPLES))
    width = np.zeros((len(unique), MAX_GROUPS), dtype="int64")
    log_count = np.full((len(unique), MAX_GROUPS), -1e4)
    share = np.zeros((len(unique), MAX_GROUPS, 100))  # share of a group's intervals holding each number
    for i, s in enumerate(unique):
        for j, x in enumerate(s):
            log_lik[i, j, :N_RULES] = RULE_LOG_LIK[:, x - 1]
            position[i, j] = j
        groups = interval_groups(s)
        if len(groups) > MAX_GROUPS:
            raise ValueError("more interval groups than MAX_GROUPS")
        for g, ((w, inside), cover) in enumerate(groups.items()):
            n_intervals = cover.sum() / w
            width[i, g] = w - 1
            log_count[i, g] = np.log(n_intervals)
            share[i, g] = cover / n_intervals
            for j, isin in enumerate(inside):
                log_lik[i, j, N_RULES + g] = IN_LOG_LIK[w - 1] if isin else OUT_LOG_LIK
    cell_set = np.array([index[s] for s, _ in cells], dtype="int64")
    cell_target = np.array([t - 1 for _, t in cells])
    return {
        "example_log_lik": log_lik,
        "example_position": position,
        "group_width": width,
        "group_log_count": log_count,
        "cell_set": cell_set,
        "cell_in_hypothesis": np.concatenate(
            [RULE_MEMBERSHIP[:, cell_target].T, share[cell_set, :, cell_target]], axis=1
        ),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
