"""Size principle for rules only; numeric ranges by weak sampling.

People treat the two kinds of concept differently. For arithmetic, digit and
digit-sum rules they reason with the size principle (strong sampling without
replacement, 10% strays): a small rule that fits all the examples is a
suspicious coincidence. For stretches of the number line they do not: an
example is merely a labelled number that the range contains (weak sampling,
probability 1/(100 - shown) if inside, with 10% strays), so every range that
covers the examples stays credible in proportion to its prior alone, however
wide. The range prior is the best model's (width-graded, open-ended discount,
margin credit beyond the examples) and there is a guessing lapse. No new free
parameter. Prediction: generalization along magnitude does not tighten as more
examples fall in a cluster (60 vs 60, 62, 64), while rule generalization does.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
N_TERMS = 14  # Taylor terms of exp(beta * c), 0 <= c < 1: relative error < 1e-4 for |beta| < 3


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


RULES = rule_concepts()
N_RULES = len(RULES)
DIGIT_SUM = np.array([sum(int(c) for c in str(n)) for n in NUMBERS])
SUM_CONCEPTS = [list(NUMBERS[DIGIT_SUM == s]) for s in range(1, 19)]
N_SUMS = len(SUM_CONCEPTS)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + SUM_CONCEPTS + INTERVALS], dtype="float64")
SIZE = MEMBERSHIP.sum(axis=1)
N_FINITE = N_RULES + N_SUMS
INTERVAL_MEMBERS = MEMBERSHIP[N_FINITE:]
INTERVAL_LO = np.array([iv[0] for iv in INTERVALS], dtype="float64")
INTERVAL_HI = np.array([iv[-1] for iv in INTERVALS], dtype="float64")
INTERVAL_WIDTH = (INTERVAL_HI - INTERVAL_LO + 1).astype(int)
INTERVAL_OPEN = ((INTERVAL_LO == 1) | (INTERVAL_HI == 100)).astype(int)
# Intervals are grouped by (width, open-ended flag); only non-empty groups kept.
_GROUP_KEYS = sorted(set(zip(INTERVAL_WIDTH.tolist(), INTERVAL_OPEN.tolist())))
_GROUP_OF = {g: i for i, g in enumerate(_GROUP_KEYS)}
INTERVAL_GROUP = np.array([_GROUP_OF[(w, o)] for w, o in zip(INTERVAL_WIDTH, INTERVAL_OPEN)])
N_GROUPS = len(_GROUP_KEYS)
GROUP_WIDTH = np.array([g[0] for g in _GROUP_KEYS], dtype="float64")
GROUP_OPEN = np.array([g[1] for g in _GROUP_KEYS], dtype="float64")
GROUP_ONEHOT = (INTERVAL_GROUP[:, None] == np.arange(N_GROUPS)[None, :]).astype("float64")
FACTORIAL = np.array([math.factorial(k) for k in range(N_TERMS)], dtype="float64")


def set_log_lik(examples):
    """log P(examples in order | h) for every hypothesis, sampling members without replacement."""
    total = np.zeros(len(MEMBERSHIP))
    shown = np.zeros(len(MEMBERSHIP))
    for x in examples:
        member = MEMBERSHIP[:, x - 1]
        remaining = np.maximum(SIZE - shown, 1.0)
        total += np.log(ALPHA * member / remaining + (1 - ALPHA) / 100)
        shown += member
    return total


def weak_log_lik(examples):
    """log P(examples | h) under weak sampling: distinct arbitrary numbers, labelled if in h."""
    total = np.zeros(len(MEMBERSHIP))
    for k, x in enumerate(examples):
        total += np.log(ALPHA * MEMBERSHIP[:, x - 1] / (100 - k) + (1 - ALPHA) / 100)
    return total


def margin(examples):
    """(room below the smallest example + room above the largest) / 100 for every interval."""
    above = np.clip(INTERVAL_HI - max(examples), 0, None)
    below = np.clip(min(examples) - INTERVAL_LO, 0, None)
    return (above + below) / 100.0


# Rules and digit-sum concepts are weighed by the size principle (strong
# sampling); intervals by weak sampling (their evidence is the same for every
# interval that contains all the examples, so only the prior separates them).
# Interval tables are Taylor moments of the margin factor exp(beta * c) per
# (width, open) group.

with pm.Model() as model:
    rule_ev = pm.Data("rule_ev", np.zeros(1))  # (sets,)
    sum_ev = pm.Data("sum_ev", np.zeros(1))
    scale = pm.Data("scale", np.zeros(1))  # (sets,)
    prior_moments = pm.Data("prior_moments", np.zeros((1, N_GROUPS * N_TERMS)))  # (sets, groups x k)
    evid = pm.Data("evid", np.zeros((1, N_GROUPS * N_TERMS)))
    member = pm.Data("member", np.zeros((1, N_GROUPS * N_TERMS)))  # (cells, groups x k)
    rule_in = pm.Data("rule_in", np.zeros(1))  # (cells,)
    sum_in = pm.Data("sum_in", np.zeros(1))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 1.0)  # width bias of the range prior
    log_sum_share = pm.Normal("log_sum_share", -3.0, 1.5)
    log_open = pm.Normal("log_open", 0.0, 1.0)  # open-ended vs bounded range
    beta = pm.Normal("beta", 0.0, 1.0)  # credit for a range's reach beyond the examples

    beta_powers = pt.concatenate([pt.ones(1), beta ** np.arange(1, N_TERMS)]) / FACTORIAL
    group_w = pt.exp(gamma * np.log(GROUP_WIDTH) + log_open * GROUP_OPEN)
    coef = (group_w[:, None] * beta_powers[None, :]).reshape((-1,))  # groups x k
    log_z = pt.log(pt.dot(prior_moments, coef))
    e = pt.dot(evid, coef)
    interval_log_evidence = pt.log(e) + scale - log_z
    interval_in_cell = pt.dot(member, coef) / e[cell_set]

    log_rules_only = rule_ev - np.log(N_RULES)
    log_sums_only = sum_ev - np.log(N_SUMS) + log_sum_share
    p_sum = pm.math.sigmoid(log_sums_only - log_rules_only)[cell_set]
    finite_in_cell = (1 - p_sum) * rule_in + p_sum * sum_in
    log_finite = (
        pt.log(lambda_0) + pt.logaddexp(log_rules_only, log_sums_only) - pt.logaddexp(0.0, log_sum_share)
    )
    log_intervals = pt.log1p(-lambda_0) + interval_log_evidence
    p_rule = pm.math.sigmoid(log_finite - log_intervals)[cell_set]
    in_concept = p_rule * finite_in_cell + (1 - p_rule) * interval_in_cell

    p_concept = pt.clip(in_concept[row_cell], 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    keys = [(set_index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    n_sets, n_cells = len(unique), len(cells)
    cell_set = np.array([c[0] for c in cells], dtype="int64")
    cell_target = np.array([c[1] - 1 for c in cells], dtype="int64")
    rule_ev, sum_ev, scale = np.zeros(n_sets), np.zeros(n_sets), np.zeros(n_sets)
    rule_in_set, sum_in_set = np.zeros((n_sets, 100)), np.zeros((n_sets, 100))
    prior_m = np.zeros((n_sets, N_GROUPS, N_TERMS))
    evid_m = np.zeros((n_sets, N_GROUPS, N_TERMS))
    member_m = np.zeros((n_cells, N_GROUPS, N_TERMS))
    for i, s in enumerate(unique):
        strong, weak = set_log_lik(s), weak_log_lik(s)
        rule_ll, sum_ll, int_ll = strong[:N_RULES], strong[N_RULES:N_FINITE], weak[N_FINITE:]
        rule_ev[i] = logsumexp(rule_ll)
        sum_ev[i] = logsumexp(sum_ll)
        rule_in_set[i] = np.exp(rule_ll - rule_ev[i]) @ MEMBERSHIP[:N_RULES]
        sum_in_set[i] = np.exp(sum_ll - sum_ev[i]) @ MEMBERSHIP[N_RULES:N_FINITE]
        scale[i] = int_ll.max()
        powers = margin(s)[:, None] ** np.arange(N_TERMS)[None, :]
        prior_m[i] = GROUP_ONEHOT.T @ powers
        weighted = np.exp(int_ll - scale[i])[:, None] * powers
        evid_m[i] = GROUP_ONEHOT.T @ weighted
        for t in [t for (si, t) in cells if si == i]:
            inside = INTERVAL_MEMBERS[:, t - 1] > 0
            member_m[cell_index[(i, t)]] = GROUP_ONEHOT[inside].T @ weighted[inside]
    return {
        "rule_ev": rule_ev,
        "sum_ev": sum_ev,
        "scale": scale,
        "prior_moments": prior_m.reshape(n_sets, -1),
        "evid": evid_m.reshape(n_sets, -1),
        "member": member_m.reshape(n_cells, -1),
        "rule_in": rule_in_set[cell_set, cell_target],
        "sum_in": sum_in_set[cell_set, cell_target],
        "cell_set": cell_set,
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


SAMPLER_SETTINGS = {"target_accept": 0.9}
