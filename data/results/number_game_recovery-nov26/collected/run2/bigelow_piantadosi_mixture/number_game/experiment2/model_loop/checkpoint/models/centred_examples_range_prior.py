"""Centred-examples range prior (refines margin_beyond_examples_range_prior).

People do Bayesian concept learning over 45 arithmetic and digit rules, the 18
"digits add up to s" concepts (an inferred family share) and every interval of
1..100, reading the examples as distinct draws (without replacement) with 10%
strays and a guessing lapse, with a width-graded range prior, an inferred
discount on open-ended ranges and the incumbent's credit for how far a range
reaches past the examples, exp(beta * (room below + room above) / 100). The one
change: people expect the examples to sit in the middle of the program's range,
not off to one side of it, so a range is also weighted by
exp(kappa * |room above - room below| / (room above + room below + 1)), its
lopsidedness about the examples (kappa < 0 favours ranges centred on them). The
prior is normalised over ranges per set. This predicts "yes" beyond one end of a
cluster to be held back when the range would have to run much further on that
side than on the other, so generalization spreads evenly around the examples;
it is order-free, and rule and digit-sum generalization are unchanged.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
# exp(beta T + kappa L) is expanded about (BETA_0, KAPPA_0) in a 2-D Taylor
# series of total degree < N_DEGREE (0 <= T < 1, 0 <= L < 1).
BETA_0, KAPPA_0 = 0.8, -0.4
N_DEGREE = 8
POWERS = [(a, b) for a in range(N_DEGREE) for b in range(N_DEGREE - a)]
N_TERMS = len(POWERS)
POW_A = np.array([a for a, _ in POWERS])
POW_B = np.array([b for _, b in POWERS])
INV_FACT = np.array([1.0 / (math.factorial(a) * math.factorial(b)) for a, b in POWERS])


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


def group_sum(values):
    """Sum rows of (intervals, k) into (groups, k)."""
    out = np.zeros((N_GROUPS, values.shape[1]))
    np.add.at(out, INTERVAL_GROUP, values)
    return out


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


def room_features(examples):
    """Total room past the examples / 100, and lopsidedness of that room, for every interval."""
    above = np.clip(INTERVAL_HI - max(examples), 0, None)
    below = np.clip(min(examples) - INTERVAL_LO, 0, None)
    return (above + below) / 100.0, np.abs(above - below) / (above + below + 1.0)


with pm.Model() as model:
    rule_log_evidence = pm.Data("rule_log_evidence", np.zeros(1))  # (sets,)
    sum_log_evidence = pm.Data("sum_log_evidence", np.zeros(1))  # (sets,)
    interval_log_scale = pm.Data("interval_log_scale", np.zeros(1))  # (sets,)
    prior_moments = pm.Data("prior_moments", np.zeros((1, N_GROUPS * N_TERMS)))  # (sets, groups x terms)
    evidence_moments = pm.Data("evidence_moments", np.zeros((1, N_GROUPS * N_TERMS)))  # (sets, groups x terms)
    member_moments = pm.Data("member_moments", np.zeros((1, N_GROUPS * N_TERMS)))  # (cells, groups x terms)
    rule_in_cell = pm.Data("rule_in_cell", np.zeros(1))  # (cells,)
    sum_in_cell = pm.Data("sum_in_cell", np.zeros(1))  # (cells,)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 1.0)  # width bias of the range prior
    log_sum_share = pm.Normal("log_sum_share", -3.0, 1.5)
    log_open = pm.Normal("log_open", 0.0, 1.0)  # open-ended vs bounded range
    beta = pm.Normal("beta", 0.0, 1.0)  # credit for a range's reach beyond the examples
    kappa = pm.Normal("kappa", 0.0, 1.0)  # credit for lopsided room (negative: examples centred)

    # d_beta^a d_kappa^b / (a! b!); zero powers kept apart (d(x^0)/dx is NaN at x = 0)
    d_beta, d_kappa = beta - BETA_0, kappa - KAPPA_0
    pa = pt.concatenate([pt.ones(1), d_beta ** np.arange(1, N_DEGREE)])
    pb = pt.concatenate([pt.ones(1), d_kappa ** np.arange(1, N_DEGREE)])
    term = pa[POW_A] * pb[POW_B] * INV_FACT
    group_w = pt.exp(gamma * np.log(GROUP_WIDTH) + log_open * GROUP_OPEN)
    coef = (group_w[:, None] * term[None, :]).reshape((-1,))  # groups x terms
    log_z = pt.log(pt.dot(prior_moments, coef))
    evidence = pt.dot(evidence_moments, coef)
    interval_log_evidence = pt.log(evidence) + interval_log_scale - log_z
    interval_in_cell = pt.dot(member_moments, coef) / evidence[cell_set]

    log_rules_only = rule_log_evidence - np.log(N_RULES)
    log_sums_only = sum_log_evidence - np.log(N_SUMS) + log_sum_share
    p_sum = pm.math.sigmoid(log_sums_only - log_rules_only)[cell_set]
    finite_in_cell = (1 - p_sum) * rule_in_cell + p_sum * sum_in_cell
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
    rule_ev = np.zeros(n_sets)
    sum_ev = np.zeros(n_sets)
    scale = np.zeros(n_sets)
    prior_m = np.zeros((n_sets, N_GROUPS, N_TERMS))
    evid_m = np.zeros((n_sets, N_GROUPS, N_TERMS))
    rule_in_set = np.zeros((n_sets, 100))
    sum_in_set = np.zeros((n_sets, 100))
    member_m = np.zeros((n_cells, N_GROUPS, N_TERMS))
    for i, s in enumerate(unique):
        ll = set_log_lik(s)
        rule_ll, sum_ll, int_ll = ll[:N_RULES], ll[N_RULES:N_FINITE], ll[N_FINITE:]
        rule_ev[i] = logsumexp(rule_ll)
        sum_ev[i] = logsumexp(sum_ll)
        rule_in_set[i] = np.exp(rule_ll - rule_ev[i]) @ MEMBERSHIP[:N_RULES]
        sum_in_set[i] = np.exp(sum_ll - sum_ev[i]) @ MEMBERSHIP[N_RULES:N_FINITE]
        scale[i] = int_ll.max()
        room, lopsided = room_features(s)
        powers = (
            np.exp(BETA_0 * room + KAPPA_0 * lopsided)[:, None]
            * room[:, None] ** POW_A[None, :]
            * lopsided[:, None] ** POW_B[None, :]
        )  # (intervals, terms)
        prior_m[i] = group_sum(powers)
        weighted = np.exp(int_ll - scale[i])[:, None] * powers
        evid_m[i] = group_sum(weighted)
        for t in [t for (si, t) in cells if si == i]:
            inside = INTERVAL_MEMBERS[:, t - 1] > 0
            out = np.zeros((N_GROUPS, N_TERMS))
            np.add.at(out, INTERVAL_GROUP[inside], weighted[inside])
            member_m[cell_index[(i, t)]] = out
    cell_set = np.array([c[0] for c in cells], dtype="int64")
    cell_target = np.array([c[1] - 1 for c in cells], dtype="int64")
    return {
        "rule_log_evidence": rule_ev,
        "sum_log_evidence": sum_ev,
        "interval_log_scale": scale,
        "prior_moments": prior_m.reshape(n_sets, -1),
        "evidence_moments": evid_m.reshape(n_sets, -1),
        "member_moments": member_m.reshape(n_cells, -1),
        "rule_in_cell": rule_in_set[cell_set, cell_target],
        "sum_in_cell": sum_in_set[cell_set, cell_target],
        "cell_set": cell_set,
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


SAMPLER_SETTINGS = {"target_accept": 0.9}
