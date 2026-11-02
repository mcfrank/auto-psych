"""Overall trend direction (first to last example) with Weber width and upward counting.

Refinement of `heading_weber_with_upward_counting`: people do Bayesian concept
learning over 45 arithmetic and digit rules, the 18 digit-sum concepts (an
inferred family share) and every interval of 1..100, reading the examples as
distinct draws without replacement with 10% strays and a guessing lapse,
judging a range's prior breadth relative to its magnitude, (width / midpoint)^gamma,
and expecting the numbers to carry on upward past the largest example. The one
change is what sets the sequence's direction: people read the overall trend of
the sequence, from the first example shown to the last, rather than the
direction of the final step, and that trend tilts the room they expect beyond
the examples (strength and sign inferred; a negative sign means they take the
shown run to have started partway into the program's range, so ranges reach
back past where the sequence began). The range prior is
(w/mid)^gamma * exp(beta * trend * (above - below) / 100 + delta * above / 100),
normalised over ranges per set, with trend = sign(last - first).
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
K_G = 12  # Taylor terms in gamma * scaled log relative width
K_R = 10  # Taylor terms in each of the room-above and room-below exponents


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
# log(width / midpoint), centred and scaled to [-1, 1]
LOG_REL = np.log((INTERVAL_HI - INTERVAL_LO + 1) / ((INTERVAL_LO + INTERVAL_HI) / 2))
REL_CENTRE = (LOG_REL.max() + LOG_REL.min()) / 2
REL_HALF = (LOG_REL.max() - LOG_REL.min()) / 2
REL_SCALED = (LOG_REL - REL_CENTRE) / REL_HALF
REL_POWERS = REL_SCALED[:, None] ** np.arange(K_G)[None, :]  # (intervals, K_G)
FACT_G = np.array([math.factorial(k) for k in range(K_G)], dtype="float64")
FACT_R = np.array([math.factorial(k) for k in range(K_R)], dtype="float64")


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


def heading_of(examples):
    """Overall trend of the sequence: sign of last minus first example."""
    return float(np.sign(examples[-1] - examples[0])) if len(examples) > 1 else 0.0


def rooms(examples):
    """Room above the largest and below the smallest example for every interval, /100."""
    above = np.clip(INTERVAL_HI - max(examples), 0, None) / 100.0
    below = np.clip(min(examples) - INTERVAL_LO, 0, None) / 100.0
    return above, below


def taylor_coef(x, n_terms, factorial):
    """x^k / k! for k < n_terms (k = 0 kept apart: d(x^0)/dx is NaN at x = 0)."""
    return pt.concatenate([pt.ones(1), x ** np.arange(1, n_terms)]) / factorial


# With heading h, the room exponent is (delta + h beta) * above - h beta * below,
# so the prior factor is a 3-D Taylor series in (relative width, above, below);
# prepare_observed tabulates the joint moments per set and per (set, target)
# cell, and the model only needs one coefficient vector per heading class.

with pm.Model() as model:
    rule_log_evidence = pm.Data("rule_log_evidence", np.zeros(1))  # (sets,)
    sum_log_evidence = pm.Data("sum_log_evidence", np.zeros(1))  # (sets,)
    interval_log_scale = pm.Data("interval_log_scale", np.zeros(1))  # (sets,)
    prior_moments = pm.Data("prior_moments", np.zeros((1, K_G * K_R * K_R)))  # (sets, moments)
    evidence_moments = pm.Data("evidence_moments", np.zeros((1, K_G * K_R * K_R)))  # (sets, moments)
    member_moments = pm.Data("member_moments", np.zeros((1, K_G * K_R * K_R)))  # (cells, moments)
    set_heading = pm.Data("set_heading", np.zeros(1, dtype="int64"))  # (sets,) 0: down, 1: none, 2: up
    rule_in_cell = pm.Data("rule_in_cell", np.zeros(1))  # (cells,)
    sum_in_cell = pm.Data("sum_in_cell", np.zeros(1))  # (cells,)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 0.5)  # bias on relative width of the range prior
    beta = pm.Normal("beta", 0.0, 1.0)  # tilt of the ranges toward (+) or back against (-) the overall trend
    delta = pm.Normal("delta", 0.0, 0.5)  # order-independent credit for room above the examples
    log_sum_share = pm.Normal("log_sum_share", -3.0, 1.5)

    coef_g = taylor_coef(gamma * REL_HALF, K_G, FACT_G)
    coefs = []
    for h in (-1.0, 0.0, 1.0):
        coef_a = taylor_coef(delta + h * beta, K_R, FACT_R)
        coef_b = taylor_coef(-h * beta, K_R, FACT_R) if h != 0.0 else pt.as_tensor(np.eye(K_R)[0])
        coefs.append((coef_g[:, None, None] * coef_a[None, :, None] * coef_b[None, None, :]).reshape((-1,)))
    coef = pt.stack(coefs, axis=1)  # (moments, 3 heading classes)
    set_rows = pt.arange(set_heading.shape[0])
    cell_heading = set_heading[cell_set]

    log_z = pt.log(pt.dot(prior_moments, coef)[set_rows, set_heading])
    evidence = pt.dot(evidence_moments, coef)[set_rows, set_heading]
    interval_log_evidence = pt.log(evidence) + interval_log_scale - log_z
    member_mass = pt.dot(member_moments, coef)[pt.arange(cell_set.shape[0]), cell_heading]
    interval_in_cell = member_mass / evidence[cell_set]

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


def joint_moments(rel_w, above_p, below_p):
    """sum_i rel_w[i,k] above_p[i,a] below_p[i,b] -> flat (K_G * K_R * K_R,)."""
    ra = (rel_w[:, :, None] * above_p[:, None, :]).reshape(len(rel_w), -1)
    return (ra.T @ below_p).reshape(-1)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    keys = [(set_index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    n_sets, n_cells, n_mom = len(unique), len(cells), K_G * K_R * K_R
    rule_ev = np.zeros(n_sets)
    sum_ev = np.zeros(n_sets)
    scale = np.zeros(n_sets)
    heading = np.zeros(n_sets, dtype="int64")
    prior_m = np.zeros((n_sets, n_mom))
    evid_m = np.zeros((n_sets, n_mom))
    rule_in_set = np.zeros((n_sets, 100))
    sum_in_set = np.zeros((n_sets, 100))
    member_m = np.zeros((n_cells, n_mom))
    for i, s in enumerate(unique):
        ll = set_log_lik(s)
        rule_ll, sum_ll, int_ll = ll[:N_RULES], ll[N_RULES:N_FINITE], ll[N_FINITE:]
        rule_ev[i] = logsumexp(rule_ll)
        sum_ev[i] = logsumexp(sum_ll)
        rule_in_set[i] = np.exp(rule_ll - rule_ev[i]) @ MEMBERSHIP[:N_RULES]
        sum_in_set[i] = np.exp(sum_ll - sum_ev[i]) @ MEMBERSHIP[N_RULES:N_FINITE]
        scale[i] = int_ll.max()
        heading[i] = int(heading_of(s)) + 1
        above, below = rooms(s)
        above_p = above[:, None] ** np.arange(K_R)[None, :]
        below_p = below[:, None] ** np.arange(K_R)[None, :]
        prior_m[i] = joint_moments(REL_POWERS, above_p, below_p)
        weighted = np.exp(int_ll - scale[i])[:, None] * REL_POWERS
        evid_m[i] = joint_moments(weighted, above_p, below_p)
        for t in [t for (si, t) in cells if si == i]:
            inside = INTERVAL_MEMBERS[:, t - 1] > 0
            member_m[cell_index[(i, t)]] = joint_moments(weighted[inside], above_p[inside], below_p[inside])
    cell_set = np.array([c[0] for c in cells], dtype="int64")
    cell_target = np.array([c[1] - 1 for c in cells], dtype="int64")
    return {
        "rule_log_evidence": rule_ev,
        "sum_log_evidence": sum_ev,
        "interval_log_scale": scale,
        "prior_moments": prior_m,
        "evidence_moments": evid_m,
        "member_moments": member_m,
        "set_heading": heading,
        "rule_in_cell": rule_in_set[cell_set, cell_target],
        "sum_in_cell": sum_in_set[cell_set, cell_target],
        "cell_set": cell_set,
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


SAMPLER_SETTINGS = {"target_accept": 0.9}
