"""Heading-direction range extrapolation with a range location prior
(refines heading_direction_range_extrapolation). The one change, the component
taken from range_location_gradient_prior: an interval's log prior also rises
(delta > 0) or falls linearly with its lower bound, so ranges starting higher
up the number line are judged more (or less) plausible a priori, regardless of
the examples' order.

Base model: people read the examples as a sequence and take the direction of
the last step (whether the final example went up or down from the one before)
as where the program is heading, so the numeric ranges that come to mind extend
further past the examples in that direction and less in the other. Otherwise
Bayesian concept learning: 45 arithmetic and digit rules, the 18 digit-sum
concepts (an inferred family share), every interval of 1..100, distinct
examples drawn without replacement, 10% strays and a guessing lapse. The range
prior of a set is
w^gamma * exp(beta * heading * (room above - room below) / 100) * exp(delta * (lo - 1) / 99),
normalised over intervals per set. For speed both exponential factors are
expanded as exact Taylor series within each width group (the lower bound as an
offset from the width's centre), so the model reduces to matrix-vector products
over moments tabulated once per set and per (set, target) cell.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
K_HEAD = 16  # Taylor terms of exp(beta * c), |c| <= 1
K_LOC = 12  # Taylor terms of exp(delta * u), |u| <= 0.5
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
    return list(rules.values())


RULES = rule_concepts()
N_RULES = len(RULES)
DIGIT_SUM = np.array([sum(int(c) for c in str(n)) for n in NUMBERS])
SUM_CONCEPTS = [list(NUMBERS[DIGIT_SUM == s]) for s in range(1, 19)]
N_SUMS = len(SUM_CONCEPTS)
FINITE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + SUM_CONCEPTS], dtype="float64")
FINITE_SIZE = FINITE_MEMBERSHIP.sum(axis=1)

# Intervals on a (lo, hi) grid: entry [l, h] is the range l+1 .. h+1, valid for l <= h.
LO_GRID, HI_GRID = np.meshgrid(np.arange(1, 101.0), np.arange(1, 101.0), indexing="ij")
VALID = (LO_GRID <= HI_GRID).astype("float64")
WIDTH_GRID = np.where(VALID > 0, HI_GRID - LO_GRID + 1, 1.0)
LOG_WIDTH = np.log(WIDTH_GRID)
LOWER = (LO_GRID - 1) / 99.0
WIDTHS = np.arange(1, 101, dtype="float64")
CENTRE = (100 - WIDTHS) / 2 / 99.0  # mid lower bound (0..1 scale) of each width
OFFSET = LOWER - CENTRE[np.clip(WIDTH_GRID.astype(int) - 1, 0, 99)]  # in [-0.5, 0.5]
# Flat grid positions of the intervals of each width, and their offset powers.
WIDTH_CELLS = [np.flatnonzero((WIDTH_GRID == w).reshape(-1) & (VALID.reshape(-1) > 0)) for w in range(1, 101)]
OFFSET_POWERS = [OFFSET.reshape(-1)[j][:, None] ** np.arange(K_LOC) for j in WIDTH_CELLS]
FACT_HEAD = np.array([math.factorial(k) for k in range(K_HEAD)], dtype="float64")
FACT_LOC = np.array([math.factorial(k) for k in range(K_LOC)], dtype="float64")
N_COLS = 100 * K_HEAD * K_LOC


def finite_log_lik(examples):
    """log P(examples in order | rule) for the rules and digit-sum concepts, without replacement."""
    total = np.zeros(len(FINITE_MEMBERSHIP))
    shown = np.zeros(len(FINITE_MEMBERSHIP))
    for x in examples:
        member = FINITE_MEMBERSHIP[:, x - 1]
        remaining = np.maximum(FINITE_SIZE - shown, 1.0)
        total += np.log(ALPHA * member / remaining + (1 - ALPHA) / 100)
        shown += member
    return total


def interval_log_lik(examples):
    """The same likelihood for every interval on the (lo, hi) grid."""
    total = np.zeros((100, 100))
    shown = np.zeros((100, 100))
    for x in examples:
        member = ((LO_GRID <= x) & (HI_GRID >= x)).astype("float64")
        remaining = np.maximum(WIDTH_GRID - shown, 1.0)
        total += np.log(ALPHA * member / remaining + (1 - ALPHA) / 100)
        shown += member
    return total


def heading_room(examples):
    """heading * (room above - room below) / 100 on the grid, in [-1, 1]."""
    heading = float(np.sign(examples[-1] - examples[-2])) if len(examples) > 1 else 0.0
    above = np.clip(HI_GRID - max(examples), 0, None)
    below = np.clip(min(examples) - LO_GRID, 0, None)
    return heading * (above - below) / 100.0


with pm.Model() as model:
    rule_log_evidence = pm.Data("rule_log_evidence", np.zeros(1))  # (sets,)
    sum_log_evidence = pm.Data("sum_log_evidence", np.zeros(1))  # (sets,)
    interval_log_scale = pm.Data("interval_log_scale", np.zeros(1))  # (sets,)
    prior_moments = pm.Data("prior_moments", np.zeros((1, N_COLS)))  # (sets, widths x k_head x k_loc)
    evidence_moments = pm.Data("evidence_moments", np.zeros((1, N_COLS)))  # (sets, ...)
    member_moments = pm.Data("member_moments", np.zeros((1, N_COLS)))  # (cells, ...)
    rule_in_cell = pm.Data("rule_in_cell", np.zeros(1))  # (cells,)
    sum_in_cell = pm.Data("sum_in_cell", np.zeros(1))  # (cells,)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 1.0)  # width bias of the range prior
    beta = pm.Normal("beta", 0.0, 1.0)  # pull of the ranges toward the heading direction
    delta = pm.Normal("delta", 0.0, 1.0)  # location gradient of the range prior (lower bound 1 -> 100)
    log_sum_share = pm.Normal("log_sum_share", -3.0, 1.5)

    # x^k / k! (k = 0 kept apart: d(x^0)/dx is NaN at x = 0)
    beta_powers = pt.concatenate([pt.ones(1), beta ** np.arange(1, K_HEAD)]) / FACT_HEAD
    delta_powers = pt.concatenate([pt.ones(1), delta ** np.arange(1, K_LOC)]) / FACT_LOC
    width_w = pt.exp(gamma * np.log(WIDTHS) + delta * CENTRE)
    coef = (width_w[:, None, None] * beta_powers[None, :, None] * delta_powers[None, None, :]).reshape((-1,))
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


def moments(weights, head):
    """sum over intervals of weights * c^k1 * u^k2 per width: (n, widths * K_HEAD * K_LOC)."""
    w = weights.reshape(-1, 10000)
    hp = head.reshape(-1)[:, None] ** np.arange(K_HEAD)  # (10000, K_HEAD)
    out = np.zeros((len(w), 100, K_HEAD, K_LOC))
    for k, (j, up) in enumerate(zip(WIDTH_CELLS, OFFSET_POWERS)):
        out[:, k] = np.einsum("nm,ma,mb->nab", w[:, j], hp[j], up, optimize=True)
    return out.reshape(len(w), -1)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    keys = [(set_index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    n_sets = len(unique)
    rule_ev = np.zeros(n_sets)
    sum_ev = np.zeros(n_sets)
    scale = np.zeros(n_sets)
    prior_m = np.zeros((n_sets, N_COLS))
    evid_m = np.zeros((n_sets, N_COLS))
    member_m = np.zeros((len(cells), N_COLS))
    rule_in_set = np.zeros((n_sets, 100))
    sum_in_set = np.zeros((n_sets, 100))
    for i, s in enumerate(unique):
        ll = finite_log_lik(s)
        rule_ll, sum_ll = ll[:N_RULES], ll[N_RULES:]
        rule_ev[i] = logsumexp(rule_ll)
        sum_ev[i] = logsumexp(sum_ll)
        rule_in_set[i] = np.exp(rule_ll - rule_ev[i]) @ FINITE_MEMBERSHIP[:N_RULES]
        sum_in_set[i] = np.exp(sum_ll - sum_ev[i]) @ FINITE_MEMBERSHIP[N_RULES:]
        int_ll = interval_log_lik(s)
        scale[i] = int_ll[VALID > 0].max()
        evid = np.exp(int_ll - scale[i]) * VALID
        head = heading_room(s) * VALID
        prior_m[i] = moments(VALID, head)[0]
        evid_m[i] = moments(evid, head)[0]
        targets = [t for (si, t) in cells if si == i]
        covers = np.stack([evid * (LO_GRID <= t) * (HI_GRID >= t) for t in targets])
        member_m[[cell_index[(i, t)] for t in targets]] = moments(covers, head)
    cell_set = np.array([c[0] for c in cells], dtype="int64")
    cell_target = np.array([c[1] - 1 for c in cells], dtype="int64")
    return {
        "rule_log_evidence": rule_ev,
        "sum_log_evidence": sum_ev,
        "interval_log_scale": scale,
        "prior_moments": prior_m,
        "evidence_moments": evid_m,
        "member_moments": member_m,
        "rule_in_cell": rule_in_set[cell_set, cell_target],
        "sum_in_cell": sum_in_set[cell_set, cell_target],
        "cell_set": cell_set,
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


SAMPLER_SETTINGS = {"target_accept": 0.9}
