"""Round-edge ranges on graded final-digit and digit-sum concepts.

Refinement of `graded_final_digit_digit_sum_concepts`, adding one component
from `round_number_range_boundaries`: people do the same Bayesian concept
learning over number rules (a prior mass per family, exception rules in their
own family), digit-sum rules whose within-family prior is proportional to
exp(-dsum_decay * s), final-digit rules whose prior is proportional to
exp(-digit_decay * d), and the 5,050 ranges of 1..100 (Erlang prior on width,
uniform outputs, the size principle, 10% of examples uniform noise). The one
change: multiples of ten are reference points when people judge how plausible
a range is a priori, so a range's prior is multiplied by exp(round_bonus) for
each of its edges that falls exactly on a multiple of ten (direction and
strength inferred; 0 recovers the parent). This decides whether
magnitude-based "yes" answers stop just before or exactly at a nearby
multiple of ten.

Range likelihoods are parameter-free, so prepare_observed sums them per set
and per observed (set, target) cell, grouped by (width, number of round
edges), and the graph works on (cells x 300) arrays.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
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
    rules.update({f"digit sum {t}": [n for n in NUMBERS if sum(map(int, str(n))) == t] for t in range(1, 19)})
    return rules


def family(name):
    if name in ("odd", "even"):
        return 0
    if name.startswith("multiples"):
        return 2
    if name.startswith("ends"):
        return 3
    if name.startswith("contains"):
        return 4
    if name.startswith("digit sum"):
        return 6
    if name.startswith("powers of 2,"):
        return 5  # gerrymandered exception rules
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
N_RULES = len(_RULES)
RULE_FAMILY = np.array([family(n) for n in _RULES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
DSUM_INDEX = np.where(RULE_FAMILY == 6)[0]
DSUM_VALUE = np.arange(1, 19, dtype="float64")
ENDS_INDEX = np.where(RULE_FAMILY == 3)[0]
ENDS_DIGIT = np.arange(10, dtype="float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in _RULES.values()], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)

LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
WIDTH = HI - LO + 1
WIDTHS = np.arange(1, 101, dtype="float64")
N_PER_WIDTH = 101.0 - WIDTHS
N_ROUND = ((LO % 10) == 0).astype(int) + ((HI % 10) == 0).astype(int)
GROUP = (WIDTH - 1) * 3 + N_ROUND  # (width, round-edge count) group of each range
GROUP_ONEHOT = np.eye(300)[GROUP]  # (ranges, groups)
GROUP_COUNT = GROUP_ONEHOT.sum(axis=0)
GROUP_WIDTH = np.repeat(WIDTHS, 3)
GROUP_ROUND = np.tile(np.arange(3, dtype="float64"), 100)
GROUP_LOG_COUNT = np.where(GROUP_COUNT > 0, np.log(np.maximum(GROUP_COUNT, 1.0)), -np.inf)


def interval_log_lik(examples):
    ll = np.zeros(len(LO))
    for x in examples:
        inside = (LO <= x) & (x <= HI)
        ll += np.log(ALPHA * inside / WIDTH + (1 - ALPHA) / 100)
    return ll


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules), minus set max
    set_width = pm.Data("set_width", np.zeros((1, 300)))  # (sets, groups) summed range likelihoods
    cell_width = pm.Data("cell_width", np.zeros((1, 300)))  # (cells, groups), ranges containing target
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the seven rule families relative to ranges, as log-odds
    # against the range family (softmax of unconstrained logits: no simplex
    # boundary for families the data push towards zero).
    family_logit = pm.Normal("family_logit", mu=-2.0, sigma=2.5, shape=7)
    family_weight = pm.Deterministic(
        "family_weight", pt.special.softmax(pt.concatenate([family_logit, pt.zeros(1)]))
    )
    # Characteristic range width of the Erlang size prior.
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    # Gradient of prior plausibility over digit sums (incumbent mechanism).
    dsum_decay = pm.Normal("dsum_decay", mu=0.0, sigma=0.5)
    # The change: gradient of prior plausibility over final digits.
    digit_decay = pm.Normal("digit_decay", mu=0.0, sigma=0.5)
    # The change: log prior-odds bonus per range edge on a multiple of ten.
    round_bonus = pm.Normal("round_bonus", mu=0.0, sigma=1.0)

    dsum_share = pt.special.softmax(-dsum_decay * DSUM_VALUE)
    ends_share = pt.special.softmax(-digit_decay * ENDS_DIGIT)
    within = pt.as_tensor(1.0 / FAMILY_SIZE[RULE_FAMILY])
    within = pt.set_subtensor(within[DSUM_INDEX], dsum_share)
    within = pt.set_subtensor(within[ENDS_INDEX], ends_share)
    rule_prior = family_weight[:7][RULE_FAMILY] * within
    rule_weight = rule_prior[None, :] * pt.exp(rule_log_lik)  # (sets, rules)
    rule_total = rule_weight.sum(axis=1)
    rule_in = (rule_weight @ pt.as_tensor(RULE_MEMBERSHIP))[cell_set, cell_target]

    raw = np.log(GROUP_WIDTH) - GROUP_WIDTH / interval_scale + round_bonus * GROUP_ROUND
    log_per_range = raw - pt.logsumexp(raw + GROUP_LOG_COUNT)  # (groups,)
    width_weight = family_weight[7] * pt.exp(log_per_range)
    int_total = pt.dot(set_width, width_weight)  # (sets,)
    int_in = pt.dot(cell_width, width_weight)  # (cells,)

    p_cell = (rule_in + int_in) / (rule_total + int_total)[cell_set]
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    rule_ll = np.zeros((len(unique), N_RULES))
    set_w = np.zeros((len(unique), 300))
    range_weights = []
    for i, s in enumerate(unique):
        r_ll = RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1)
        i_ll = interval_log_lik(s)
        top = max(r_ll.max(), i_ll.max())
        rule_ll[i] = r_ll - top
        e = np.exp(i_ll - top)
        range_weights.append(e)
        set_w[i] = GROUP_ONEHOT.T @ e
    cell_w = np.zeros((len(cells), 300))
    for j, (s, t) in enumerate(cells):
        cell_w[j] = GROUP_ONEHOT.T @ (range_weights[set_index[s]] * ((LO <= t) & (t <= HI)))
    return {
        "rule_log_lik": rule_ll,
        "set_width": set_w,
        "cell_width": cell_w,
        "cell_set": np.array([set_index[s] for s, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
