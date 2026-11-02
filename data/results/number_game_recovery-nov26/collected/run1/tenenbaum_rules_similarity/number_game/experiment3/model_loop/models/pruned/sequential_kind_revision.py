"""Sequential kind revision.

People infer the hidden program as a Bayesian concept learner over number rules
(parity, special sets, multiples, final digit, contained digit, exception
rules, digit sums "digits add up to s") and numeric ranges (Erlang prior on
width), with the size principle and a prior mass per kind of program. The one
claim is a sequential process: people read the examples in order and form an
impression of what *kind* of program it is from the first example; the later
examples fully decide which program within each kind is meant, but their
weight in revising the kind itself (later_kind_weight) is inferred from the
data. One recovers the ideal learner; below one, people stick with the kind
suggested by the first example; above one, they discount that first
impression. Single-example sets are unaffected; for multi-example sets the
balance between ranges and rules depends on which example came first.

Speed: interval likelihoods do not depend on parameters, so prepare_observed
precomputes, per set (all examples / first example only) and per observed
set-target cell, sums over the ranges of each width of exp(log-likelihood).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
TINY = 1e-300


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
        return 5
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
N_RULES = len(_RULES)
RULE_FAMILY = np.array([family(n) for n in _RULES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
FAMILY_ONEHOT = np.eye(7)[RULE_FAMILY]  # (rules, 7)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in _RULES.values()], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)

LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
WIDTH = HI - LO + 1
WIDTHS = np.arange(1, 101, dtype="float64")
N_PER_WIDTH = 101.0 - WIDTHS
WIDTH_ONEHOT = np.eye(100)[WIDTH - 1]  # (ranges, widths)


def interval_log_lik(examples):
    ll = np.zeros(len(LO))
    for x in examples:
        inside = (LO <= x) & (x <= HI)
        ll += np.log(ALPHA * inside / WIDTH + (1 - ALPHA) / 100)
    return ll


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # all examples, minus set max
    rule_log_lik_first = pm.Data("rule_log_lik_first", np.zeros((1, N_RULES)))  # first example only
    set_width = pm.Data("set_width", np.zeros((1, 100)))  # (sets, widths), same offset as rule_log_lik
    set_width_first = pm.Data("set_width_first", np.zeros((1, 100)))  # same offset as rule_log_lik_first
    cell_width = pm.Data("cell_width", np.zeros((1, 100)))  # (cells, widths)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the seven rule kinds and of ranges.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(8))
    # Characteristic range width of the Erlang size prior.
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    # How much the examples after the first revise the choice of kind (1 = ideal).
    later_kind_weight = pm.LogNormal("later_kind_weight", mu=0.0, sigma=0.5)

    within_prior = 1.0 / FAMILY_SIZE[RULE_FAMILY]
    rule_w = within_prior[None, :] * pt.exp(rule_log_lik)  # (sets, rules)
    rule_w1 = within_prior[None, :] * pt.exp(rule_log_lik_first)
    kind_ev = rule_w @ FAMILY_ONEHOT  # (sets, 7): within-kind marginal likelihood
    kind_ev1 = rule_w1 @ FAMILY_ONEHOT
    kind_in = (rule_w @ (RULE_MEMBERSHIP[:, :, None] * FAMILY_ONEHOT[:, None, :]).reshape(N_RULES, 700)).reshape(
        (-1, 100, 7)
    )  # (sets, targets, 7)

    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    per_range = pt.exp(erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH)))  # (widths,)
    int_ev = pt.dot(set_width, per_range)  # (sets,)
    int_ev1 = pt.dot(set_width_first, per_range)
    int_in = pt.dot(cell_width, per_range)  # (cells,)

    ev = pt.concatenate([kind_ev, int_ev[:, None]], axis=1)  # (sets, 8)
    ev1 = pt.concatenate([kind_ev1, int_ev1[:, None]], axis=1)
    log_ev = pt.log(ev + TINY)
    log_ev1 = pt.log(ev1 + TINY)
    # Kind belief: prior x first-example evidence x later-evidence ratio ** weight.
    log_kind = pt.log(family_weight)[None, :] + log_ev1 + later_kind_weight * (log_ev - log_ev1)
    kind_post = pt.exp(log_kind - pt.logsumexp(log_kind, axis=1, keepdims=True))  # (sets, 8)

    rule_in_cell = kind_in[cell_set, cell_target]  # (cells, 7)
    in_cell = pt.concatenate([rule_in_cell, int_in[:, None]], axis=1)  # (cells, 8)
    member = in_cell / (ev[cell_set] + TINY)  # within-kind posterior membership
    p_cell = (kind_post[cell_set] * member).sum(axis=1)
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    n = len(unique)
    rule_ll = np.zeros((n, N_RULES))
    rule_ll1 = np.zeros((n, N_RULES))
    set_width = np.zeros((n, 100))
    set_width1 = np.zeros((n, 100))
    range_weights = []
    for i, s in enumerate(unique):
        r_ll = RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1)
        i_ll = interval_log_lik(s)
        top = max(r_ll.max(), i_ll.max())
        rule_ll[i] = r_ll - top
        e = np.exp(i_ll - top)
        range_weights.append(e)
        set_width[i] = WIDTH_ONEHOT.T @ e
        r1 = RULE_LOG_LIK[:, s[0] - 1]
        i1 = interval_log_lik(s[:1])
        top1 = max(r1.max(), i1.max())
        rule_ll1[i] = r1 - top1
        set_width1[i] = WIDTH_ONEHOT.T @ np.exp(i1 - top1)
    cell_width = np.zeros((len(cells), 100))
    for j, (s, t) in enumerate(cells):
        e = range_weights[set_index[s]] * ((LO <= t) & (t <= HI))
        cell_width[j] = WIDTH_ONEHOT.T @ e
    return {
        "rule_log_lik": rule_ll,
        "rule_log_lik_first": rule_ll1,
        "set_width": set_width,
        "set_width_first": set_width1,
        "cell_width": cell_width,
        "cell_set": np.array([set_index[s] for s, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
