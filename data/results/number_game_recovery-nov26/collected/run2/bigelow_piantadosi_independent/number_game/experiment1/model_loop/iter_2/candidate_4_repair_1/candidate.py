"""Tempered size-principle mixture with a rule-kind prior and an Erlang interval-size prior.

Refinement of `rule_family_prior_tempered_mixture`: people do the same Bayesian
concept learning over the same 45 rules (with a prior over kinds of rule) and
all intervals of 1-100 with a tempered size principle, but they expect ranges
of a typical, intermediate width: the prior over intervals follows an Erlang
density on interval size, p(|h|) ~ |h| exp(-|h| / sigma), with a fitted scale
sigma, instead of being uniform. Single change: this size-graded interval
prior (from `learned_rule_prior_erlang_intervals`). "Yes" is the posterior
probability that the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


def rule_concepts():
    """(family, members) for each rule; families index FAMILIES."""
    primes = [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))]
    rules = [
        (0, [n for n in NUMBERS if n % 2 == 1]),
        (0, list(range(2, 101, 2))),
        (1, primes),
        (2, [n * n for n in range(1, 11)]),
        (2, [n**3 for n in range(1, 5)]),
        (3, [2, 4, 8, 16, 64]),
        (3, [2, 4, 8, 16, 32, 37, 64]),
    ]
    rules += [(4, list(range(m, 101, m))) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rules += [(3, [m**k for k in range(1, 8) if m**k <= 100]) for m in range(2, 11)]
    rules += [(5, [n for n in NUMBERS if n % 10 == d]) for d in range(10)]
    rules += [(6, [n for n in NUMBERS if str(d) in str(n)]) for d in range(10)]
    return rules


FAMILIES = ["parity", "primes", "squares_cubes", "powers", "multiples", "final_digit", "contains_digit"]
_RULES = rule_concepts()
RULE_FAMILY = np.array([f for f, _ in _RULES], dtype="int64")
N_RULES = len(_RULES)
N_FAM = len(FAMILIES)
FAMILY_LOG_COUNT = np.log(np.bincount(RULE_FAMILY, minlength=N_FAM)).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for _, c in _RULES], dtype="float64")
INTERVALS = np.array([(m, n) for m in NUMBERS for n in range(m, 101)])
INTERVAL_SIZE = (INTERVALS[:, 1] - INTERVALS[:, 0] + 1).astype("float64")
INTERVAL_MEMBERSHIP = (NUMBERS[None, :] >= INTERVALS[:, :1]) & (NUMBERS[None, :] <= INTERVALS[:, 1:])
INTERVAL_MEMBERSHIP = INTERVAL_MEMBERSHIP.astype("float64")
SIZES = np.arange(1, 101, dtype="float64")
LOG_N_PER_SIZE = np.log(101.0 - SIZES)


def _example_log_lik(membership):
    return np.log(ALPHA * membership / membership.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)


RULE_EX_LL = _example_log_lik(RULE_MEMBERSHIP)  # (rules, 100)
INTERVAL_EX_LL = _example_log_lik(INTERVAL_MEMBERSHIP)  # (intervals, 100)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, 1)))  # (sets, interval groups)
    group_log_size = pm.Data("group_log_size", np.zeros((1, 1)))  # (sets, interval groups) log count
    group_width = pm.Data("group_width", np.ones((1, 1)))  # (sets, interval groups) interval size
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_rule_in = pm.Data("cell_rule_in", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_frac = pm.Data("cell_frac", np.zeros((1, 1)))  # (cells, interval groups)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (rows,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior log-odds of each kind of rule against the intervals as a whole.
    family_logit = pm.Normal("family_logit", mu=0.0, sigma=2.5, shape=N_FAM)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)  # sampling strength
    sigma = pm.LogNormal("sigma", mu=np.log(10.0), sigma=0.75)  # typical interval width

    log_norm = pt.logsumexp(pt.concatenate([family_logit, pt.zeros(1)]))
    log_rule_prior = family_logit[RULE_FAMILY] - FAMILY_LOG_COUNT[RULE_FAMILY] - log_norm

    # Erlang prior over interval size, normalized over all 5,050 intervals.
    log_z = pt.logsumexp(pt.log(SIZES) - SIZES / sigma + LOG_N_PER_SIZE)
    log_interval_prior = -log_norm + pt.log(group_width) - group_width / sigma - log_z

    log_rules = log_rule_prior[None, :] + beta * rule_log_lik
    log_groups = log_interval_prior + group_log_size + beta * group_log_lik
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    post_cell = posterior[cell_set]
    in_concept = (post_cell[:, :N_RULES] * cell_rule_in).sum(axis=1) + (post_cell[:, N_RULES:] * cell_frac).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`.

    Intervals with the same size and the same likelihood of a set have the same
    posterior at every parameter value, so they are grouped per distinct set;
    padding groups get a log count of -1e4 (no posterior mass).
    """
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    cells = [(set_index[s], int(r["target"])) for s, r in zip(sets, rows)]
    unique_cells = list(dict.fromkeys(cells))
    cell_index = {c: i for i, c in enumerate(unique_cells)}

    rule_ll = np.stack([RULE_EX_LL[:, np.array(s) - 1].sum(axis=1) for s in unique])
    groupings = []
    for s in unique:
        ll = INTERVAL_EX_LL[:, np.array(s) - 1].sum(axis=1)
        key = np.stack([np.round(ll, 9), INTERVAL_SIZE], axis=1)
        keys, inverse = np.unique(key, axis=0, return_inverse=True)
        groupings.append((inverse.reshape(-1), keys))
    n_groups = max(len(k) for _, k in groupings)
    log_lik = np.zeros((len(unique), n_groups))
    log_size = np.full((len(unique), n_groups), -1e4)
    width = np.ones((len(unique), n_groups))
    sizes = np.ones((len(unique), n_groups))
    for i, (inverse, keys) in enumerate(groupings):
        counts = np.bincount(inverse, minlength=len(keys))
        log_lik[i, : len(keys)] = keys[:, 0]
        width[i, : len(keys)] = keys[:, 1]
        log_size[i, : len(keys)], sizes[i, : len(keys)] = np.log(counts), counts
    frac = np.zeros((len(unique_cells), n_groups))
    rule_in = np.zeros((len(unique_cells), N_RULES))
    for j, (i, t) in enumerate(unique_cells):
        inverse, keys = groupings[i]
        frac[j, : len(keys)] = np.bincount(inverse, weights=INTERVAL_MEMBERSHIP[:, t - 1], minlength=len(keys))
        frac[j] /= sizes[i]
        rule_in[j] = RULE_MEMBERSHIP[:, t - 1]
    return {
        "rule_log_lik": rule_ll,
        "group_log_lik": log_lik,
        "group_log_size": log_size,
        "group_width": width,
        "cell_set": np.array([c[0] for c in unique_cells], dtype="int64"),
        "cell_rule_in": rule_in,
        "cell_frac": frac,
        "row_cell": np.array([cell_index[c] for c in cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
