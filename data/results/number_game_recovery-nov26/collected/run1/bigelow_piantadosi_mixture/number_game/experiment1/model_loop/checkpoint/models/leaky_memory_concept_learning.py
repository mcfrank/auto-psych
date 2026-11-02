"""Leaky-memory Bayesian concept learning.

People do Bayesian concept learning over 45 rules and all intervals of 1-100
(size principle, 10% of examples uniform noise; rule weights as in the
Bigelow & Piantadosi Independent model), but they read the examples in order
and earlier examples fade from memory: an example's log-likelihood is
discounted by a forgetting factor delta for every example shown after it
(weight delta^(n - 1 - k) for example k of n), so the last example counts
fully and the first of a long set counts least. "Yes" is the posterior
probability that the concept contains the target.

Intervals are grouped, per example set, by (size, which examples they
contain): their weighted likelihood depends only on that, so the posterior
over the 5,050 intervals is computed exactly from those groups.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
MAX_N = 4


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
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
INTERVALS = [(m, n) for m in NUMBERS for n in range(m, 101)]
INTERVAL_LO = np.array([m for m, _ in INTERVALS])
INTERVAL_HI = np.array([n for _, n in INTERVALS])
LOG_OUT = np.log((1 - ALPHA) / 100)

# "ends in 0" and "contains digit 0" are the same set; sample their total and a split.
END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_pos_ll = pm.Data("rule_pos_ll", np.zeros((1, MAX_N, N_RULES)))  # (sets, position, rules)
    lag = pm.Data("lag", np.zeros((1, MAX_N)))  # examples shown after this one
    pos_mask = pm.Data("pos_mask", np.zeros((1, MAX_N)))  # 1 for real positions
    grp_incl = pm.Data("grp_incl", np.zeros((1, 1, MAX_N)))  # (sets, groups, position): example in group's intervals
    grp_log_in = pm.Data("grp_log_in", np.zeros((1, 1)))  # log P(x | interval) for x inside, by group size
    grp_log_count = pm.Data("grp_log_count", np.zeros((1, 1)))  # log #intervals in group (-1e9 for padding)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_grp_frac = pm.Data("cell_grp_frac", np.zeros((1, 1)))  # share of group's intervals containing target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    delta = pm.Beta("delta", 2.0, 1.0)  # memory retention per subsequent example
    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    w = pos_mask * delta ** lag  # (sets, position)
    rule_ll = (w[:, :, None] * rule_pos_ll).sum(axis=1)  # (sets, rules)
    w_in = (grp_incl * w[:, None, :]).sum(axis=2)  # (sets, groups)
    w_tot = w.sum(axis=1, keepdims=True)
    grp_ll = w_in * grp_log_in + (w_tot - w_in) * LOG_OUT + grp_log_count

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_ll
    log_grps = pt.log1p(-lambda_0) - 2 * np.log(100) + grp_ll
    log_norm = pt.logaddexp(pt.logsumexp(log_rules, axis=1), pt.logsumexp(log_grps, axis=1))  # (sets,)
    post_rules = pt.exp(log_rules - log_norm[:, None])
    post_grps = pt.exp(log_grps - log_norm[:, None])
    in_concept = (post_rules[cell_set] * cell_rule_member).sum(axis=1) + (post_grps[cell_set] * cell_grp_frac).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-9, 1.0 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` (in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_pos_ll = np.zeros((S, MAX_N, N_RULES))
    lag = np.zeros((S, MAX_N))
    pos_mask = np.zeros((S, MAX_N))
    groups = []
    for i, s in enumerate(unique):
        n = len(s)
        for k, x in enumerate(s):
            rule_pos_ll[i, k] = RULE_LOG_LIK[:, x - 1]
            lag[i, k] = n - 1 - k
            pos_mask[i, k] = 1.0
        ex = np.array(s)
        incl = (ex[None, :] >= INTERVAL_LO[:, None]) & (ex[None, :] <= INTERVAL_HI[:, None])  # (intervals, n)
        sizes = INTERVAL_HI - INTERVAL_LO + 1
        keys = {}
        assign = np.empty(len(INTERVALS), dtype="int64")
        for j in range(len(INTERVALS)):
            key = (int(sizes[j]),) + tuple(bool(b) for b in incl[j])
            assign[j] = keys.setdefault(key, len(keys))
        groups.append((keys, assign))
    G = max(len(k) for k, _ in groups)
    grp_incl = np.zeros((S, G, MAX_N))
    grp_log_in = np.zeros((S, G))
    grp_log_count = np.full((S, G), -1e9)
    for i, (keys, assign) in enumerate(groups):
        counts = np.bincount(assign, minlength=len(keys))
        for key, g in keys.items():
            grp_incl[i, g, :len(key) - 1] = key[1:]
            grp_log_in[i, g] = np.log(ALPHA / key[0] + (1 - ALPHA) / 100)
            grp_log_count[i, g] = np.log(counts[g])
    cells = [(set_index[s], int(r["target"])) for s, r in zip(sets, rows)]
    unique_cells = list(dict.fromkeys(cells))
    cell_index = {c: i for i, c in enumerate(unique_cells)}
    C = len(unique_cells)
    cell_grp_frac = np.zeros((C, G))
    for c, (i, t) in enumerate(unique_cells):
        keys, assign = groups[i]
        contains = (INTERVAL_LO <= t) & (INTERVAL_HI >= t)
        counts = np.bincount(assign, minlength=len(keys))
        hit = np.bincount(assign, weights=contains.astype(float), minlength=len(keys))
        cell_grp_frac[c, :len(keys)] = hit / counts
    return {
        "rule_pos_ll": rule_pos_ll,
        "lag": lag,
        "pos_mask": pos_mask,
        "grp_incl": grp_incl,
        "grp_log_in": grp_log_in,
        "grp_log_count": grp_log_count,
        "cell_set": np.array([i for i, _ in unique_cells], dtype="int64"),
        "cell_rule_member": np.array([RULE_MEMBERSHIP[:, t - 1] for _, t in unique_cells]),
        "cell_grp_frac": cell_grp_frac,
        "row_cell": np.array([cell_index[c] for c in cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
