"""Weak-sampling concept learning: people treat the examples only as numbers the
program is allowed to produce, not as samples drawn from its concept. A rule or
magnitude range gains plausibility by containing the examples (each missed
example costs a factor eps, a possible stray), but not for being small: there is
no size principle, so a concept's plausibility rests on its prior (one weight per
kind of rule, shared by its rules; a range's prior falls with its width as width^-beta) and does not
tighten as more consistent examples arrive. "Yes" is the posterior probability
that the concept contains the target.

p(X | h) = (1 - eps)^k eps^(n - k) with k of the n examples in h, independent of
the concept's size, so intervals only need per-set (k, size) counts.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

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
INT_LO = np.array([m for m in NUMBERS for n in range(m, 101)])
INT_HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INT_SIZE_IDX = INT_HI - INT_LO  # size - 1

FAMILIES = ["parity", "primes", "squares and cubes", "powers", "multiples", "ends in", "contains digit"]


def family_of(name):
    if name in ("odd", "even"):
        return "parity"
    if name in ("squares", "cubes"):
        return "squares and cubes"
    for f in ("primes", "powers", "multiples", "ends in", "contains digit"):
        if name.startswith(f):
            return f
    raise ValueError(name)


RULE_FAMILY = np.array([FAMILIES.index(family_of(r)) for r in RULE_NAMES])
LOG_SIZE = np.log(np.arange(1, 101, dtype="float64"))
N_OF_SIZE = 101.0 - np.arange(1, 101, dtype="float64")  # intervals of each size
K_GRID = np.arange(MAX_N + 1, dtype="float64")

with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES), dtype="int64"))  # (sets, rules)
    set_n = pm.Data("set_n", np.ones(1, dtype="float64"))  # (sets,)
    interval_count = pm.Data("interval_count", np.zeros((1, MAX_N + 1, 100)))  # (sets, k, size)
    cell_count = pm.Data("cell_count", np.zeros((1, MAX_N + 1, 100)))  # (cells, k, size)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    # Rules are plausible by kind only: one weight per family (parity fixed at 0),
    # shared equally by the family's rules.
    family_free = pm.Normal("family_free", mu=0.0, sigma=1.5, shape=len(FAMILIES) - 1)
    family_level = pt.concatenate([pt.zeros(1), family_free])
    rule_weights = pm.Deterministic("rule_weights", pt.special.softmax(family_level[RULE_FAMILY]))
    # Chance an example is a stray (lies outside the concept).
    eps_logit = pm.Normal("eps_logit", mu=-2.5, sigma=1.5)
    eps = pm.Deterministic("eps", pm.math.sigmoid(eps_logit))
    # Prior preference for narrow ranges: interval prior proportional to width^-beta.
    beta = pm.HalfNormal("beta", sigma=2.0)

    log_eps, log_1m = pt.log(eps), pt.log1p(-eps)
    rule_kf = pt.cast(rule_k, "float64")
    rule_ll = rule_kf * log_1m + (set_n[:, None] - rule_kf) * log_eps  # (sets, rules)
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_ll

    size_logw = -beta * LOG_SIZE
    log_z_int = pt.logsumexp(size_logw + np.log(N_OF_SIZE))
    size_w = pt.exp(size_logw - log_z_int)  # (size,) prior per interval of that size
    # (sets, k) weak-sampling likelihood, rescaled by the max (k = n) for stability.
    k_lik = pt.exp((K_GRID[None, :] - set_n[:, None]) * (log_1m - log_eps))
    k_lik = pt.switch(K_GRID[None, :] <= set_n[:, None], k_lik, 0.0)
    log_scale = set_n * log_1m
    # Contract the size axis first with the shared width prior: (sets, k) and (cells, k).
    set_k_mass = pt.dot(interval_count, size_w)
    cell_k_mass = pt.dot(cell_count, size_w)
    int_ev = (set_k_mass * k_lik).sum(axis=1)
    log_intervals = pt.log1p(-lambda_0) + pt.log(int_ev) + log_scale

    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    interval_post = pt.exp(log_intervals - log_norm)

    cell_int_share = (cell_k_mass * k_lik[cell_set]).sum(axis=1) / int_ev[cell_set]
    in_concept = (rule_post[cell_set] * cell_rule_member).sum(axis=1) + interval_post[cell_set] * cell_int_share
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-6, 1 - 1e-6)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_k = np.zeros((S, N_RULES), dtype="int64")
    set_n = np.zeros(S, dtype="float64")
    int_k = np.zeros((S, len(INT_LO)), dtype="int64")
    counts = np.zeros((S, MAX_N + 1, 100))
    for i, s in enumerate(unique):
        ex = np.array(s)
        set_n[i] = len(ex)
        rule_k[i] = RULE_MEMBERSHIP[:, ex - 1].sum(axis=1).astype("int64")
        int_k[i] = ((ex[None, :] >= INT_LO[:, None]) & (ex[None, :] <= INT_HI[:, None])).sum(axis=1)
        np.add.at(counts[i], (int_k[i], INT_SIZE_IDX), 1.0)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_count = np.zeros((len(cells), MAX_N + 1, 100))
    for j, (i, t) in enumerate(cells):
        contains = (INT_LO <= t) & (t <= INT_HI)
        np.add.at(cell_count[j], (int_k[i][contains], INT_SIZE_IDX[contains]), 1.0)
    return {
        "rule_k": rule_k,
        "set_n": set_n,
        "interval_count": counts,
        "cell_count": cell_count,
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
