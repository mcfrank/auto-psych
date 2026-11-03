"""Bayesian concept learning with serial-position-weighted evidence.

People do Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11
and 12, ending in 0-9, containing the digit 0-9) and every interval of 1-100,
with the size principle and 10% noisy examples (as in
bigelow_piantadosi_independent). The one distortion: the examples, seen in
order, are not weighted equally. The log-likelihood of the k-th of n examples
is weighted by w_k proportional to exp(gamma * (k - n)), normalised so the
weights sum to n (total evidence unchanged); gamma > 0 is recency, gamma < 0
primacy. "Yes" is the posterior probability that the concept contains the target.
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
RULE_EX_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
INT_LO = np.array([m for m in NUMBERS for n in range(m, 101)])
INT_HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INT_SIZE = INT_HI - INT_LO + 1
L_OUT = np.log((1 - ALPHA) / 100)
L_IN = np.log(ALPHA / NUMBERS + (1 - ALPHA) / 100)  # by interval size 1..100
# Intervals contain a contiguous run (in sorted order) of the examples, so for
# each set they are grouped by (run, size): within a group the likelihood is
# the same for every interval. 10 runs of up to 4 examples plus the empty run.
RUNS = [(a, b) for a in range(MAX_N) for b in range(a, MAX_N)] + [None]
N_G = len(RUNS)
N_Z = 100

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_pos_ll = pm.Data("rule_pos_ll", np.zeros((1, MAX_N, N_RULES)))  # (sets, pos, rules)
    # Each set's non-empty (run, size) groups, padded to a common width.
    col_pos = pm.Data("col_pos", np.zeros((1, 1, MAX_N)))  # presented example k inside the group's run
    col_gain = pm.Data("col_gain", np.zeros((1, 1)))  # L_in(size) - L_out
    col_log_count = pm.Data("col_log_count", np.zeros((1, 1)))  # log #intervals (-1e10 for padding)
    pos_lag = pm.Data("pos_lag", np.zeros((1, MAX_N)))  # k - n (<= 0) for shown positions
    pos_mask = pm.Data("pos_mask", np.ones((1, MAX_N)))  # 1 for shown positions
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_group_frac = pm.Data("cell_group_frac", np.zeros((1, 1)))  # share of group containing target
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    gamma = pm.Normal("gamma", 0.0, 1.0)  # order bias: > 0 recency, < 0 primacy

    raw = pt.exp(gamma * pos_lag) * pos_mask
    n_shown = pos_mask.sum(axis=1, keepdims=True)
    w = raw * n_shown / raw.sum(axis=1, keepdims=True)  # (sets, pos), sums to n

    rule_ll = (w[:, :, None] * rule_pos_ll).sum(axis=1)  # (sets, rules)
    w_in = (col_pos * w[:, None, :]).sum(axis=2)  # (sets, cols): weighted examples inside
    group_ll = w_in * col_gain + n_shown * L_OUT
    log_ints = pt.log1p(-lambda_0) - 2 * np.log(100) + group_ll + col_log_count

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_ll
    log_joint = pt.concatenate([log_rules, log_ints], axis=1)
    post = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cpost = post[cell_set]
    in_concept = (cpost[:, :N_RULES] * cell_rule_member).sum(axis=1) + (cpost[:, N_RULES:] * cell_group_frac).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    sidx = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_pos_ll = np.zeros((S, MAX_N, N_RULES))
    group_pos = np.zeros((S, N_G, MAX_N))
    counts = np.zeros((S, N_G, N_Z))
    int_group = np.zeros((S, len(INT_LO)), dtype=int)
    pos_lag = np.zeros((S, MAX_N))
    pos_mask = np.zeros((S, MAX_N))
    for i, s in enumerate(unique):
        n = len(s)
        order = sorted(range(n), key=lambda k: s[k])  # presented positions in sorted order
        vals = np.array([s[k] for k in order])
        for k, x in enumerate(s):
            rule_pos_ll[i, k] = RULE_EX_LOG_LIK[:, x - 1]
            pos_lag[i, k] = k - (n - 1)
            pos_mask[i, k] = 1.0
        for g, run in enumerate(RUNS):
            if run is not None and run[1] < n:
                for r in range(run[0], run[1] + 1):
                    group_pos[i, g, order[r]] = 1.0
        inside = (INT_LO[:, None] <= vals[None, :]) & (vals[None, :] <= INT_HI[:, None])  # (intervals, n)
        for j in range(len(INT_LO)):
            idx = np.flatnonzero(inside[j])
            g = RUNS.index((idx[0], idx[-1])) if len(idx) else N_G - 1
            int_group[i, j] = g
        np.add.at(counts[i], (int_group[i], INT_SIZE - 1), 1.0)
    cells = [(sidx[s], int(r["target"])) for s, r in zip(sets, rows)]
    ucells = list(dict.fromkeys(cells))
    cidx = {c: i for i, c in enumerate(ucells)}
    targets = np.array([t for _, t in ucells])
    frac = np.zeros((len(ucells), N_G, N_Z))
    for c, (si, t) in enumerate(ucells):
        has = (INT_LO <= t) & (t <= INT_HI)
        np.add.at(frac[c], (int_group[si][has], INT_SIZE[has] - 1), 1.0)
        frac[c] = np.divide(frac[c], counts[si], out=np.zeros((N_G, N_Z)), where=counts[si] > 0)
    nz = [np.flatnonzero(counts[i].reshape(-1) > 0) for i in range(S)]
    width = max(len(c) for c in nz)
    col_pos = np.zeros((S, width, MAX_N))
    col_gain = np.zeros((S, width))
    col_log_count = np.full((S, width), -1e10)
    col_frac = np.zeros((len(ucells), width))
    for i, c in enumerate(nz):
        g, z = np.divmod(c, N_Z)
        col_pos[i, :len(c)] = group_pos[i, g]
        col_gain[i, :len(c)] = L_IN[z] - L_OUT
        col_log_count[i, :len(c)] = np.log(counts[i].reshape(-1)[c])
    for k, (si, _) in enumerate(ucells):
        col_frac[k, :len(nz[si])] = frac[k].reshape(-1)[nz[si]]
    return {
        "rule_pos_ll": rule_pos_ll,
        "col_pos": col_pos,
        "col_gain": col_gain,
        "col_log_count": col_log_count,
        "pos_lag": pos_lag,
        "pos_mask": pos_mask,
        "cell_set": np.array([s for s, _ in ucells], dtype="int64"),
        "cell_group_frac": col_frac,
        "cell_rule_member": RULE_MEMBERSHIP[:, targets - 1].T.copy(),
        "row_cell": np.array([cidx[c] for c in cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
