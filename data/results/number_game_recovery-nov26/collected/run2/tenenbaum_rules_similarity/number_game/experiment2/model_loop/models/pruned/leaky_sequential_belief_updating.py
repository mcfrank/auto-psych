"""Leaky sequential belief updating. People do the same Bayesian concept
learning as extended_multiples_concepts (45 independently weighted rules plus
multiples of 13-50, intervals of 1..100 with an Erlang width prior, an inferred
share of random examples, examples drawn without replacement), but they update
one running belief example by example and the belief is leaky: before each new
example is taken in, the current belief is flattened (raised to a power
`retention` in (0, 1]). After n examples the log-prior and the first example
are thus weighted retention^(n-1), example i by retention^(n-i), and the last
example in full. retention = 1 recovers the incumbent; one-example sets are
unaffected. Longer sets lean less on the prior and more on the final examples.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_EX = 4


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
N_BASE = len(_RULES)
EXT_M = list(range(13, 51))
N_EXT = len(EXT_M)
_RULES.update({f"multiples of {m}": list(range(m, 101, m)) for m in EXT_M})
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)

# Intervals grouped by (size, pattern of which serial positions fall inside).
N_PAT = 2**MAX_EX
N_GROUPS = 100 * N_PAT
SIZES_F = np.arange(1, 101).astype("float64")
N_PER_SIZE = 101.0 - SIZES_F
PAT_BIT = np.array([[(p >> i) & 1 for i in range(MAX_EX)] for p in range(N_PAT)], dtype="float64")
PAT_J = np.array([[bin(p & ((1 << i) - 1)).count("1") for i in range(MAX_EX)] for p in range(N_PAT)])
# One-hot of the without-replacement draw index j for each (pattern, position).
PAT_INJ = np.zeros((N_PAT, MAX_EX, MAX_EX))
for _p in range(N_PAT):
    for _i in range(MAX_EX):
        if PAT_BIT[_p, _i]:
            PAT_INJ[_p, _i, PAT_J[_p, _i]] = 1.0

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_BASE - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0
POS = np.arange(MAX_EX, dtype="float64")
J_SHIFT = np.arange(MAX_EX, dtype="float64")

with pm.Model() as model:
    pos_mask = pm.Data("pos_mask", np.ones((1, MAX_EX)))  # (sets, positions)
    rule_inj = pm.Data("rule_inj", np.zeros((1, N_RULES, MAX_EX, MAX_EX)))  # (sets, rules, pos, j)
    rule_in = pm.Data("rule_in", np.zeros((1, N_RULES, MAX_EX)))  # (sets, rules, pos)
    # Each set keeps only the (size, pattern) groups that occur for it, padded to K.
    group_idx = pm.Data("group_idx", np.zeros((1, 1), dtype="int64"))  # (sets, K)
    group_log_count = pm.Data("group_log_count", np.zeros((1, 1)))  # (sets, K)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_group_frac = pm.Data("cell_group_frac", np.zeros((1, 1)))  # (cells, K)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    noise = pm.Beta("noise", 1.5, 6.0)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    ext_share = pm.Beta("ext_share", 1.0, 4.0)
    all_weights = pt.concatenate([
        (1 - ext_share) * rule_weights, pt.ones(N_EXT) * ext_share / N_EXT,
    ])

    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    log_erlang = pt.log(SIZES_F) - SIZES_F / sigma
    log_per_interval = log_erlang - pt.logsumexp(log_erlang + np.log(N_PER_SIZE))

    # Leaky updating: belief is flattened by `retention` before each new example.
    retention = pm.Beta("retention", 4.0, 1.5)
    n_ex = pos_mask.sum(axis=1, keepdims=True)
    lags = pt.maximum(n_ex - 1.0 - POS[None, :], 0.0)
    w = pt.exp(lags * pt.log(retention)) * pos_mask  # (sets, positions)
    prior_w = pt.exp((n_ex - 1.0) * pt.log(retention))  # (sets, 1)
    n_w = w.sum(axis=1, keepdims=True)

    log_out = pt.log(noise / 100.0)
    # Rules: in-concept term for draw j is one of size - j unseen members.
    rule_term = pt.log((1 - noise) / np.maximum(RULE_SIZE[:, None] - J_SHIFT[None, :], 1.0) + noise / 100.0)
    in_ll = (rule_inj * rule_term[None, :, None, :]).sum(axis=3)  # (sets, rules, pos)
    out_ll = (pos_mask[:, None, :] - rule_in) * log_out
    rule_log_lik = ((in_ll + out_ll) * w[:, None, :]).sum(axis=2)

    # Intervals: (size, j) term table, then per (size, pattern, position).
    group_term = pt.log((1 - noise) / np.maximum(SIZES_F[:, None] - J_SHIFT[None, :], 1.0) + noise / 100.0)
    pat_in = (PAT_INJ[None, :, :, :] * group_term[:, None, None, :]).sum(axis=3)  # (sizes, pat, pos)
    g_in = pt.tensordot(w, pat_in.reshape((100 * N_PAT, MAX_EX)), axes=[[1], [1]])  # (sets, groups)
    w_in_pat = pt.dot(w, PAT_BIT.T)  # (sets, pat): weight of in-concept positions
    g_out = (n_w - pt.tile(w_in_pat, (1, 100))) * log_out
    group_score = g_in + g_out + prior_w * pt.repeat(pt.log1p(-lambda_0) + log_per_interval, N_PAT)[None, :]  # (sets, groups)
    log_groups = pt.take_along_axis(group_score, group_idx, axis=1) + group_log_count
    log_rules = prior_w * (pt.log(lambda_0) + pt.log(all_weights))[None, :] + rule_log_lik
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_post = posterior[cell_set]
    in_concept = (cell_post[:, :N_RULES] * cell_rule_member).sum(axis=1) + (
        cell_post[:, N_RULES:] * cell_group_frac
    ).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` (in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    mask = np.zeros((S, MAX_EX))
    r_in = np.zeros((S, N_RULES, MAX_EX))
    r_inj = np.zeros((S, N_RULES, MAX_EX, MAX_EX))
    counts = np.zeros((S, N_GROUPS))
    keys = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {k: i for i, k in enumerate(cells)}
    targets_by_set = {}
    for ci, (si, t) in enumerate(cells):
        targets_by_set.setdefault(si, []).append((ci, t))
    contain = np.zeros((len(cells), N_GROUPS))
    for i, s in enumerate(unique):
        mask[i, :len(s)] = 1.0
        earlier = np.zeros(N_RULES)
        for pos, x in enumerate(s):
            mem = RULE_MEMBERSHIP[:, x - 1]
            r_in[i, :, pos] = mem
            for r in np.nonzero(mem)[0]:
                r_inj[i, r, pos, int(earlier[r])] = 1.0
            earlier += mem
        tl = targets_by_set.get(i, [])
        for m in range(1, 101):
            for n in range(m, 101):
                p = sum(1 << k for k, x in enumerate(s) if m <= x <= n)
                g = (n - m) * N_PAT + p
                counts[i, g] += 1
                for ci, t in tl:
                    if m - 1 <= t <= n - 1:
                        contain[ci, g] += 1
    cs = np.array([k[0] for k in cells], dtype="int64")
    ct = np.array([k[1] for k in cells], dtype="int64")
    frac_full = np.where(counts[cs] > 0, contain / np.maximum(counts[cs], 1), 0.0)
    nz = [np.nonzero(counts[i])[0] for i in range(S)]
    K = max(len(z) for z in nz)
    gidx = np.zeros((S, K), dtype="int64")
    log_count = np.full((S, K), -1e3)
    for i, z in enumerate(nz):
        gidx[i, :len(z)] = z
        log_count[i, :len(z)] = np.log(counts[i, z])
    frac = np.take_along_axis(frac_full, gidx[cs], axis=1) * (log_count[cs] > -1e3)
    return {
        "pos_mask": mask,
        "rule_inj": r_inj,
        "rule_in": r_in,
        "group_idx": gidx,
        "group_log_count": log_count,
        "cell_set": cs,
        "cell_rule_member": RULE_MEMBERSHIP[:, ct].T.copy(),
        "cell_group_frac": frac,
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
