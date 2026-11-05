"""Recency-weighted example evidence in Bayesian concept learning.

People do Bayesian concept learning over mathematical rules and magnitude
intervals (size principle), but they read the examples one at a time and
memory for earlier examples fades: each example's evidential force is
discounted by how many examples were read after it. Earlier examples therefore
constrain the inferred concept less than the most recent one.

Hypotheses: 44 rules (prior lambda * w_r, w ~ Dirichlet) and every interval
[m, n] of 1..100 (prior (1 - lambda) / 5050). Each example x_j contributes
log P(x_j | h) = log(0.9 * [x_j in h] / |h| + 0.1 / 100), scaled by the memory
weight exp(-decay * (#examples shown after x_j)). P(yes) = posterior
probability the concept contains the target.

For speed (exact, not an approximation), per example set the 5050 intervals are
grouped by (which examples they contain, size; size is irrelevant
when they contain none): intervals in a group share one
likelihood, so the group enters with log(count) and, for each target, the
fraction of its intervals that contain the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
LOG_NOISE = np.log((1 - ALPHA) / 100)
NUMBERS = np.arange(1, 101)
MAX_EX = 4


def rule_concepts():
    rules = {
        "any number": list(NUMBERS),
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "even": list(range(2, 101, 2)),
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
        "powers of 2, not 32": [2, 4, 8, 16, 64],
        "powers of 2, or 37": [2, 4, 8, 16, 32, 37, 64],
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(1, 10)})
    return rules


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
RULE_MEM = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")  # (R, 100)
RULE_LOGIN = np.log(ALPHA / RULE_MEM.sum(axis=1) + (1 - ALPHA) / 100)  # (R,)
_LO, _HI = np.triu_indices(100)  # 0-based interval bounds
N_INT = len(_LO)


def interval_groups(examples):
    """Group intervals by (mask of examples contained, size)."""
    groups = {}
    for lo, hi in zip(_LO, _HI):
        mask = tuple(int(lo <= x - 1 <= hi) for x in examples)
        # Intervals containing no example share one likelihood whatever their size.
        key = (mask, hi - lo + 1 if any(mask) else 0)
        groups.setdefault(key, []).append((lo, hi))
    return groups


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    sidx = {s: i for i, s in enumerate(unique)}
    cells = [(sidx[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    ucells = list(dict.fromkeys(cells))
    S = len(unique)
    allg = [interval_groups(s) for s in unique]
    G = max(len(g) for g in allg)

    lag = np.zeros((S, MAX_EX))
    present = np.zeros((S, MAX_EX))
    rule_in = np.zeros((S, MAX_EX, N_RULES))  # example j in rule r
    g_in = np.zeros((S, MAX_EX, G))  # example j in group g
    g_login = np.zeros((S, G))  # log(alpha / size + noise)
    g_logcount = np.full((S, G), -1e4)  # padding groups get ~zero weight
    g_keys = []
    for i, (s, groups) in enumerate(zip(unique, allg)):
        k = len(s)
        for j, x in enumerate(s):
            lag[i, j] = k - 1 - j
            present[i, j] = 1.0
            rule_in[i, j] = RULE_MEM[:, x - 1]
        keys = list(groups)
        g_keys.append(keys)
        for g, (mask, size) in enumerate(keys):
            g_in[i, :k, g] = mask
            g_login[i, g] = np.log(ALPHA / size + (1 - ALPHA) / 100) if size else LOG_NOISE
            g_logcount[i, g] = np.log(len(groups[(mask, size)]))
    # Per set, its distinct targets (padded to T): rule membership and the
    # fraction of each interval group's intervals that contain the target.
    per_set = [[c for c in ucells if c[0] == i] for i in range(S)]
    T = max(len(p) for p in per_set)
    cell_rule = np.zeros((S, N_RULES, T))
    cell_cover = np.zeros((S, G, T))
    cpos = {}
    for i, cs in enumerate(per_set):
        for k, (_, t) in enumerate(cs):
            cpos[(i, t)] = i * T + k
            cell_rule[i, :, k] = RULE_MEM[:, t]
            for g, key in enumerate(g_keys[i]):
                ivs = allg[i][key]
                cell_cover[i, g, k] = np.mean([lo <= t <= hi for lo, hi in ivs])
    return {
        "ex_lag": lag,
        "ex_present": present,
        "rule_in": rule_in,
        "grp_in": g_in,
        "grp_login": g_login,
        "grp_logcount": g_logcount,
        "cell_rule": cell_rule,
        "cell_cover": cell_cover,
        "row_cell": np.array([cpos[c] for c in cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    ex_lag = pm.Data("ex_lag", np.zeros((1, MAX_EX)))
    ex_present = pm.Data("ex_present", np.zeros((1, MAX_EX)))
    rule_in = pm.Data("rule_in", np.zeros((1, MAX_EX, N_RULES)))
    grp_in = pm.Data("grp_in", np.zeros((1, MAX_EX, 1)))
    grp_login = pm.Data("grp_login", np.zeros((1, 1)))
    grp_logcount = pm.Data("grp_logcount", np.zeros((1, 1)))
    cell_rule = pm.Data("cell_rule", np.zeros((1, N_RULES, 1)))
    cell_cover = pm.Data("cell_cover", np.zeros((1, 1, 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    lam = pm.Beta("lambda_rules", 2, 2)
    decay = pm.HalfNormal("decay", sigma=1.0)  # memory loss per later example

    mem_w = pt.exp(-decay * ex_lag) * ex_present  # (S, K)
    w_tot = mem_w.sum(axis=1, keepdims=True)  # (S, 1)
    # Weighted log-likelihood: sum_j w_j [in_j log(in-lik) + (1 - in_j) log(noise)]
    w_in_rule = (mem_w[:, :, None] * rule_in).sum(axis=1)  # (S, R)
    ll_rule = w_in_rule * (RULE_LOGIN - LOG_NOISE)[None, :] + w_tot * LOG_NOISE
    w_in_grp = (mem_w[:, :, None] * grp_in).sum(axis=1)  # (S, G)
    ll_grp = w_in_grp * (grp_login - LOG_NOISE) + w_tot * LOG_NOISE

    lj_rule = pt.log(lam) + pt.log(rule_weights)[None, :] + ll_rule
    lj_grp = pt.log1p(-lam) - np.log(N_INT) + grp_logcount + ll_grp
    norm = pt.logsumexp(pt.concatenate([lj_rule, lj_grp], axis=1), axis=1, keepdims=True)
    post_rule = pt.exp(lj_rule - norm)
    post_grp = pt.exp(lj_grp - norm)
    in_concept = pt.batched_dot(post_rule, cell_rule) + pt.batched_dot(post_grp, cell_cover)  # (S, T)
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
