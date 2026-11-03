"""Centre-weighted sampling from magnitude ranges.

People do Bayesian concept learning over 45 familiar rules (one inferred
plausibility per kind of rule, size principle, an uncertain Beta belief over
the program's reliability, strays partly "familiar-rule numbers") and over
every magnitude range [lo, hi] of 1-100 (uniform prior over ranges, sharp
membership). The one change is how people think a range program produces its
numbers: not evenly, but more often near the middle of the range than near its
edges -- a bell over the range, centred on its midpoint with spread
(width / sqrt(12)), truncated to the range. No new free parameter.

p(X | h) = E_a prod_i (a g_h(x_i) + (1 - a) q(x_i))
         = Q(X) sum_J c_|J| prod_{i in J} g_h(x_i) / q(x_i)
(rules: g = 1[x in h] / |h|). The products of g over each subset J of a set's
examples are parameter-free, so they are pre-summed over all ranges (and over
the ranges containing each target) in prepare_observed.
"""

import math

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
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
# One range [lo, hi] per pair: sharp membership, centre-weighted generation.
_LO = np.array([m for m in NUMBERS for n in range(m, 101)], dtype="float64")
_HI = np.array([n for m in NUMBERS for n in range(m, 101)], dtype="float64")
CL_MU = (_LO + _HI) / 2
CL_SIG = (_HI - _LO + 1) / np.sqrt(12.0)
N_CL = len(CL_MU)
_KERN = np.exp(-((NUMBERS[None, :] - CL_MU[:, None]) ** 2) / (2 * CL_SIG[:, None] ** 2))  # (clusters, 100)
CL_TYPICALITY = ((NUMBERS[None, :] >= _LO[:, None]) & (NUMBERS[None, :] <= _HI[:, None])).astype(float)  # membership
CL_PROB = _KERN * CL_TYPICALITY / (_KERN * CL_TYPICALITY).sum(axis=1, keepdims=True)  # generation g_h(x)
STRAY_RULE_GEN = (RULE_MEMBERSHIP / RULE_SIZE[:, None]).mean(axis=0)  # (100,): equal rules, then a member

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

SUBSET_IN = np.array([[(m >> i) & 1 for i in range(MAX_N)] for m in range(2**MAX_N)], dtype="float64")
RULE_SIZE_IDX = RULE_SIZE.astype("int64") - 1

_N = np.arange(1, MAX_N + 1, dtype="float64")[:, None]
_J = np.arange(MAX_N + 1, dtype="float64")[None, :]
_VALID = _J <= _N
_N_MINUS_J = np.maximum(_N - _J, 0.0)
_SIZES = np.arange(1, 101, dtype="float64")
_SIZE_POW = _SIZES[:, None] ** -np.arange(MAX_N + 1)[None, :]  # (size, j)


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


def _elementary(weights):
    """Elementary symmetric polynomials e_0..e_4 over the last axis (4 examples)."""
    e = [pt.ones_like(weights[..., 0])] + [pt.zeros_like(weights[..., 0])] * MAX_N
    for i in range(MAX_N):
        w = weights[..., i]
        e = [e[0]] + [e[j] + w * e[j - 1] for j in range(1, MAX_N + 1)]
    return pt.stack(e, axis=-1)


with pm.Model() as model:
    example_idx = pm.Data("example_idx", np.zeros((1, MAX_N), dtype="int64"))  # (sets, 4): x - 1, sorted
    example_mask = pm.Data("example_mask", np.zeros((1, MAX_N)))  # (sets, 4): 1 if a real example
    rule_sub = pm.Data("rule_sub", np.zeros((1, N_RULES), dtype="int64"))  # (sets, rules): subset of examples in rule
    set_n_idx = pm.Data("set_n_idx", np.zeros(1, dtype="int64"))  # (sets,): n - 1
    sub_g = pm.Data("sub_g", np.zeros((1, 2**MAX_N)))  # (sets, subsets): sum_h prod_{i in J} g_h(x_i)
    cell_sub_g = pm.Data("cell_sub_g", np.zeros((1, 2**MAX_N)))  # same, weighted by the target's typicality
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    # Family-only rule prior: every rule of a family equally plausible
    # (sum-to-zero levels, so the softmax has no unidentified shift).
    family_level = pm.ZeroSumNormal("family_level", sigma=1.5, shape=len(FAMILIES))
    rule_weights = pm.Deterministic("rule_weights", pt.special.softmax(family_level[RULE_FAMILY]))
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa

    # Stray distribution: a random member of a rule drawn from the learner's rule prior.
    mix_logit = pm.Normal("mix_logit", mu=0.0, sigma=1.5)
    mix = pm.Deterministic("mix", pm.math.sigmoid(mix_logit))
    q = (1 - mix) * 0.01 + mix * STRAY_RULE_GEN
    inv_q = example_mask / q[example_idx]  # (sets, 4)

    # Reliability-marginal weights c_j = B(u + j, v + n - j) / B(u, v) for each n.
    c_table = pt.where(_VALID, pt.exp(_betaln(u + _J, v + _N_MINUS_J) - _betaln(u, v)), 0.0)
    c_set = c_table[set_n_idx]  # (sets, j)

    # e_j for every subset of a set's examples, folded with c_j and size^-j.
    e_sub = _elementary(SUBSET_IN[None, :, :] * inv_q[:, None, :])  # (sets, subsets, j)
    sub_size_lik = pt.dot(e_sub * c_set[:, None, :], _SIZE_POW.T)  # (sets, subsets, size)

    # Rules: the subset of examples each rule contains, and its size.
    rule_lik = sub_size_lik[pt.arange(rule_sub.shape[0])[:, None], rule_sub, RULE_SIZE_IDX[None, :]]
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + pt.log(rule_lik)

    # Ranges: sum_h sum_J c_|J| prod_{i in J} g_h(x_i) / q(x_i); the
    # g-products are parameter-free and pre-summed over clusters per subset J.
    w_sub = pt.prod(1 - SUBSET_IN[None, :, :] + SUBSET_IN[None, :, :] * inv_q[:, None, :], axis=2)  # (sets, J)
    sub_coef = w_sub * c_set[:, SUBSET_IN.sum(axis=1).astype("int64")]
    cl_ev = (sub_coef * sub_g).sum(axis=1)
    log_clusters = pt.log1p(-lambda_0) - np.log(N_CL) + pt.log(cl_ev)

    log_joint = pt.concatenate([log_rules, log_clusters[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    cluster_post = pt.exp(log_clusters - log_norm)

    cell_cl_share = (sub_coef[cell_set] * cell_sub_g).sum(axis=1) / cl_ev[cell_set]
    in_concept = (rule_post[cell_set] * cell_rule_member).sum(axis=1) + cluster_post[cell_set] * cell_cl_share
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-9, 1 - 1e-9)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    example_idx = np.zeros((S, MAX_N), dtype="int64")
    example_mask = np.zeros((S, MAX_N))
    set_n_idx = np.zeros(S, dtype="int64")
    cl_sub = []  # per set: (clusters, subsets) products of g over each subset
    sub_g = np.zeros((S, 2**MAX_N))
    for i, s in enumerate(unique):
        ex = np.sort(np.array(s))
        n = len(ex)
        set_n_idx[i] = n - 1
        example_idx[i, :n] = ex - 1
        example_mask[i, :n] = 1.0
        g = np.zeros((N_CL, MAX_N))
        g[:, :n] = CL_PROB[:, ex - 1]
        prods = np.prod(1 - SUBSET_IN[None, :, :] + SUBSET_IN[None, :, :] * g[:, None, :], axis=2)
        prods[:, SUBSET_IN[:, n:].sum(axis=1) > 0] = 0.0  # subsets with padding
        cl_sub.append(prods)
        sub_g[i] = prods.sum(axis=0)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    return {
        "example_idx": example_idx,
        "example_mask": example_mask,
        "rule_sub": (
            (RULE_MEMBERSHIP[:, example_idx].transpose(1, 0, 2) * example_mask[:, None, :])
            @ (2 ** np.arange(MAX_N))
        ).astype("int64"),
        "set_n_idx": set_n_idx,
        "sub_g": sub_g,
        "cell_sub_g": np.stack([CL_TYPICALITY[:, t - 1] @ cl_sub[i] for i, t in cells]),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
