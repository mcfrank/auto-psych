"""Refinement of power_law_endpoints_erlang_width: Bayesian concept learning
over 45 rules and all intervals [m, n] of 1-100 (size principle, 10% of
examples uniform on 1-100, a prior weight per rule, power-law endpoint prior
(m n)^-gamma and an Erlang prior on interval size). The one change: people
believe the program samples examples from the concept on a compressed mental
number line, so a member x is drawn with probability proportional to x^-c
(c fitted in [-1, 2]; c = 0 recovers uniform sampling, c = 1 is a logarithmic
number line). A range among large numbers is then psychologically narrower, so
the size principle penalises extending it less, and generalisation beyond the
examples scales with their magnitude. "Yes" is the posterior probability that
the concept contains the target.

For speed, each set's interval log likelihood as a function of c is computed
exactly at Chebyshev nodes once (prepare_observed) and evaluated in the model
as a degree-8 Chebyshev polynomial in c (max error ~1e-2 nats per interval);
the rules' likelihood is computed exactly.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NOISE = (1 - ALPHA) / 100
NUMBERS = np.arange(1, 101)
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
MAX_N = 4
C_LO, C_HI = -1.0, 2.0  # range of the compression exponent
DEGREE = 8
NODES = np.cos(np.pi * (np.arange(DEGREE + 1) + 0.5) / (DEGREE + 1))


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

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

# The 5,050 intervals [m, n], m <= n, as flat vectors.
LO = np.array([m for m in NUMBERS for n in range(m, 101)])
HI = np.array([n for m in NUMBERS for n in range(m, 101)])
SIZE = (HI - LO + 1).astype("float64")
LOG_MN = np.log(LO * HI.astype("float64"))
CONTAINS = ((LO[:, None] <= NUMBERS[None, :]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")


def interval_log_lik(examples, c):
    """Exact log likelihood of the examples under every interval, sampling density x^-c."""
    w = NUMBERS ** (-c)
    cum = np.concatenate([[0.0], np.cumsum(w)])
    z = cum[HI] - cum[LO - 1]
    return sum(np.log(ALPHA * CONTAINS[:, x - 1] * x ** (-c) / z + NOISE) for x in examples)


with pm.Model() as model:
    ex_log_x = pm.Data("ex_log_x", np.zeros((1, MAX_N)))  # log of each example (sets, slots)
    ex_mask = pm.Data("ex_mask", np.zeros((1, MAX_N)))  # slot holds an example
    ex_rule_in = pm.Data("ex_rule_in", np.zeros((1, MAX_N, N_RULES)))  # example in rule
    int_cheb = pm.Data("int_cheb", np.zeros((DEGREE + 1, 1, len(LO))))  # Chebyshev coefficients in c
    int_log_scale = pm.Data("int_log_scale", np.zeros(1))  # per-set max log likelihood over the c range
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
    gamma = pm.Normal("gamma", mu=0.0, sigma=1.0)
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))
    # Compression of the mental number line used for sampling (prior mean 0 = linear).
    compression_u = pm.Beta("compression_u", 2.0, 4.0)
    compression = pm.Deterministic("compression", C_LO + (C_HI - C_LO) * compression_u)

    # Rules: exact likelihood with members drawn proportionally to x^-c.
    w = pt.exp(-compression * pt.as_tensor(LOG_NUMBERS))  # (100,)
    log_z_rule = pt.log(pt.as_tensor(RULE_MEMBERSHIP) @ w)  # (rules,)
    lx = -compression * ex_log_x[:, :, None]  # (S, slots, 1)
    rule_ll = pt.log(ALPHA * ex_rule_in * pt.exp(lx - log_z_rule[None, None, :]) + NOISE)
    rule_log_lik = (rule_ll * ex_mask[:, :, None]).sum(axis=1)  # (S, rules)

    # Intervals: Chebyshev polynomial in c of the precomputed exact log likelihood.
    t = 2.0 * compression_u - 1.0
    basis = [pt.ones(()), t]
    for _ in range(2, DEGREE + 1):
        basis.append(2.0 * t * basis[-1] - basis[-2])
    int_log_lik = pt.tensordot(pt.stack(basis), int_cheb, axes=[[0], [0]])  # (S, intervals)

    log_prior = -gamma * LOG_MN + np.log(SIZE) - SIZE / sigma
    log_prior = log_prior - pt.logsumexp(log_prior)
    # A fixed per-set scale: c is bounded, so the shifted log likelihood stays far from overflow.
    weighted = pt.exp(int_log_lik - int_log_scale[:, None] + log_prior[None, :])  # (S, intervals)
    total_mass = weighted.sum(axis=1)
    mass = weighted @ pt.as_tensor(CONTAINS)  # (S, 100)

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_interval = pt.log1p(-lambda_0) + pt.log(total_mass) + int_log_scale
    log_joint = pt.concatenate([log_rules, log_interval[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1, keepdims=True)
    post_rules = pt.exp(log_rules - log_norm)
    post_interval = pt.exp(log_interval[:, None] - log_norm)
    in_concept = post_rules @ pt.as_tensor(RULE_MEMBERSHIP) + post_interval * mass / total_mass[:, None]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1.0 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n = len(unique)
    log_x = np.zeros((n, MAX_N))
    mask = np.zeros((n, MAX_N))
    rule_in = np.zeros((n, MAX_N, N_RULES))
    for i, s in enumerate(unique):
        for k, x in enumerate(s):
            log_x[i, k] = np.log(x)
            mask[i, k] = 1.0
            rule_in[i, k] = RULE_MEMBERSHIP[:, x - 1]
    at_nodes = np.stack([
        np.stack([interval_log_lik(s, C_LO + (C_HI - C_LO) * (node + 1) / 2) for s in unique]).ravel()
        for node in NODES
    ])  # (nodes, S * intervals)
    scale = at_nodes.reshape(len(NODES), n, len(LO)).max(axis=(0, 2))
    cheb = np.polynomial.chebyshev.chebfit(NODES, at_nodes, DEGREE).reshape(DEGREE + 1, n, len(LO))
    return {
        "ex_log_x": log_x,
        "ex_mask": mask,
        "ex_rule_in": rule_in,
        "int_cheb": cheb,
        "int_log_scale": scale,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
