"""Refinement of monotone_order_range_prior_shift: magnitude ranges live on a
compressed (logarithmic) mental number line.

Same tempered, serial-position-weighted Bayesian concept learning over 45
weighted rules and all intervals of 1..100, with the sorted-order prior shift
toward ranges. The one change: an interval [a, b] is represented on the
mental number line f(n) = log(n + c), so its psychological length is
W = f(b + .5) - f(a - .5), and examples from it are assumed spread evenly on
that line (number x has probability (f(x + .5) - f(x - .5)) / W). Compression
c is inferred; c -> infinity recovers the linear (original) model. This makes
generalization from ranges asymmetric: extending a range downward costs more
than extending it upward.
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
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
N_INT = len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INT_LO = np.array([c[0] for c in INTERVALS], dtype="float64")
INT_HI = np.array([c[-1] for c in INTERVALS], dtype="float64")

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    # Per-set, per-position example log-likelihoods (zero-padded).
    rule_pos_ll = pm.Data("rule_pos_ll", np.zeros((1, MAX_N, N_RULES)))
    ux_val = pm.Data("ux_val", np.ones(1, dtype="int64"))                 # distinct example values
    ux_mem = pm.Data("ux_mem", np.zeros((1, N_INT)))                      # value in interval?
    pos_ux = pm.Data("pos_ux", np.zeros((1, MAX_N), dtype="int64"))       # index into ux_val
    pos_rel = pm.Data("pos_rel", np.zeros((1, MAX_N)))   # -0.5 .. +0.5 serial position
    pos_mask = pm.Data("pos_mask", np.ones((1, MAX_N)))  # 1 for real examples
    sorted_order = pm.Data("sorted_order", np.zeros(1))  # 1 if >=3 examples shown sorted
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
    log_beta = pm.Normal("log_beta", mu=0.0, sigma=0.75)
    beta = pm.Deterministic("beta", pt.exp(log_beta))
    # Serial-position gradient: >0 recency, <0 primacy.
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)
    # Sorted-order shift of prior log-odds from rules toward intervals.
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)
    rule_logodds = pt.log(lambda_0) - pt.log1p(-lambda_0) - kappa * sorted_order  # (sets,)
    log_rule_share = -pt.softplus(-rule_logodds)
    log_int_share = -pt.softplus(rule_logodds)

    raw = pt.exp(gamma * pos_rel) * pos_mask
    n_ex = pos_mask.sum(axis=1, keepdims=True)
    pos_w = beta * raw * n_ex / raw.sum(axis=1, keepdims=True)  # (sets, MAX_N), mean beta

    # Log-compressed mental number line for ranges.
    log_c = pm.Normal("log_c", mu=2.0, sigma=1.5)
    c = pt.exp(log_c)
    f = lambda v: pt.log(v + c)
    w_num = f(NUMBERS + 0.5) - f(NUMBERS - 0.5)                      # (100,)
    W_int = f(INT_HI + 0.5) - f(INT_LO - 0.5)                        # (N_INT,)
    # Log-likelihood of each distinct example value under each interval.
    ux_ll = pt.log(ALPHA * ux_mem * w_num[ux_val - 1][:, None] / W_int[None, :] + (1 - ALPHA) / 100)
    set_idx = pt.arange(pos_ux.shape[0])[:, None] + pt.zeros_like(pos_ux)
    A = pt.zeros((pos_ux.shape[0], ux_val.shape[0]))[set_idx, pos_ux].inc(pos_w)  # (sets, n_ux)

    rule_ll = (pos_w[:, :, None] * rule_pos_ll).sum(axis=1)
    t_int = pt.dot(A, ux_ll)
    int_log_evidence = pt.logsumexp(t_int, axis=1)
    int_post = pt.exp(t_int - int_log_evidence[:, None])
    interval_in_concept = pt.dot(int_post, pt.as_tensor(INTERVAL_MEMBERSHIP))

    log_rules = log_rule_share[:, None] + pt.log(rule_weights)[None, :] + rule_ll
    log_intervals = log_int_share - 2 * np.log(100) + int_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + posterior[:, N_RULES][:, None] * interval_in_concept

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    pos_ll = np.zeros((S, MAX_N, N_RULES))
    ux = sorted({x for st in unique for x in st})
    ux_index = {x: j for j, x in enumerate(ux)}
    pos_ux = np.zeros((S, MAX_N), dtype="int64")
    pos_rel = np.zeros((S, MAX_N))
    pos_mask = np.zeros((S, MAX_N))
    sorted_order = np.zeros(S)
    for i, s in enumerate(unique):
        n = len(s)
        for k, x in enumerate(s):
            pos_ll[i, k] = EXAMPLE_LOG_LIK[:N_RULES, x - 1]
            pos_ux[i, k] = ux_index[x]
            pos_mask[i, k] = 1.0
            pos_rel[i, k] = (k / (n - 1) - 0.5) if n > 1 else 0.0
        diffs = [b - a for a, b in zip(s, s[1:])]
        if n >= 3 and (all(x > 0 for x in diffs) or all(x < 0 for x in diffs)):
            sorted_order[i] = 1.0
    return {
        "rule_pos_ll": pos_ll,
        "ux_val": np.array(ux, dtype="int64"),
        "ux_mem": INTERVAL_MEMBERSHIP[:, np.array(ux) - 1].T.copy(),
        "pos_ux": pos_ux,
        "pos_rel": pos_rel,
        "pos_mask": pos_mask,
        "sorted_order": sorted_order,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
