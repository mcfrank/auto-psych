"""Refinement of monotone_order_range_prior_shift: range-bounded rules.

People do the same tempered, serial-position-weighted Bayesian concept
learning over 45 weighted rules and all intervals of 1..100, with sorted lists
of >= 3 examples shifting prior log-odds from rules toward intervals (kappa).
The one change (a component from bounded_range_rule_learning): people can also
restrict a rule to a range [a, b] covering the examples ("multiples of 3
between 9 and 70"); these bounded rules take an inferred share eta of the
rule prior (eta = 0 recovers the incumbent). A bounded rule is smaller, so the
size principle favours it and rule members outside the examples' min-max
range are endorsed less than those inside. "Yes" is the posterior probability
that the concept contains the target.
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

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0


def _bounded_pos_ll(examples):
    """Per-example log-lik under each (rule, [a, b]) with a <= min, b >= max: (R, A, B, n)."""
    lo, hi = min(examples), max(examples)
    ex = np.array(examples) - 1
    rm_ex = RULE_MEMBERSHIP[:, ex]
    cum = np.concatenate([np.zeros((N_RULES, 1)), np.cumsum(RULE_MEMBERSHIP, axis=1)], axis=1)
    a = np.arange(1, lo + 1)
    b = np.arange(hi, 101)
    size = cum[:, b][:, None, :] - cum[:, a - 1][:, :, None]  # (R, A, B)
    p = ALPHA * rm_ex[:, None, None, :] / np.maximum(size, 1.0)[..., None] + (1 - ALPHA) / 100
    return np.log(p), lo, hi


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    pos_ll = np.zeros((S, MAX_N, N_RULES + N_INT))
    pos_rel = np.zeros((S, MAX_N))
    pos_mask = np.zeros((S, MAX_N))
    sorted_order = np.zeros(S)
    hyp_ll, hyp_set, hyp_rule, a_all, b_all = [], [], [], [], []
    pos_max = np.zeros((S, MAX_N))
    for i, s in enumerate(unique):
        n = len(s)
        for k, x in enumerate(s):
            pos_ll[i, k] = EXAMPLE_LOG_LIK[:, x - 1]
            pos_mask[i, k] = 1.0
            pos_rel[i, k] = (k / (n - 1) - 0.5) if n > 1 else 0.0
        diffs = [b - a for a, b in zip(s, s[1:])]
        if n >= 3 and (all(x > 0 for x in diffs) or all(x < 0 for x in diffs)):
            sorted_order[i] = 1.0
        ll, lo, hi = _bounded_pos_ll(s)
        flat = ll.reshape((-1, n))
        padded = np.zeros((flat.shape[0], MAX_N))
        padded[:, :n] = flat
        pos_max[i, :n] = flat.max(axis=0)
        r, ai, bj = np.meshgrid(np.arange(N_RULES), np.arange(lo), np.arange(101 - hi), indexing="ij")
        hyp_ll.append(padded)
        hyp_set.append(np.full(flat.shape[0], i))
        hyp_rule.append(r.ravel())
        a_all.append(ai.ravel())
        b_all.append(hi - 1 + bj.ravel())
    hset = np.concatenate(hyp_set)
    sr = hset * N_RULES + np.concatenate(hyp_rule)
    return {
        "rule_pos_ll": pos_ll[:, :, :N_RULES],
        "interval_pos_ll": pos_ll[:, :, N_RULES:],
        "pos_rel": pos_rel,
        "pos_mask": pos_mask,
        "sorted_order": sorted_order,
        "hyp_pos_ll": np.concatenate(hyp_ll),
        "hyp_pos_max": pos_max,
        "hyp_set": hset.astype("int64"),
        "hyp_rule": np.concatenate(hyp_rule).astype("int64"),
        "hyp_sr": sr.astype("int64"),
        "hyp_sra": (sr * 100 + np.concatenate(a_all)).astype("int64"),
        "hyp_srb": (sr * 100 + np.concatenate(b_all)).astype("int64"),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    rule_pos_ll = pm.Data("rule_pos_ll", np.zeros((1, MAX_N, N_RULES)))
    interval_pos_ll = pm.Data("interval_pos_ll", np.zeros((1, MAX_N, N_INT)))
    pos_rel = pm.Data("pos_rel", np.zeros((1, MAX_N)))
    pos_mask = pm.Data("pos_mask", np.ones((1, MAX_N)))
    sorted_order = pm.Data("sorted_order", np.zeros(1))
    hyp_pos_ll = pm.Data("hyp_pos_ll", np.zeros((1, MAX_N)))
    hyp_pos_max = pm.Data("hyp_pos_max", np.zeros((1, MAX_N)))
    hyp_set = pm.Data("hyp_set", np.zeros(1, dtype="int64"))
    hyp_rule = pm.Data("hyp_rule", np.zeros(1, dtype="int64"))
    hyp_sr = pm.Data("hyp_sr", np.zeros(1, dtype="int64"))
    hyp_sra = pm.Data("hyp_sra", np.zeros(1, dtype="int64"))
    hyp_srb = pm.Data("hyp_srb", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    n_sets = pos_mask.shape[0]

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
    gamma = pm.Normal("gamma", mu=0.0, sigma=2.0)   # serial-position gradient
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)   # sorted-order shift rules -> intervals
    eta = pm.Beta("eta", 1.0, 3.0)                  # the one change: bounded-rule share
    rule_logodds = pt.log(lambda_0) - pt.log1p(-lambda_0) - kappa * sorted_order
    log_rule_share = -pt.softplus(-rule_logodds)
    log_int_share = -pt.softplus(rule_logodds)

    raw = pt.exp(gamma * pos_rel) * pos_mask
    n_ex = pos_mask.sum(axis=1, keepdims=True)
    pos_w = beta * raw * n_ex / raw.sum(axis=1, keepdims=True)  # (sets, MAX_N)

    rule_ll = (pos_w[:, :, None] * rule_pos_ll).sum(axis=1)
    t_int = (pos_w[:, :, None] * interval_pos_ll).sum(axis=1)
    int_log_evidence = pt.logsumexp(t_int, axis=1)
    int_post = pt.exp(t_int - int_log_evidence[:, None])
    interval_in_concept = pt.dot(int_post, pt.as_tensor(INTERVAL_MEMBERSHIP))

    # Range-bounded rules: prior share_rule * eta * w_r / N_INT for each covering range.
    log_rules = log_rule_share[:, None] + pt.log1p(-eta) + pt.log(rule_weights)[None, :] + rule_ll
    log_b_prior = pt.log(eta) + pt.log(rule_weights) - np.log(N_INT)
    hyp_ll = (pos_w[hyp_set] * hyp_pos_ll).sum(axis=1)
    shift = (pos_w * hyp_pos_max).sum(axis=1) + pt.max(log_b_prior)  # upper bound per set
    w = pt.exp(log_b_prior[hyp_rule] + hyp_ll - shift[hyp_set])
    Zb = pt.inc_subtensor(pt.zeros(n_sets)[hyp_set], w)
    log_Zb = log_rule_share + pt.log(Zb) + shift
    w_sr = pt.inc_subtensor(pt.zeros(n_sets * N_RULES)[hyp_sr], w).reshape((n_sets, N_RULES))
    w_a = pt.inc_subtensor(pt.zeros(n_sets * N_RULES * 100)[hyp_sra], w).reshape((n_sets, N_RULES, 100))
    w_b = pt.inc_subtensor(pt.zeros(n_sets * N_RULES * 100)[hyp_srb], w).reshape((n_sets, N_RULES, 100))
    a_le_t = pt.cumsum(w_a, axis=2)
    b_ge_t = pt.cumsum(w_b[:, :, ::-1], axis=2)[:, :, ::-1]
    in_range = a_le_t + b_ge_t - w_sr[:, :, None]
    bounded_in = (in_range * pt.as_tensor(RULE_MEMBERSHIP)[None, :, :]).sum(axis=1) / Zb[:, None]

    log_intervals = log_int_share - 2 * np.log(100) + int_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None], log_Zb[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)
                  + posterior[:, N_RULES][:, None] * interval_in_concept
                  + posterior[:, N_RULES + 1][:, None] * bounded_in)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
