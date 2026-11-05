"""Bounded-range rule learning.

People believe the program produces numbers that follow a mathematical rule
but only within a bounded range spanned by the examples: they do Bayesian
concept learning over "rule restricted to a range" concepts (any of the 45
rules, or no rule at all, intersected with an interval [a, b] that covers
every example), weighting each by the size principle (10% of examples
uniform noise) with an inferred, tempered evidence strength. Because a rule
restricted to the examples' range is a smaller, better-fitting concept than
the unbounded rule, people generalize mainly to rule members inside the
[min, max] span of the examples and much less to rule members outside it.
"Yes" is the posterior probability that the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


def rule_concepts():
    rules = {
        "all numbers": list(NUMBERS),
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


RULES = list(rule_concepts().values())
N_RULES = len(RULES)  # 46: "all numbers" (index 0) + 45 rules
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")  # (R, 100)


def _set_tables(examples):
    """Size-principle log-likelihoods of the examples under every (rule, range [a, b])
    concept whose range covers all examples, on a grid a = 1..min, b = max..100."""
    lo, hi = min(examples), max(examples)
    ex = np.array(examples) - 1
    rm_ex = RULE_MEMBERSHIP[:, ex]  # (R, n)
    cum = np.concatenate([np.zeros((N_RULES, 1)), np.cumsum(RULE_MEMBERSHIP, axis=1)], axis=1)
    ll = np.empty((N_RULES, lo, 101 - hi))
    for i, a in enumerate(range(1, lo + 1)):
        for j, b in enumerate(range(hi, 101)):
            size = cum[:, b] - cum[:, a - 1]  # |rule ∩ [a, b]|
            p = ALPHA * rm_ex / np.maximum(size, 1.0)[:, None] + (1 - ALPHA) / 100
            ll[:, i, j] = np.log(p).sum(axis=1)
    return ll, lo, hi


def prepare_observed(rows):
    """Flatten every set's (rule, a, b) concepts into one hypothesis list (computed once)."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    ll_all, set_all, rule_all, a_all, b_all, set_max = [], [], [], [], [], []
    for s, ex in enumerate(unique):
        ll, lo, hi = _set_tables(ex)
        r, i, j = np.meshgrid(np.arange(N_RULES), np.arange(lo), np.arange(101 - hi), indexing="ij")
        ll_all.append(ll.ravel())
        set_all.append(np.full(ll.size, s))
        rule_all.append(r.ravel())
        a_all.append(i.ravel())  # a - 1
        b_all.append(hi - 1 + j.ravel())  # b - 1
        set_max.append(ll.max())
    set_h = np.concatenate(set_all)
    rule_h = np.concatenate(rule_all)
    sr = set_h * N_RULES + rule_h
    return {
        "hyp_log_lik": np.concatenate(ll_all),
        "hyp_set": set_h.astype("int64"),
        "hyp_rule": rule_h.astype("int64"),
        "hyp_sr": sr.astype("int64"),
        "hyp_sra": (sr * 100 + np.concatenate(a_all)).astype("int64"),
        "hyp_srb": (sr * 100 + np.concatenate(b_all)).astype("int64"),
        "set_max_log_lik": np.array(set_max),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    hyp_log_lik = pm.Data("hyp_log_lik", np.zeros(1))  # size-principle log P(examples | concept)
    hyp_set = pm.Data("hyp_set", np.zeros(1, dtype="int64"))
    hyp_rule = pm.Data("hyp_rule", np.zeros(1, dtype="int64"))
    hyp_sr = pm.Data("hyp_sr", np.zeros(1, dtype="int64"))
    hyp_sra = pm.Data("hyp_sra", np.zeros(1, dtype="int64"))
    hyp_srb = pm.Data("hyp_srb", np.zeros(1, dtype="int64"))
    set_max_log_lik = pm.Data("set_max_log_lik", np.zeros(1))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    n_sets = set_max_log_lik.shape[0]

    # Prior share of the rule-free ("all numbers") range concepts; the 45 rules split the rest.
    lambda_free = pm.Beta("lambda_free", 2, 2)
    log_rule_prior = pt.concatenate([
        pt.log(lambda_free)[None],
        pt.repeat(pt.log1p(-lambda_free) - np.log(N_RULES - 1), N_RULES - 1),
    ])
    # Evidence-strength exponent on the size-principle likelihood.
    log_beta = pm.Normal("log_beta", mu=-0.5, sigma=0.75)
    beta = pm.Deterministic("beta", pt.exp(log_beta))

    shift = beta * set_max_log_lik + pt.max(log_rule_prior)
    w = pt.exp(log_rule_prior[hyp_rule] + beta * hyp_log_lik - shift[hyp_set])
    Z = pt.inc_subtensor(pt.zeros(n_sets)[hyp_set], w)
    w_sr = pt.inc_subtensor(pt.zeros(n_sets * N_RULES)[hyp_sr], w).reshape((n_sets, N_RULES))
    w_a = pt.inc_subtensor(pt.zeros(n_sets * N_RULES * 100)[hyp_sra], w).reshape((n_sets, N_RULES, 100))
    w_b = pt.inc_subtensor(pt.zeros(n_sets * N_RULES * 100)[hyp_srb], w).reshape((n_sets, N_RULES, 100))
    # P(a <= t <= b) = P(a <= t) + P(b >= t) - P(rule), since a > t > b is impossible.
    a_le_t = pt.cumsum(w_a, axis=2)
    b_ge_t = pt.cumsum(w_b[:, :, ::-1], axis=2)[:, :, ::-1]
    in_range = a_le_t + b_ge_t - w_sr[:, :, None]  # (S, R, 100), unnormalized
    in_concept = (in_range * pt.as_tensor(RULE_MEMBERSHIP)[None, :, :]).sum(axis=1) / Z[:, None]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
