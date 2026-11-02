"""Refinement of learned_rule_prior_mixture: Bayesian concept learning over 45
rules (each with its own learned prior plausibility) and all intervals of
1..100 with the full size principle, where intervals are NOT equally likely a
priori: their prior follows an Erlang density on interval size,
p(|h|) ~ (|h| / sigma^2) exp(-|h| / sigma), with a fitted scale sigma (as in
tenenbaum_rules_similarity), so people expect ranges of a typical width.
"Yes" is the posterior probability that the concept contains the target.
Single change versus the incumbent: uniform interval prior -> Erlang
size-graded interval prior.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


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
RULES = list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
# Intervals [lo, lo + L - 1], grouped by size L = 1..100 (prior depends only on L).
SIZES = np.arange(1, 101, dtype="float64")
N_PER_SIZE = 101.0 - SIZES  # number of intervals of each size
INTERVALS = [(lo, lo + L - 1) for L in range(1, 101) for lo in range(1, 102 - L)]
INT_LO = np.array([a for a, _ in INTERVALS])
INT_HI = np.array([b for _, b in INTERVALS])
INT_SIZE = INT_HI - INT_LO + 1


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_rule_in = pm.Data("cell_rule_in", np.zeros((1, N_RULES)))  # (cells, rules)
    size_log_lik = pm.Data("size_log_lik", np.zeros((1, 100)))  # (cells, sizes) log sum of example lik over intervals of size L
    size_frac_in = pm.Data("size_frac_in", np.zeros((1, 100)))  # (cells, sizes) lik-weighted share containing target
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    rule_logit = pm.ZeroSumNormal("rule_logit", sigma=1.0, shape=N_RULES)
    sigma = pm.LogNormal("sigma", mu=np.log(10.0), sigma=0.75)

    # Erlang prior over interval sizes, normalized over all intervals of 1..100.
    log_erlang = pt.log(pt.as_tensor(SIZES)) - SIZES / sigma
    log_z = pt.logsumexp(log_erlang + np.log(N_PER_SIZE))
    log_size_prior = log_erlang - log_z  # per-interval prior (within the interval mass)

    log_rules = pt.log(lambda_0) + pt.special.log_softmax(rule_logit)[None, :] + rule_log_lik
    log_sizes = pt.log1p(-lambda_0) + log_size_prior[None, :] + size_log_lik
    log_joint = pt.concatenate([log_rules, log_sizes], axis=1)
    post = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (post[:, :N_RULES] * cell_rule_in).sum(axis=1) + (post[:, N_RULES:] * size_frac_in).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Per-trial arrays; the parameter-free likelihoods are computed once per set."""
    cache = {}
    rl, ri, sl, sf = [], [], [], []
    for r in rows:
        s = tuple(int(x) for x in str(r["number_set"]).split(","))
        t = int(r["target"])
        if s not in cache:
            x = np.array(s)
            inside = (x[None, :] >= INT_LO[:, None]) & (x[None, :] <= INT_HI[:, None])
            ll = np.log(ALPHA * inside / INT_SIZE[:, None] + (1 - ALPHA) / 100).sum(axis=1)
            g = np.array([logsumexp(ll[INT_SIZE == L]) for L in range(1, 101)])
            cache[s] = (RULE_LOG_LIK[:, x - 1].sum(axis=1), ll, g)
        rule_ll, ll, g = cache[s]
        contains = (INT_LO <= t) & (INT_HI >= t)
        w = np.exp(ll - g[INT_SIZE - 1])
        frac = np.bincount(INT_SIZE - 1, weights=w * contains, minlength=100)
        rl.append(rule_ll)
        ri.append(RULE_MEMBERSHIP[:, t - 1])
        sl.append(g)
        sf.append(np.clip(frac, 0.0, 1.0))
    return {
        "rule_log_lik": np.array(rl),
        "cell_rule_in": np.array(ri),
        "size_log_lik": np.array(sl),
        "size_frac_in": np.array(sf),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
