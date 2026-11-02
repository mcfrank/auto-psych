"""Fuzzy-trace memory for the examples in Bayesian concept learning.

People do Bayesian concept learning over 45 rules and every interval of 1..100
(size principle, 10% noisy examples, rules sharing prior mass lambda_0 equally,
an Erlang prior on interval size), but each example is remembered verbatim only
with probability r; otherwise only its gist -- its magnitude to within +-5 --
is retained. The likelihood of an example's trace under a concept is therefore
r * p(e | h) + (1 - r) * mean_{|x - e| <= 5} p(x | h). Identity-dependent
concepts (digit rules, multiples, powers) lose support with every fuzzily
remembered example, while ranges, which depend only on magnitude, keep it.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
WINDOW = 5


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


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
LO = np.array([m for m in NUMBERS for n in range(m, 101)])
HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INTERVAL_SIZE = (HI - LO + 1).astype("float64")
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, INTERVAL_MEMBERSHIP], axis=0)  # (hyp, 100)
N_HYP = MEMBERSHIP.shape[0]
# p(x | h) for every hypothesis and number, with 10% uniform noise.
EX_LIK = ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100
# Gist likelihood: mean of p(x | h) over numbers within WINDOW of each value.
BLUR = np.array([(np.abs(NUMBERS - e) <= WINDOW) / (np.abs(NUMBERS - e) <= WINDOW).sum() for e in NUMBERS])
GIST_LIK = EX_LIK @ BLUR.T  # (hyp, 100): column e = mean over window(e)
MAX_N = 4

with pm.Model() as model:
    exact_lik = pm.Data("exact_lik", np.ones((1, MAX_N, N_HYP)))  # p(e_k | h); 1 for missing examples
    gist_lik = pm.Data("gist_lik", np.ones((1, MAX_N, N_HYP)))
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))  # target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pt.exp(log_sigma)
    # Probability that an example is remembered verbatim rather than as gist.
    r = pm.Beta("r", 2, 2)

    size = pt.as_tensor(INTERVAL_SIZE)
    log_erlang = pt.log(size) - size / sigma
    log_prior = pt.concatenate([
        pt.zeros(N_RULES) + (pt.log(lambda_0) - np.log(N_RULES)),
        pt.log1p(-lambda_0) + log_erlang - pt.logsumexp(log_erlang),
    ])
    trace_ll = pt.log(r * exact_lik + (1 - r) * gist_lik).sum(axis=1)  # (sets, hyp)
    post = pt.special.softmax(log_prior[None, :] + trace_ll, axis=1)
    gen = pt.dot(post, pt.as_tensor(MEMBERSHIP))  # (sets, 100)
    p_yes = pm.Deterministic("p_yes", pt.clip(gen[row_set, row_target], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    exact = np.ones((len(unique), MAX_N, N_HYP))
    gist = np.ones((len(unique), MAX_N, N_HYP))
    for i, s in enumerate(unique):
        for k, e in enumerate(s):
            exact[i, k] = EX_LIK[:, e - 1]
            gist[i, k] = GIST_LIK[:, e - 1]
    return {
        "exact_lik": exact,
        "gist_lik": gist,
        "row_set": np.array([index[s] for s in sets], dtype="int64"),
        "row_target": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
