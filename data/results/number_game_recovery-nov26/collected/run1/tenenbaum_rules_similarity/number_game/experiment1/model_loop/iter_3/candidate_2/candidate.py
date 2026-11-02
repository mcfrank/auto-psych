"""Judgment by predicted production probability, not by concept membership.

People infer the hidden program by Bayesian concept learning (45 rules with a
prior mass per concept family, plus all intervals of 1..100, size principle,
10% of examples random), but they answer "would the program produce this
number?" by how probable the target is as the program's *next output* -- the
posterior predictive production probability, which divides each concept's
support by its size -- rather than by the posterior probability that the target
is a member. Targets in small, tight concepts are thus endorsed more than targets
that belong only to broad concepts, and the yes rate rises smoothly (logistic)
with the log ratio of this production probability to a uniform random program.
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


def family(name):
    if name in ("odd", "even"):
        return 0
    if name.startswith("multiples"):
        return 2
    if name.startswith("ends"):
        return 3
    if name.startswith("contains"):
        return 4
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in RULE_NAMES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
# Probability that each concept's program emits each number next.
PRODUCTION = ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100
EXAMPLE_LOG_LIK = np.log(PRODUCTION)
RULE_PRODUCTION = PRODUCTION[:N_RULES]
N_INTERVALS = len(INTERVALS)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    interval_log_evidence = pm.Data("interval_log_evidence", np.zeros(1))
    interval_production = pm.Data("interval_production", np.full((1, 100), 0.01))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_weight = pm.Dirichlet("family_weight", a=np.ones(6))
    log_w = pt.log(family_weight)
    log_rules = (log_w[:5] - np.log(FAMILY_SIZE))[RULE_FAMILY][None, :] + rule_log_lik
    log_intervals = log_w[5] - np.log(N_INTERVALS) + interval_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))

    # Posterior predictive probability that the program produces each number next.
    production = posterior[:, :N_RULES] @ pt.as_tensor(RULE_PRODUCTION) + posterior[:, N_RULES][:, None] * interval_production
    log_ratio = pt.log(100.0 * production).reshape((-1,))[row_cell]

    # Yes rate rises with production probability relative to a random program.
    bias = pm.Normal("bias", mu=-2.0, sigma=2.0)
    sensitivity = pm.HalfNormal("sensitivity", sigma=2.0)
    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(bias + sensitivity * log_ratio), 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_log_lik = set_log_lik[:, N_RULES:]
    log_evidence = logsumexp(interval_log_lik, axis=1)
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "interval_log_evidence": log_evidence,
        "interval_production": np.exp(interval_log_lik - log_evidence[:, None]) @ PRODUCTION[N_RULES:],
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
