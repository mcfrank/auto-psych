"""Noisy decision criterion on Bayesian membership probability.

People infer how likely the target is to belong to the hidden concept exactly
as the current best model's Bayesian concept learner does (45 rules and every
interval of 1..100; lambda_0 for rules with inferred family weights, uniform
within family; each interval (1 - lambda_0) / 100^2; size principle with 90%
of examples from the concept), but they do not report that probability
directly: they answer "yes" only when it exceeds an internal criterion that
varies noisily (logistically) from judgment to judgment around an inferred
location tau with inferred spread 1 / gain.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

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
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
INTERVAL_MEMBERSHIP = np.array(
    [(NUMBERS >= m) & (NUMBERS <= n) for m in NUMBERS for n in range(m, 101)], dtype="float64"
)


def example_log_lik(membership):
    return np.log(ALPHA * membership / membership.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)


RULE_EX_LL = example_log_lik(RULE_MEMBERSHIP)
INTERVAL_EX_LL = example_log_lik(INTERVAL_MEMBERSHIP)


def rule_family(name):
    if name in ("odd", "even"):
        return 0
    if name == "primes":
        return 1
    if name in ("squares", "cubes"):
        return 2
    if name.startswith("powers"):
        return 3
    if name.startswith("multiples"):
        return 4
    if name.startswith("ends in"):
        return 5
    return 6


FAMILY = np.array([rule_family(n) for n in RULE_NAMES])
N_FAMILIES = FAMILY.max() + 1
FAMILY_SIZE = np.bincount(FAMILY)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules): log P(D | rule)
    interval_log_marg = pm.Data("interval_log_marg", np.zeros(1))  # (sets,): log sum_I P(D | I)
    interval_pred = pm.Data("interval_pred", np.zeros((1, 100)))  # (sets, targets): P(t in I | D, interval)
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))  # target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    # Decision criterion on membership probability: location and noise.
    tau = pm.Beta("tau", 2, 2)
    gain = pm.LogNormal("gain", mu=np.log(8.0), sigma=0.7)

    rule_log_prior = pt.log(lambda_0) + pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])
    log_rules = rule_log_prior[None, :] + rule_log_lik  # (sets, rules)
    log_int = (pt.log1p(-lambda_0) - 2 * np.log(100) + interval_log_marg)[:, None]
    log_all = pt.concatenate([log_rules, log_int], axis=1)
    post = pt.exp(log_all - pt.logsumexp(log_all, axis=1, keepdims=True))
    membership = post[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + post[:, N_RULES:] * interval_pred
    q = membership[row_set, row_target]

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(gain * (q - tau)), 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n = len(unique)
    rule_ll = np.zeros((n, N_RULES))
    int_marg = np.zeros(n)
    int_pred = np.zeros((n, 100))
    for i, s in enumerate(unique):
        cols = np.array(s) - 1
        rule_ll[i] = RULE_EX_LL[:, cols].sum(axis=1)
        ll = INTERVAL_EX_LL[:, cols].sum(axis=1)
        m = ll.max()
        w = np.exp(ll - m)
        int_marg[i] = m + np.log(w.sum())
        int_pred[i] = (w / w.sum()) @ INTERVAL_MEMBERSHIP
    return {
        "rule_log_lik": rule_ll,
        "interval_log_marg": int_marg,
        "interval_pred": int_pred,
        "row_set": np.array([index[s] for s in sets], dtype="int64"),
        "row_target": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
