"""Refinement of bigelow_piantadosi_independent: Bayesian concept learning over
45 rules and all intervals of 1..100 (size principle, 10% uniform noise; "yes"
is the posterior probability the concept contains the target), with one
change: an interval's prior weight decays exponentially with its width at an
inferred rate kappa (kappa > 0 favours tight ranges around the examples), so
generalization within the example span is stronger than outside it.
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
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
INTERVAL_SIZE = np.array([len(c) for c in INTERVALS])
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
SIZES = np.arange(1, 101, dtype="float64")
N_PER_SIZE = 101.0 - SIZES  # number of intervals of each width

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

# Intervals of one width share a prior, so per example set they are grouped by
# width: log evidence per width and, per width, the share of that evidence
# from intervals containing each target (precomputed in prepare_observed).

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    width_log_evidence = pm.Data("width_log_evidence", np.zeros((1, 100)))  # (sets, widths)
    width_in_concept = pm.Data("width_in_concept", np.zeros((1, 100, 100)))  # (sets, widths, targets)
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
    # Width decay rate (per 100 numbers); 0 recovers a uniform interval prior.
    kappa = pm.Normal("kappa", mu=0.0, sigma=5.0)

    log_q = -kappa * SIZES / 100.0  # unnormalised log prior of one interval of each width
    log_q = log_q - pt.logsumexp(log_q + np.log(N_PER_SIZE))

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_widths = pt.log1p(-lambda_0) + log_q[None, :] + width_log_evidence
    log_joint = pt.concatenate([log_rules, log_widths], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + (
        posterior[:, N_RULES:, None] * width_in_concept).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_ll = set_log_lik[:, N_RULES:]
    interval_mem = MEMBERSHIP[N_RULES:]
    n_sets = len(unique)
    w_ev = np.zeros((n_sets, 100))
    w_in = np.zeros((n_sets, 100, 100))
    for k in range(100):
        sel = INTERVAL_SIZE == k + 1
        ll = interval_ll[:, sel]
        ev = logsumexp(ll, axis=1)
        w_ev[:, k] = ev
        w_in[:, k, :] = np.exp(ll - ev[:, None]) @ interval_mem[sel]
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "width_log_evidence": w_ev,
        "width_in_concept": w_in,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
