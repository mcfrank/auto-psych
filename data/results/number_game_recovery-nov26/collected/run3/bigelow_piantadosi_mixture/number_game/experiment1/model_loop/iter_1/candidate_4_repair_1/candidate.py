"""Refinement of `bigelow_piantadosi_independent`: the same Bayesian concept
learner over 45 rules (a prior weight inferred per rule, lambda_0 for rules as
a group) and all 5,050 intervals of 1..100, with the size principle (10% of
examples uniform on 1..100), but people think about magnitude in decades:
intervals aligned to decade boundaries ([10a + 1, 10b], e.g. 61-70, 1-20) get
an inferred multiplicative prior boost exp(gamma) over arbitrary ranges, the
intervals' total prior mass (1 - lambda_0) * 5050 / 100^2 held fixed
(gamma = 0 recovers the incumbent). "Yes" is the posterior probability that
the concept contains the target.

Intervals split into two groups (aligned, not aligned), so the interval
evidence and the share of it containing each target are precomputed exactly
per group in prepare_observed; gamma only reweights the two groups.
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
BOUNDS = [(m, n) for m in NUMBERS for n in range(m, 101)]
INTERVALS = [list(range(m, n + 1)) for m, n in BOUNDS]
ALIGNED = np.array([m % 10 == 1 and n % 10 == 0 for m, n in BOUNDS])
N_ALIGNED, N_OTHER = int(ALIGNED.sum()), int((~ALIGNED).sum())
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]

# Same parameterisation of the rule weights as the incumbent: "ends in 0" and
# "contains digit 0" are one set, sampled as one weight and a Beta(2, 2) split.
END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules): log P(D | rule)
    aligned_log_evidence = pm.Data("aligned_log_evidence", np.zeros(1))  # (sets,)
    other_log_evidence = pm.Data("other_log_evidence", np.zeros(1))  # (sets,)
    aligned_in_concept = pm.Data("aligned_in_concept", np.zeros(1))  # (trials,)
    other_in_concept = pm.Data("other_in_concept", np.zeros(1))  # (trials,)
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    # Log prior boost of a decade-aligned interval over an arbitrary one.
    gamma = pm.Normal("gamma", mu=0.0, sigma=3.0)

    log_norm = np.log(N_ALIGNED + N_OTHER) - pt.log(N_ALIGNED * pt.exp(gamma) + N_OTHER)
    group_log = pt.stack([gamma + aligned_log_evidence, other_log_evidence + pt.zeros_like(gamma)], axis=1)
    interval_log_evidence = pt.logsumexp(group_log, axis=1) + log_norm  # (sets,)
    group_share = pt.exp(group_log - pt.logsumexp(group_log, axis=1, keepdims=True))  # (sets, 2)

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + interval_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, rules + 1)
    rule_in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)  # (sets, 100)
    interval_in_concept = (group_share[row_set, 0] * aligned_in_concept
                           + group_share[row_set, 1] * other_in_concept)
    in_concept = rule_in_concept.reshape((-1,))[row_cell] + posterior[row_set, N_RULES] * interval_in_concept

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_log_lik = set_log_lik[:, N_RULES:]
    row_set = np.array([index[s] for s in sets], dtype="int64")
    row_target = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    out = {}
    for name, mask in (("aligned", ALIGNED), ("other", ~ALIGNED)):
        group = interval_log_lik[:, mask]
        log_z = logsumexp(group, axis=1)
        share = np.exp(group - log_z[:, None]) @ INTERVAL_MEMBERSHIP[mask]  # (sets, 100)
        out[f"{name}_log_evidence"] = log_z
        out[f"{name}_in_concept"] = share[row_set, row_target]
    out.update({
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "row_set": row_set,
        "row_cell": row_set * 100 + row_target,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    })
    return out
