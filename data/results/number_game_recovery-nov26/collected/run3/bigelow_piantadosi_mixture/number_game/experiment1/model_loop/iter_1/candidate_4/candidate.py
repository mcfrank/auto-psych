"""Refinement of `bigelow_piantadosi_independent`: the same Bayesian concept
learner over 45 rules (a prior weight inferred per rule, lambda_0 for rules as
a group) and all 5,050 intervals of 1..100 (each with prior
(1 - lambda_0) / 100^2), but the examples' likelihood under each concept (the
size principle, 10% of examples uniform on 1..100) is raised to an inferred
exponent beta: people weigh the evidence of the examples more weakly (beta < 1)
or more strongly (beta > 1) than an ideal strong sampler. "Yes" is the
tempered posterior probability that the concept contains the target.

The exponent changes the relative evidence of the intervals, so they no longer
collapse into one fixed hypothesis. Their summed tempered evidence and the
share of it from intervals containing each target are parameter-free given
beta; prepare_observed tabulates both on a fine grid of log beta (step 0.025)
and the model interpolates linearly between grid points. Rules are exact.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
LOG_BETA_GRID = np.linspace(-2.5, 2.5, 201)
GRID_STEP = LOG_BETA_GRID[1] - LOG_BETA_GRID[0]


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
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]

# Same parameterisation of the rule weights as the incumbent: "ends in 0" and
# "contains digit 0" are one set, sampled as one weight and a Beta(2, 2) split.
END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules): log P(D | rule)
    interval_log_evidence_grid = pm.Data("interval_log_evidence_grid", np.zeros((1, 1)))  # (grid, sets)
    interval_in_concept_grid = pm.Data("interval_in_concept_grid", np.zeros((1, 1)))  # (grid, trials)
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
    # Evidence-weighting exponent on the size-principle likelihood (1 = incumbent).
    log_beta = pm.Normal("log_beta", mu=0.0, sigma=0.5)
    beta = pm.Deterministic("beta", pt.exp(log_beta))

    position = (pt.clip(log_beta, LOG_BETA_GRID[0], LOG_BETA_GRID[-1] - 1e-9) - LOG_BETA_GRID[0]) / GRID_STEP
    lo = pt.cast(pt.floor(position), "int64")
    frac = position - lo
    interval_log_evidence = (1 - frac) * interval_log_evidence_grid[lo] + frac * interval_log_evidence_grid[lo + 1]
    interval_in_concept = (1 - frac) * interval_in_concept_grid[lo] + frac * interval_in_concept_grid[lo + 1]

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + beta * rule_log_lik
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + interval_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, rules + 1)
    rule_in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)  # (sets, 100)
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
    evidence, in_concept = [], []
    for beta in np.exp(LOG_BETA_GRID):
        tempered = beta * interval_log_lik
        log_z = logsumexp(tempered, axis=1)
        share = np.exp(tempered - log_z[:, None]) @ MEMBERSHIP[N_RULES:]  # (sets, 100)
        evidence.append(log_z)
        in_concept.append(share[row_set, row_target])
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "interval_log_evidence_grid": np.stack(evidence),
        "interval_in_concept_grid": np.stack(in_concept),
        "row_set": row_set,
        "row_cell": row_set * 100 + row_target,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
