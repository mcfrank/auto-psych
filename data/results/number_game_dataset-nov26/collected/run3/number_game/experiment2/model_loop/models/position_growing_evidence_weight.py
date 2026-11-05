"""Position-growing evidence weight: a running guess that sharpens with each example.

People revise a running guess about the concept as each example arrives. The
first example only seeds a broad set of candidate concepts (45 individually
weighted rules and all intervals of 1-100, size principle, 10% noise) and
counts as weak evidence; each later example is weighed more heavily as
confirmation. The k-th example shown has evidence weight beta * k**growth
(growth = 0 recovers a fixed tempered learner), with no renormalisation, so a
lone example is weak evidence (broad generalization) and later examples
dominate (order matters).
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


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    pos_ll = np.zeros((S, MAX_N, N_RULES + N_INT))
    log_pos = np.zeros((S, MAX_N))
    pos_mask = np.zeros((S, MAX_N))
    for i, s in enumerate(unique):
        for k, x in enumerate(s):
            pos_ll[i, k] = EXAMPLE_LOG_LIK[:, x - 1]
            pos_mask[i, k] = 1.0
            log_pos[i, k] = np.log(k + 1.0)
    return {
        "rule_pos_ll": pos_ll[:, :, :N_RULES],
        "interval_pos_ll": pos_ll[:, :, N_RULES:],
        "log_position": log_pos,
        "position_mask": pos_mask,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    rule_pos_ll = pm.Data("rule_pos_ll", np.zeros((1, MAX_N, N_RULES)))
    interval_pos_ll = pm.Data("interval_pos_ll", np.zeros((1, MAX_N, N_INT)))
    log_position = pm.Data("log_position", np.zeros((1, MAX_N)))
    position_mask = pm.Data("position_mask", np.ones((1, MAX_N)))
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
    log_beta = pm.Normal("log_beta", mu=0.0, sigma=0.75)  # evidence weight of the first example
    growth = pm.Normal("growth", mu=0.0, sigma=1.0)       # the mechanism: weight grows as k**growth

    pos_w = pt.exp(log_beta + growth * log_position) * position_mask  # (sets, MAX_N)
    rule_ll = (pos_w[:, :, None] * rule_pos_ll).sum(axis=1)
    t_int = (pos_w[:, :, None] * interval_pos_ll).sum(axis=1)
    int_log_evidence = pt.logsumexp(t_int, axis=1)
    int_post = pt.exp(t_int - int_log_evidence[:, None])
    interval_in_concept = pt.dot(int_post, pt.as_tensor(INTERVAL_MEMBERSHIP))

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_ll
    log_intervals = pt.log1p(-lambda_0) - np.log(N_INT) + int_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (pt.dot(posterior[:, :N_RULES], pt.as_tensor(RULE_MEMBERSHIP))
                  + posterior[:, N_RULES][:, None] * interval_in_concept)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
