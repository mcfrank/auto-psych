"""Concept learning with magnitude intervals on a log-compressed mental number line.

People represent numerical-magnitude concepts as regions of a logarithmically
compressed mental number line: an interval's subjective size is its log
extent, and examples drawn from it fall uniformly in compressed units (number
x carries weight log((x+.5)/(x-.5)), roughly 1/x). Arithmetic and digit rules
(the 45 rules of the mixture model) are sampled uniformly as before, with a
fitted size-principle temper, exception rate and rule prior. Because the same
step is subjectively smaller among large numbers, intervals are extended
upward past the largest example more readily than downward past the smallest.
"Yes" is the posterior probability that the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.sparse as ps
import pytensor.tensor as pt

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


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)

# Weight of each number on the compressed line, and every interval [lo, hi].
LOG_WEIGHT = np.log((NUMBERS + 0.5) / (NUMBERS - 0.5))
INTERVALS = np.array([(lo, hi) for lo in range(1, 101) for hi in range(lo, 101)])
N_INT = len(INTERVALS)
INT_LOG_SIZE = np.log((INTERVALS[:, 1] + 0.5) / (INTERVALS[:, 0] - 0.5))
CONTAINS = ((NUMBERS[:, None] >= INTERVALS[None, :, 0]) & (NUMBERS[:, None] <= INTERVALS[None, :, 1])).astype("float32")
DENS = (LOG_WEIGHT[:, None] / INT_LOG_SIZE[None, :]).astype("float32")  # (numbers, intervals)
F32 = "float32"  # the interval arrays are large; single precision halves the cost

with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples inside the rule
    set_n = pm.Data("set_n", np.ones(1))  # (sets,): number of examples
    set_count = pm.Data("set_count", np.zeros((1, 100)))  # (sets, numbers): times shown
    int_k = pm.Data("int_k", np.zeros((1, N_INT)))  # (sets, intervals): examples inside
    row_flat = pm.Data("row_flat", np.zeros(1, dtype="int64"))  # set * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)  # sharpness of the size principle
    eps = pm.Beta("eps", 2, 5)  # believed rate of off-concept examples

    log_noise = pt.log(eps) - np.log(100)
    n_col = set_n[:, None]
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)
    rule_log_lik = rule_k * rule_in[None, :] + (n_col - rule_k) * log_noise

    # Inside interval h, number x has probability (1-eps) w_x / W_h + eps/100.
    eps32 = pt.cast(eps, F32)
    in_term = CONTAINS * pt.log((1 - eps32) * DENS + eps32 / 100)  # (numbers, intervals)
    counts = ps.csr_from_dense(pt.cast(set_count, F32))
    int_log_lik = ps.structured_dot(counts, in_term) + pt.cast(n_col - int_k, F32) * pt.cast(log_noise, F32)

    log_rules = pt.log(lambda_0) - np.log(N_RULES) + beta * rule_log_lik
    log_ints = pt.cast(pt.log1p(-lambda_0) - np.log(N_INT), F32) + pt.cast(beta, F32) * int_log_lik
    log_joint = pt.concatenate([pt.cast(log_rules, F32), log_ints], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    # (sets, numbers): posterior probability that the concept contains each number.
    contains = pt.dot(posterior[:, :N_RULES], RULE_MEMBERSHIP.astype(F32)) + pt.dot(posterior[:, N_RULES:], CONTAINS.T)
    in_concept = pt.cast(contains, "float64").flatten()[row_flat]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_count = np.zeros((len(unique), 100))
    for i, s in enumerate(unique):
        for x in s:
            set_count[i, x - 1] += 1
    return {
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "set_count": set_count,
        "int_k": set_count @ CONTAINS,
        "row_flat": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
