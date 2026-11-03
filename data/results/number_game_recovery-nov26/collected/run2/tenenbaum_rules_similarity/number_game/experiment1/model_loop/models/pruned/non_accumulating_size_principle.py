"""Non-accumulating size principle.

People do Bayesian concept learning over the 45 rules (odd, even, primes,
squares, cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of
3-9, 11 and 12, ending in 0-9, containing the digit 0-9) and every interval of
1-100, but they do not accumulate size-principle evidence across examples: a
concept's fit to the set is the average per-example log-likelihood times one
free sharpness beta, the same whatever the number of examples. So
generalisation is no tighter after four examples than after one (the standard
model's exponent is the number of examples). Priors as in the Independent
model: lambda_0 for rules as a group, a weight for each rule, intervals
(1 - lambda_0) / 100^2. Examples are 90% from the concept, 10% uniform.
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
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
LOG_IN = np.log(ALPHA / NUMBERS + (1 - ALPHA) / 100)  # one in-concept example, interval of size s = 1..100
LOG_OUT = np.log((1 - ALPHA) / 100)
N_GROUPS = 100 * (MAX_N + 1)  # intervals grouped by (size, examples inside)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

# Intervals with the same size s and number k of examples inside have the same
# likelihood, so they are pooled: per set, log(count) of each (s, k) group and
# the share of the group's intervals that contain each target.
with pm.Model() as model:
    rule_mean_ll = pm.Data("rule_mean_ll", np.zeros((1, N_RULES)))  # mean per-example log P(x | rule)
    group_k = pm.Data("group_k", np.zeros((1, N_GROUPS)))  # fraction of examples inside, k / n
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, N_GROUPS, 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    beta = pm.LogNormal("beta", mu=np.log(2.5), sigma=0.5)  # sharpness of the (non-accumulating) fit
    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ])

    size_in = pt.as_tensor(np.repeat(LOG_IN, MAX_N + 1))[None, :]
    group_mean_ll = group_k * size_in + (1 - group_k) * LOG_OUT
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + beta * rule_mean_ll
    log_groups = pt.log1p(-lambda_0) - 2 * np.log(100) + group_log_count + beta * group_mean_ll
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_rules = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)
    in_intervals = pt.batched_dot(posterior[:, N_RULES:], group_in_concept)
    in_concept = in_rules + in_intervals

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_ll = np.zeros((S, N_RULES))
    gk = np.tile(np.tile(np.arange(MAX_N + 1), 100), (S, 1)).astype("float64")
    counts = np.zeros((S, N_GROUPS))
    inside = np.zeros((S, N_GROUPS, 100))
    lo = NUMBERS[:, None]
    hi = NUMBERS[None, :]
    valid = lo <= hi
    size = (hi - lo + 1)
    for i, s in enumerate(unique):
        ex = np.array(s)
        n = len(ex)
        rule_ll[i] = RULE_LOG_LIK[:, ex - 1].mean(axis=1)
        gk[i] = gk[i] / n
        k = ((ex[None, None, :] >= lo[..., None]) & (ex[None, None, :] <= hi[..., None])).sum(axis=2)
        g = (size - 1) * (MAX_N + 1) + k
        m_idx, n_idx = np.nonzero(valid)
        gg = g[m_idx, n_idx]
        np.add.at(counts[i], gg, 1.0)
        # targets t in [m, n]: cumulative trick per interval
        diff = np.zeros((N_GROUPS, 102))
        np.add.at(diff, (gg, m_idx), 1.0)
        np.add.at(diff, (gg, n_idx + 1), -1.0)
        inside[i] = np.cumsum(diff, axis=1)[:, :100]
    safe = np.maximum(counts, 1.0)
    return {
        "rule_mean_ll": rule_ll,
        "group_k": gk,
        "group_log_count": np.where(counts > 0, np.log(safe), -1e3),
        "group_in_concept": inside / safe[..., None],
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
