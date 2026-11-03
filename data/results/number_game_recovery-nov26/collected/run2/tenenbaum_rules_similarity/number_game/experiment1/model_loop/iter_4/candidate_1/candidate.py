"""Refinement of noisy_examples_erlang_intervals: people do the same Bayesian
concept learning over 45 independently weighted rules and every interval of
1..100 (size principle, an inferred share of examples believed random, an
Erlang prior favouring a typical range width), but they think of ranges in
round, decade-aligned terms -- "from 20 to 40", "the 60s", "1 to 10" -- so a
range whose endpoints fall on decade boundaries is a priori more plausible
than an equally wide range with arbitrary endpoints such as 23-47. The one
change: each interval's prior is multiplied by exp(round_bonus * r), where r
(0, 1 or 2) counts its endpoints that sit on a decade boundary (lower end
ending in 0 or 1, upper end ending in 0 or 9), with round_bonus inferred
(zero recovers the incumbent). "Yes" is the posterior probability that the
concept contains the target, so generalisation from examples that fit no rule
tends to stop at decade lines.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_EX = 4


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
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)


def roundness(lo, hi):
    """Number of decade-aligned endpoints of the interval [lo, hi]."""
    return int(lo % 10 in (0, 1)) + int(hi % 10 in (0, 9))


# Intervals are grouped by (size, examples inside) as in the incumbent; within a
# group they are further counted by roundness r, whose prior factor
# exp(round_bonus * r) is applied to those counts inside the model.
N_R = 3
N_GROUPS = 100 * (MAX_EX + 1)
GROUP_SIZE = np.repeat(np.arange(1, 101), MAX_EX + 1).astype("float64")
GROUP_NIN = np.tile(np.arange(MAX_EX + 1), 100).astype("float64")
SIZES_F = np.arange(1, 101, dtype="float64")
# How many intervals of each size have each roundness, for normalising the prior.
N_SIZE_R = np.zeros((100, N_R))
for _lo in range(1, 101):
    for _hi in range(_lo, 101):
        N_SIZE_R[_hi - _lo, roundness(_lo, _hi)] += 1
R_LEVELS = np.arange(N_R, dtype="float64")

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))  # (sets, rules)
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    group_count = pm.Data("group_count", np.zeros((1, N_GROUPS, N_R)))  # (sets, groups, r)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    # Intervals of each (group, r) that contain the cell's target, flattened to (cells * groups, r).
    cell_group_count = pm.Data("cell_group_count", np.zeros((N_GROUPS, N_R)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    noise = pm.Beta("noise", 1.5, 6.0)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    # Preference for decade-aligned endpoints (log prior gain per round endpoint).
    round_bonus = pm.Normal("round_bonus", mu=0.0, sigma=1.0)
    log_erlang = pt.log(SIZES_F) - SIZES_F / sigma  # (sizes,)
    r_gain = pt.exp(round_bonus * R_LEVELS)  # (r,)
    # Log prior of one interval with roundness 0, per size, normalised over all intervals.
    log_base = log_erlang - pt.log((pt.exp(log_erlang - log_erlang.max())[:, None] * N_SIZE_R * r_gain).sum()) + log_erlang.max()

    log_out = pt.log(noise / 100.0)
    rule_log_in = pt.log((1 - noise) / RULE_SIZE + noise / 100.0)
    rule_log_lik = rule_n_in * rule_log_in[None, :] + (n_examples[:, None] - rule_n_in) * log_out
    group_log_in = pt.log((1 - noise) / GROUP_SIZE + noise / 100.0)
    group_log_lik = (GROUP_NIN * group_log_in)[None, :] + (n_examples[:, None] - GROUP_NIN[None, :]) * log_out
    # Log joint of a single roundness-0 interval in each group.
    log_unit = (pt.log1p(-lambda_0) + pt.repeat(log_base, MAX_EX + 1))[None, :] + group_log_lik
    weighted_count = pt.dot(group_count, r_gain)  # (sets, groups)
    log_groups = log_unit + pt.log(weighted_count + 1e-300)
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_z = pt.logsumexp(pt.concatenate([log_rules, log_groups], axis=1), axis=1, keepdims=True)
    rule_post = pt.exp(log_rules - log_z)  # (sets, rules)
    unit_post = pt.exp(log_unit - log_z)  # (sets, groups): one roundness-0 interval
    in_concept = (rule_post[cell_set] * cell_rule_member).sum(axis=1) + (
        unit_post[cell_set] * pt.dot(cell_group_count, r_gain).reshape((cell_set.shape[0], N_GROUPS))
    ).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set`, `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_n_in = np.zeros((S, N_RULES))
    n_ex = np.zeros(S)
    counts = np.zeros((S, N_GROUPS, N_R))
    contain = np.zeros((S, N_GROUPS, N_R, 100))
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        n_ex[i] = len(s)
        rule_n_in[i] = RULE_MEMBERSHIP[:, ex].sum(axis=1)
        for lo in range(1, 101):
            for hi in range(lo, 101):
                k = int(sum(lo <= x <= hi for x in s))
                g, r = (hi - lo) * (MAX_EX + 1) + k, roundness(lo, hi)
                counts[i, g, r] += 1
                contain[i, g, r, lo - 1:hi] += 1
    keys = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {k: i for i, k in enumerate(cells)}
    cs = np.array([k[0] for k in cells], dtype="int64")
    ct = np.array([k[1] for k in cells], dtype="int64")
    return {
        "rule_n_in": rule_n_in,
        "n_examples": n_ex,
        "group_count": counts,
        "cell_set": cs,
        "cell_rule_member": RULE_MEMBERSHIP[:, ct].T.copy(),
        "cell_group_count": contain[cs, :, :, ct].reshape(-1, N_R),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
