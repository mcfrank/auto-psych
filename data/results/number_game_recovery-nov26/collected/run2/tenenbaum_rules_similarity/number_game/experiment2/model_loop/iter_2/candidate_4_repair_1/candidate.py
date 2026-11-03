"""A refinement of extended_multiples_concepts: the same Bayesian concept learning
(45 independently weighted rules plus multiples of 13..50, intervals of 1..100 with
an Erlang width prior, an inferred share of random examples, examples drawn without
replacement), but people believe the program shows a concept's members with
probability proportional to magnitude**gamma (gamma inferred; gamma = 0 recovers the
incumbent). With gamma > 0 small members are rarely shown, so concepts stretching
down toward small numbers cost little under the size principle and "yes" extends
below the smallest example more readily than above the largest. "Yes" is the
posterior probability that the concept contains the target.
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
N_BASE = len(_RULES)
EXT_M = list(range(13, 51))  # less-familiar multiples rules
N_EXT = len(EXT_M)
_RULES.update({f"multiples of {m}": list(range(m, 101, m)) for m in EXT_M})
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")

# Every interval [a, b] of 1..100 (0-based ends) and the numbers it contains.
INT_A = np.array([a for a in range(100) for b in range(a, 100)])
INT_B = np.array([b for a in range(100) for b in range(a, 100)])
N_INT = len(INT_A)  # 5050
INT_MEMBERSHIP = ((NUMBERS[None, :] - 1 >= INT_A[:, None]) & (NUMBERS[None, :] - 1 <= INT_B[:, None])).astype("float64")
INT_ID = {(a, b): q for q, (a, b) in enumerate(zip(INT_A, INT_B))}
SIZES_F = np.arange(1, 101).astype("float64")
N_PER_SIZE = 101.0 - SIZES_F
LOG_MAG = np.log(NUMBERS / 100.0)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_BASE - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    ex_value = pm.Data("ex_value", np.zeros((1, MAX_EX), dtype="int64"))  # (sets, 4) value - 1
    ex_mask = pm.Data("ex_mask", np.zeros((1, MAX_EX)))
    ex_in_rule = pm.Data("ex_in_rule", np.zeros((1, MAX_EX, N_RULES)))
    # Sparse list of (set, interval, example) where the example lies in the interval.
    ent_pos = pm.Data("ent_pos", np.zeros(1, dtype="int64"))  # set * N_INT + interval
    ent_int = pm.Data("ent_int", np.zeros(1, dtype="int64"))  # interval
    ent_ex = pm.Data("ent_ex", np.zeros(1, dtype="int64"))  # set * 4 + example
    ent_prev = pm.Data("ent_prev", np.zeros(1, dtype="int64"))  # set * 8 + subset of earlier examples in interval
    subset_mask = pm.Data("subset_mask", np.zeros((1, 8, MAX_EX)))  # (sets, subsets, examples)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
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
    ext_share = pm.Beta("ext_share", 1.0, 4.0)
    all_weights = pt.concatenate([
        (1 - ext_share) * rule_weights, pt.ones(N_EXT) * ext_share / N_EXT,
    ])

    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    log_erlang = pt.log(SIZES_F) - SIZES_F / sigma
    log_per_interval = log_erlang - pt.logsumexp(log_erlang + np.log(N_PER_SIZE))

    # The one change: the program shows a concept's member n with weight n**gamma.
    gamma = pm.Normal("gamma", mu=0.0, sigma=1.0)
    w = pt.exp(gamma * LOG_MAG)  # (100,)
    cum = pt.concatenate([pt.zeros(1), pt.cumsum(w)])
    w_int = cum[INT_B + 1] - cum[INT_A]  # (intervals,) total weight of each interval
    w_rule = pt.dot(RULE_MEMBERSHIP, w)  # (rules,)
    log_out = pt.log(noise / 100.0)

    def draw_log_prob(w_this, total_unseen):
        """Without replacement: an in-concept draw among the members not yet shown."""
        return pt.log((1 - noise) * w_this / pt.maximum(total_unseen, w_this) + noise / 100.0)

    # Rules (dense).
    w_ex = w[ex_value]  # (sets, 4)
    seen = pt.zeros_like(ex_in_rule[:, 0, :])
    rule_log_lik = n_examples[:, None] * log_out
    for j in range(MAX_EX):
        inj = ex_mask[:, j][:, None] * ex_in_rule[:, j, :]
        wj = w_ex[:, j][:, None]
        rule_log_lik = rule_log_lik + inj * (draw_log_prob(wj, w_rule[None, :] - seen) - log_out)
        seen = seen + inj * wj
    # Intervals (sparse over the examples each interval contains).
    n_sets = n_examples.shape[0]
    subset_w = (subset_mask * w_ex[:, None, :]).sum(axis=2).ravel()  # (sets * 8,)
    ent_term = draw_log_prob(w_ex.ravel()[ent_ex], w_int[ent_int] - subset_w[ent_prev]) - log_out
    int_log_lik = pt.inc_subtensor(pt.zeros(n_sets * N_INT)[ent_pos], ent_term).reshape((n_sets, N_INT))
    int_log_lik = int_log_lik + n_examples[:, None] * log_out

    log_rules = pt.log(lambda_0) + pt.log(all_weights)[None, :] + rule_log_lik
    log_ints = (pt.log1p(-lambda_0) + log_per_interval[INT_B - INT_A])[None, :] + int_log_lik
    log_joint = pt.concatenate([log_rules, log_ints], axis=1)
    posterior = pt.special.softmax(log_joint, axis=1)
    yes_table = pt.dot(posterior[:, :N_RULES], RULE_MEMBERSHIP) + pt.dot(posterior[:, N_RULES:], INT_MEMBERSHIP)
    in_concept = yes_table[cell_set, cell_target]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set`, `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    n_ex = np.zeros(S)
    ex_value = np.zeros((S, MAX_EX), dtype="int64")
    ex_mask = np.zeros((S, MAX_EX))
    ex_in_rule = np.zeros((S, MAX_EX, N_RULES))
    subset_mask = np.zeros((S, 8, MAX_EX))
    for c in range(8):
        subset_mask[:, c, :3] = [(c >> k) & 1 for k in range(3)]
    pos, ent_int, ent_ex, ent_prev = [], [], [], []
    for i, s in enumerate(unique):
        xs = [x - 1 for x in s[:MAX_EX]]
        n_ex[i] = len(xs)
        for j, x in enumerate(xs):
            ex_value[i, j] = x
            ex_mask[i, j] = 1.0
            ex_in_rule[i, j] = RULE_MEMBERSHIP[:, x]
            for a in range(x + 1):
                for b in range(x, 100):
                    q = INT_ID[(a, b)]
                    pos.append(i * N_INT + q)
                    ent_int.append(q)
                    ent_ex.append(i * MAX_EX + j)
                    ent_prev.append(i * 8 + sum(1 << k for k in range(j) if a <= xs[k] <= b))
    keys = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {k: i for i, k in enumerate(cells)}
    return {
        "n_examples": n_ex,
        "ex_value": ex_value,
        "ex_mask": ex_mask,
        "ex_in_rule": ex_in_rule,
        "ent_pos": np.array(pos, dtype="int64"),
        "ent_int": np.array(ent_int, dtype="int64"),
        "ent_ex": np.array(ent_ex, dtype="int64"),
        "ent_prev": np.array(ent_prev, dtype="int64"),
        "subset_mask": subset_mask,
        "cell_set": np.array([k[0] for k in cells], dtype="int64"),
        "cell_target": np.array([k[1] for k in cells], dtype="int64"),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
