"""Refinement of bigelow_piantadosi_independent: Bayesian concept learning
over the same 45 rules and the 5,050 intervals of 1..100, with a prior weight
inferred for each rule and a share lambda_0 for the rules (each interval has
prior (1 - lambda_0) / 100^2), but with a partial size principle. Each example
is in-concept with probability 0.9 and then has likelihood |h|^(-gamma), else
it is uniform on 1..100 (1/100). gamma = 1 is the incumbent's strong sampling;
gamma = 0 is weak sampling, where concept size does not matter. "Yes" is the
posterior probability that the concept contains the target. The one change is
the inferred size-principle strength gamma.
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
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
RULE_LOG_SIZE = np.log(RULE_MEMBERSHIP.sum(axis=1))
# Intervals share one prior, and an interval's likelihood depends only on its
# size L and the number k of examples inside it, so for each example set the
# intervals are grouped exactly into cells (L, k): the count of intervals in
# each cell and, per target, how many of them contain it.
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_SIZE = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
MAX_K = 5
N_CELLS = 100 * MAX_K
CELL_LOG_SIZE = np.log(np.repeat(np.arange(1, 101), MAX_K).astype("float64"))
CELL_K = np.tile(np.arange(MAX_K), 100).astype("float64")
_IV_ROWS, _IV_COLS = np.nonzero(INTERVAL_MEMBERSHIP)

# "ends in 0" and "contains digit 0" are the same set; sample their summed
# weight and a Beta(2, 2) split (equivalent to Dirichlet(2, ..., 2) on all 45).
END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))  # (sets, rules): examples inside each rule
    n_examples = pm.Data("n_examples", np.zeros(1))  # (sets,)
    cell_log_count = pm.Data("cell_log_count", np.zeros((1, N_CELLS)))  # log #intervals per (L, k); -1e4 if none
    cell_in_concept = pm.Data("cell_in_concept", np.zeros((1, N_CELLS, 100)))  # share of a cell's intervals holding each target
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
    # Size-principle strength: 1 = strong sampling, 0 = weak sampling.
    gamma = pm.Beta("gamma", 2, 2)

    log_out = np.log((1 - ALPHA) / 100)
    rule_log_in = pt.log(ALPHA * pt.exp(-gamma * RULE_LOG_SIZE) + (1 - ALPHA) / 100)
    cell_log_in = pt.log(ALPHA * pt.exp(-gamma * CELL_LOG_SIZE) + (1 - ALPHA) / 100)
    log_rules = (pt.log(lambda_0) + pt.log(rule_weights)[None, :]
                 + rule_n_in * rule_log_in[None, :] + (n_examples[:, None] - rule_n_in) * log_out)
    log_cells = (pt.log1p(-lambda_0) - 2 * np.log(100) + cell_log_count
                 + CELL_K[None, :] * cell_log_in[None, :] + (n_examples[:, None] - CELL_K[None, :]) * log_out)
    log_joint = pt.concatenate([log_rules, log_cells], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, rules + cells)
    in_concept = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)
                  + (posterior[:, N_RULES:, None] * cell_in_concept).sum(axis=1))

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    rule_n_in, log_counts, in_cells = [], [], []
    for s in unique:
        k = INTERVAL_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1).astype("int64")
        cell = (INTERVAL_SIZE - 1) * MAX_K + k
        count = np.bincount(cell, minlength=N_CELLS).astype("float64")
        holds = np.bincount(cell[_IV_ROWS] * 100 + _IV_COLS, minlength=N_CELLS * 100).reshape(N_CELLS, 100)
        rule_n_in.append(RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1))
        log_counts.append(np.where(count > 0, np.log(np.maximum(count, 1.0)), -1e4))
        in_cells.append(holds / np.maximum(count, 1.0)[:, None])
    return {
        "rule_n_in": np.stack(rule_n_in),
        "n_examples": np.array([len(s) for s in unique], dtype="float64"),
        "cell_log_count": np.stack(log_counts),
        "cell_in_concept": np.stack(in_cells),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
