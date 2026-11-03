"""Refinement of bigelow_piantadosi_independent: Bayesian concept learning over
45 rules and all intervals of 1..100, a prior weight per rule and a share
lambda_0 for rules (each interval (1 - lambda_0) / 100^2), "yes" = posterior
probability that the concept contains the target. The single change: the
probability alpha that an example was drawn from the concept (size principle)
rather than uniformly from 1..100 is a free parameter inferred from the data,
not fixed at 0.9.
"""

import numpy as np
import pymc as pm
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


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
SIZES = MEMBERSHIP.sum(axis=1)

# Same Dirichlet aggregation trick as the incumbent for the identical sets
# "ends in 0" and "contains digit 0".
END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

# Intervals of equal size L holding equally many examples k have the same prior
# and likelihood for every alpha, so for each example set prepare_observed
# groups the 5,050 intervals into (L, k) classes, with each class's count and
# summed membership over targets. The posterior is exactly the one over all
# 5,095 hypotheses, at a fraction of the cost per gradient.
MAX_K = 5
CLASS_SIZE = np.repeat(np.arange(1, 101), MAX_K).astype("float64")  # L per class
CLASS_K = np.tile(np.arange(MAX_K), 100).astype("float64")  # k per class
N_CLASSES = CLASS_SIZE.size
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_SIZES = SIZES[N_RULES:].astype("int64")

with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))  # (sets, rules): examples inside each rule
    log_class_count = pm.Data("log_class_count", np.zeros((1, N_CLASSES)))  # (sets, classes)
    # Membership only for the (set, target) cells actually asked about.
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,): set index
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_class_member = pm.Data("cell_class_member", np.zeros((1, N_CLASSES)))  # (cells, classes): mean membership
    n_examples = pm.Data("n_examples", np.zeros(1))  # (sets,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,): cell index
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    alpha = pm.Beta("alpha", 4, 1)  # share of examples drawn from the concept
    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    # log P(D | h) = k log(alpha/|h| + (1-alpha)/100) + (n-k) log((1-alpha)/100)
    log_out = pt.log1p(-alpha) - np.log(100.0)
    rule_log_in = pt.log(alpha / SIZES[:N_RULES] + (1 - alpha) / 100.0)
    log_rules = (pt.log(lambda_0) + pt.log(rule_weights)[None, :]
                 + rule_n_in * rule_log_in[None, :] + (n_examples[:, None] - rule_n_in) * log_out)
    class_log_in = pt.log(alpha / CLASS_SIZE + (1 - alpha) / 100.0)
    log_classes = (pt.log1p(-lambda_0) - 2 * np.log(100.0) + log_class_count
                   + CLASS_K[None, :] * class_log_in[None, :] + (n_examples[:, None] - CLASS_K[None, :]) * log_out)
    log_joint = pt.concatenate([log_rules, log_classes], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, rules + classes)
    cell_posterior = posterior[cell_set]  # (cells, rules + classes)
    in_concept = ((cell_posterior[:, :N_RULES] * cell_rule_member).sum(axis=1)
                  + (cell_posterior[:, N_RULES:] * cell_class_member).sum(axis=1))  # (cells,)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_in = np.stack([MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]).astype("int64")
    counts = np.zeros((len(unique), N_CLASSES))
    mem_sum = np.zeros((len(unique), N_CLASSES, 100))
    for i in range(len(unique)):
        cls = (INTERVAL_SIZES - 1) * MAX_K + n_in[i, N_RULES:]
        np.add.at(counts[i], cls, 1.0)
        np.add.at(mem_sum[i], cls, INTERVAL_MEMBERSHIP)
    class_in_concept = mem_sum / np.maximum(counts, 1.0)[:, :, None]  # (sets, classes, targets)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_s = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    return {
        "cell_set": cell_s,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_t].T.copy(),
        "cell_class_member": class_in_concept[cell_s, :, cell_t],
        "rule_n_in": n_in[:, :N_RULES].astype("float64"),
        # Empty classes get a negligible weight rather than log(0).
        "log_class_count": np.where(counts > 0, np.log(np.maximum(counts, 1.0)), -1e4),
        "n_examples": np.array([float(len(s)) for s in unique]),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
