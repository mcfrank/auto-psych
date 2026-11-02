"""Refinement of bigelow_piantadosi_independent: Bayesian concept learning
over 45 rules and all intervals of 1..100 with the size principle, where the
probability alpha that an example was drawn from the concept (rather than
uniformly from 1..100) is inferred instead of fixed at 0.9. alpha sets how
strongly the size principle favours small, specific concepts.

Rules have weights Dirichlet(2,...,2) (with the "ends in 0"/"contains digit 0"
duplicate split as in the incumbent), rules as a group lambda_0 ~ Beta(2, 2),
each interval prior (1 - lambda_0) / 100^2. "Yes" is the posterior probability
that the concept contains the target.

Because the likelihood of an example set under an interval depends only on the
interval's size and how many examples fall inside it, intervals are grouped
per set into (size, count-inside) cells, precomputed in prepare_observed, and
only the targets actually asked about each set are tracked.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4
N_K = MAX_N + 1


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

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples inside rule
    rule_target = pm.Data("rule_target", np.zeros((1, N_RULES, 1)))  # (sets, rules, asked targets)
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    cell_size = pm.Data("cell_size", np.ones((1, 1)))  # (sets, cells): interval size
    cell_k = pm.Data("cell_k", np.zeros((1, 1)))  # (sets, cells): examples inside
    cell_log_count = pm.Data("cell_log_count", np.zeros((1, 1)))  # (sets, cells): log #intervals
    cell_target = pm.Data("cell_target", np.zeros((1, 1, 1)))  # (sets, cells, asked targets): share containing it
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * slots + target slot
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    alpha = pm.Beta("alpha", 4, 2)
    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    log_out = pt.log1p(-alpha) - np.log(100.0)  # log P(x | h) for x outside h
    n = n_examples[:, None]

    rule_in = pt.log(alpha / RULE_SIZE + (1 - alpha) / 100.0)[None, :]
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_k * rule_in + (n - rule_k) * log_out

    cell_log_lik = cell_k * pt.log(alpha / cell_size + (1 - alpha) / 100.0) + (n - cell_k) * log_out + cell_log_count
    interval_log_evidence = pt.logsumexp(cell_log_lik, axis=1)  # (sets,)
    cell_post = pt.exp(cell_log_lik - interval_log_evidence[:, None])
    interval_in_concept = pt.matmul(cell_post[:, None, :], cell_target)[:, 0, :]  # (sets, slots)

    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + interval_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    rule_in_concept = pt.matmul(posterior[:, None, :N_RULES], rule_target)[:, 0, :]
    in_concept = rule_in_concept + posterior[:, N_RULES][:, None] * interval_in_concept

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    asked = [sorted({t for s2, t in zip(sets, targets) if s2 == s}) for s in unique]
    n_slots = max(len(a) for a in asked)
    lo, hi = NUMBERS[:, None], NUMBERS[None, :]
    per_set = []
    for s, a in zip(unique, asked):
        k = sum(((lo <= x) & (x <= hi)).astype(int) for x in s)
        cells = {}
        for m in range(100):
            for nn in range(m, 100):
                key = (nn - m + 1, k[m, nn])
                c = cells.setdefault(key, [0, np.zeros(len(a))])
                c[0] += 1
                c[1] += [(m + 1 <= t <= nn + 1) for t in a]
        per_set.append(cells)
    n_cells = max(len(c) for c in per_set)
    n_sets = len(unique)
    cell_size = np.ones((n_sets, n_cells))
    cell_k = np.zeros((n_sets, n_cells))
    cell_log_count = np.full((n_sets, n_cells), -1e4)
    cell_target = np.zeros((n_sets, n_cells, n_slots))
    rule_target = np.zeros((n_sets, N_RULES, n_slots))
    for i, (cells, a) in enumerate(zip(per_set, asked)):
        for j, ((size, kk), (count, contains)) in enumerate(cells.items()):
            cell_size[i, j], cell_k[i, j], cell_log_count[i, j] = size, kk, np.log(count)
            cell_target[i, j, :len(a)] = contains / count
        rule_target[i, :, :len(a)] = RULE_MEMBERSHIP[:, np.array(a) - 1]
    slot = [{t: j for j, t in enumerate(a)} for a in asked]
    return {
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "rule_target": rule_target,
        "n_examples": np.array([len(s) for s in unique], dtype="float64"),
        "cell_size": cell_size,
        "cell_k": cell_k,
        "cell_log_count": cell_log_count,
        "cell_target": cell_target,
        "row_cell": np.array([index[s] * n_slots + slot[index[s]][t] for s, t in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
