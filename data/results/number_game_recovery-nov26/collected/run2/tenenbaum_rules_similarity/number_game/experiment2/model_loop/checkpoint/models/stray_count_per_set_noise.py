"""Refinement of without_replacement_example_sampling: the same Bayesian concept
learning (45 independently weighted rules, every interval of 1..100 with an
Erlang width prior, in-concept examples drawn without replacement), with one
change to how people think random examples arise. Instead of each example
independently being random with a fixed rate, a set contains k random examples
with probability proportional to r**k (k = 0..n, r inferred) whatever the set
size n, and each k-subset of the examples is equally likely to be the strays.
So for a concept, P(examples | h) = sum over stray subsets S containing every
example outside h of r**|S| / C(n, |S|) * (1/100)**|S| * prod_j 1/(size - j)
over the non-stray examples. "Yes" is the posterior probability that the
concept contains the target.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_EX = 4
NEG = -1e4


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

# Intervals grouped by (size, number of examples inside).
SIZES = np.arange(1, 101)
GROUP_SIZE = np.repeat(SIZES, MAX_EX + 1).astype("float64")  # (500,)
GROUP_NIN = np.tile(np.arange(MAX_EX + 1), 100)
GROUP_SIZE_IDX = np.repeat(np.arange(100), MAX_EX + 1)
SIZES_F = SIZES.astype("float64")
N_PER_SIZE = 101.0 - SIZES_F
N_CONCEPTS = N_RULES + 500
CONCEPT_SIZE = np.concatenate([RULE_SIZE, GROUP_SIZE])

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0


def _log_without_replacement(size, m):
    """log prod_{j<m} 1 / (size - j); -inf-like if the concept is too small."""
    if m > size:
        return NEG
    return -sum(math.log(size - j) for j in range(m))


def _stray_terms(n, n_in, size):
    """Per stray-count option s (strays inside the concept): r-free log term and stray count."""
    base = np.full(MAX_EX + 1, NEG)
    count = np.zeros(MAX_EX + 1)
    k_out = n - n_in
    if k_out < 0:
        return base, count
    for s in range(n_in + 1):
        k = k_out + s
        base[s] = (math.log(math.comb(n_in, s)) - math.log(math.comb(n, k)) - k * math.log(100.0)
                   + _log_without_replacement(size, n_in - s))
        count[s] = k
    return base, count


with pm.Model() as model:
    stray_base = pm.Data("stray_base", np.zeros((1, N_CONCEPTS, MAX_EX + 1)))  # (sets, concepts, s)
    stray_count = pm.Data("stray_count", np.zeros((1, N_CONCEPTS, MAX_EX + 1)))
    group_log_count = pm.Data("group_log_count", np.zeros((1, 500)))  # (sets, groups)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_group_frac = pm.Data("cell_group_frac", np.zeros((1, 500)))  # (cells, groups)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    slip = pm.Beta("slip", 1.5, 4.0)  # odds of each further random example in a set
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    log_erlang = pt.log(SIZES_F) - SIZES_F / sigma
    log_per_interval = log_erlang - pt.logsumexp(log_erlang + np.log(N_PER_SIZE))

    # The normaliser of the stray-count distribution depends only on the set, so it cancels.
    log_lik = pt.logsumexp(stray_base + stray_count * pt.log(slip), axis=2)  # (sets, concepts)
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + log_lik[:, :N_RULES]
    log_groups = ((pt.log1p(-lambda_0) + log_per_interval[GROUP_SIZE_IDX])[None, :]
                  + group_log_count + log_lik[:, N_RULES:])
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_post = posterior[cell_set]
    in_concept = (cell_post[:, :N_RULES] * cell_rule_member).sum(axis=1) + (
        cell_post[:, N_RULES:] * cell_group_frac
    ).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set`, `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    base = np.full((S, N_CONCEPTS, MAX_EX + 1), NEG)
    count = np.zeros((S, N_CONCEPTS, MAX_EX + 1))
    counts = np.zeros((S, 500))
    contain = np.zeros((S, 500, 100))
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        n = len(s)
        rule_n_in = RULE_MEMBERSHIP[:, ex].sum(axis=1).astype(int)
        n_in_all = np.concatenate([rule_n_in, GROUP_NIN])
        for c in range(N_CONCEPTS):
            base[i, c], count[i, c] = _stray_terms(n, int(n_in_all[c]), int(CONCEPT_SIZE[c]))
        for m in range(1, 101):
            for hi in range(m, 101):
                k = int(sum(m <= x <= hi for x in s))
                g = (hi - m) * (MAX_EX + 1) + k
                counts[i, g] += 1
                contain[i, g, m - 1:hi] += 1
    frac = np.where(counts[:, :, None] > 0, contain / np.maximum(counts, 1)[:, :, None], 0.0)
    log_count = np.where(counts > 0, np.log(np.maximum(counts, 1)), -1e3)
    keys = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {k: i for i, k in enumerate(cells)}
    cs = np.array([k[0] for k in cells], dtype="int64")
    ct = np.array([k[1] for k in cells], dtype="int64")
    return {
        "stray_base": base,
        "stray_count": count,
        "group_log_count": log_count,
        "cell_set": cs,
        "cell_rule_member": RULE_MEMBERSHIP[:, ct].T.copy(),
        "cell_group_frac": frac[cs, :, ct],
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
