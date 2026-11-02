"""Primacy-trusted Bayesian concept learning. People do the same Bayesian concept
learning as without_replacement_example_sampling (45 independently weighted
rules, intervals of 1..100 with an Erlang width prior, examples drawn without
replacement, an inferred share `noise` of random examples), but they trust the
first example more than later ones: the first example is random with
probability `trust * noise` (trust in (0, 1), inferred; trust = 1 recovers the
incumbent), later examples with probability `noise`. Concepts that exclude the
first example are therefore penalised more than ones excluding a later example.
"Yes" is the posterior probability that the concept contains the target.
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

# Intervals grouped by (size, examples inside, whether the first example is inside).
SIZES = np.arange(1, 101)
N_GROUPS = 100 * (MAX_EX + 1) * 2
GROUP_SIZE = np.repeat(SIZES, (MAX_EX + 1) * 2).astype("float64")
GROUP_NIN = np.tile(np.repeat(np.arange(MAX_EX + 1), 2), 100).astype("float64")
GROUP_FIRST = np.tile(np.array([0.0, 1.0]), 100 * (MAX_EX + 1))
GROUP_SIZE_IDX = np.repeat(np.arange(100), (MAX_EX + 1) * 2)
SIZES_F = SIZES.astype("float64")
N_PER_SIZE = 101.0 - SIZES_F

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0


def order_log_lik(n_ex, n_in, first_in, size, noise, noise1):
    """Log-likelihood of the examples, first one with noise rate noise1."""
    log_out, log_out1 = pt.log(noise / 100.0), pt.log(noise1 / 100.0)
    # First example: in concept as draw j=0, or random.
    first_in_term = pt.log((1 - noise1) / size + noise1 / 100.0)
    ll = first_in * first_in_term + (1 - first_in) * log_out1
    # Later examples: out-of-concept ones are random.
    ll = ll + (n_ex - 1 - (n_in - first_in)) * log_out
    # Later in-concept examples: j runs from first_in up to n_in - 1.
    for j in range(MAX_EX):
        term = pt.log((1 - noise) / pt.maximum(size - j, 1.0) + noise / 100.0)
        later = pt.gt(n_in, j) * pt.ge(j, first_in)
        ll = ll + later * term
    return ll


with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))
    rule_first_in = pm.Data("rule_first_in", np.zeros((1, N_RULES)))
    n_examples = pm.Data("n_examples", np.ones(1))
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_group_frac = pm.Data("cell_group_frac", np.zeros((1, N_GROUPS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    noise = pm.Beta("noise", 1.5, 6.0)
    trust = pm.Beta("trust", 2.0, 2.0)  # first example's noise rate relative to later ones
    noise1 = noise * trust
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

    rule_log_lik = order_log_lik(n_examples[:, None], rule_n_in, rule_first_in,
                                 pt.as_tensor(RULE_SIZE)[None, :], noise, noise1)
    group_log_lik = order_log_lik(n_examples[:, None], pt.as_tensor(GROUP_NIN)[None, :],
                                  pt.as_tensor(GROUP_FIRST)[None, :], pt.as_tensor(GROUP_SIZE)[None, :],
                                  noise, noise1)
    log_groups = (pt.log1p(-lambda_0) + log_per_interval[GROUP_SIZE_IDX])[None, :] + group_log_count + group_log_lik
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_post = posterior[cell_set]
    in_concept = (cell_post[:, :N_RULES] * cell_rule_member).sum(axis=1) + (
        cell_post[:, N_RULES:] * cell_group_frac
    ).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` (in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_n_in = np.zeros((S, N_RULES))
    rule_first = np.zeros((S, N_RULES))
    n_ex = np.zeros(S)
    counts = np.zeros((S, N_GROUPS))
    contain = np.zeros((S, N_GROUPS, 100))
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        n_ex[i] = len(s)
        rule_n_in[i] = RULE_MEMBERSHIP[:, ex].sum(axis=1)
        rule_first[i] = RULE_MEMBERSHIP[:, ex[0]]
        for m in range(1, 101):
            for n in range(m, 101):
                k = int(sum(m <= x <= n for x in s))
                f = int(m <= s[0] <= n)
                g = ((n - m) * (MAX_EX + 1) + k) * 2 + f
                counts[i, g] += 1
                contain[i, g, m - 1:n] += 1
    frac = np.where(counts[:, :, None] > 0, contain / np.maximum(counts, 1)[:, :, None], 0.0)
    log_count = np.where(counts > 0, np.log(np.maximum(counts, 1)), -1e3)
    keys = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {k: i for i, k in enumerate(cells)}
    cs = np.array([k[0] for k in cells], dtype="int64")
    ct = np.array([k[1] for k in cells], dtype="int64")
    return {
        "rule_n_in": rule_n_in,
        "rule_first_in": rule_first,
        "n_examples": n_ex,
        "group_log_count": log_count,
        "cell_set": cs,
        "cell_rule_member": RULE_MEMBERSHIP[:, ct].T.copy(),
        "cell_group_frac": frac[cs, :, ct],
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
