"""Refinement of inferred_example_noise_concepts: Bayesian concept learning over
45 independently weighted rules and every interval [m, n] of 1..100, size
principle with an inferred share `noise` of examples emitted at random -- but
intervals are pictured on a compressed (logarithmic-like) mental number line,
so the prior on an interval is proportional to (m * n) ** (-compression), with
`compression` inferred (0 recovers the incumbent's uniform interval prior;
positive values favour ranges among small numbers, so "yes" extends further
below the examples than above them). "Yes" is the posterior probability that
the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor.gradient import disconnected_grad

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
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)

# Every interval [m, n], 1 <= m <= n <= 100.
INT_LO, INT_HI = np.triu_indices(100)
INT_LO, INT_HI = INT_LO + 1, INT_HI + 1
INT_SIZE = (INT_HI - INT_LO + 1).astype("float64")
# Log magnitude of the endpoints, centred over intervals.
INT_LOG_MAG = np.log(INT_LO) + np.log(INT_HI)
INT_LOG_MAG = INT_LOG_MAG - INT_LOG_MAG.mean()
INT_MEMBERSHIP = ((NUMBERS[None, :] >= INT_LO[:, None]) & (NUMBERS[None, :] <= INT_HI[:, None])).astype("float64")

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))  # (sets, rules)
    int_n_in = pm.Data("int_n_in", np.zeros((1, len(INT_LO))))  # (sets, intervals): examples inside
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # (cells,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    noise = pm.Beta("noise", 1.5, 6.0)
    compression = pm.Normal("compression", 0.0, 1.0)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    # Log likelihood of a set under each concept, dropping the term
    # n_examples * log(noise / 100) that every concept shares.
    log_out = pt.log(noise / 100.0)
    rule_log_lik = rule_n_in * (pt.log((1 - noise) / RULE_SIZE + noise / 100.0) - log_out)[None, :]
    int_log_lik = int_n_in * (pt.log((1 - noise) / INT_SIZE + noise / 100.0) - log_out)[None, :]

    log_int_prior = -compression * INT_LOG_MAG
    log_int_prior = log_int_prior - pt.logsumexp(log_int_prior)

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik  # (sets, rules)
    log_ints = pt.log1p(-lambda_0) + log_int_prior[None, :] + int_log_lik  # (sets, intervals)
    # Any per-set shift cancels on normalising (held fixed for the gradient); the
    # best rule's score is enough, as no interval outscores it by hundreds of nats.
    shift = disconnected_grad(log_rules.max(axis=1))[:, None]
    w_rules = pt.exp(log_rules - shift)
    w_ints = pt.exp(log_ints - shift)
    evidence = w_rules.sum(axis=1) + w_ints.sum(axis=1)
    # (sets, 100): posterior probability that each number is in the concept.
    in_concept = (pt.dot(w_rules, RULE_MEMBERSHIP) + pt.dot(w_ints, INT_MEMBERSHIP)) / evidence[:, None]
    cell_p = in_concept.flatten()[cell_set * 100 + cell_target]

    p_yes = pm.Deterministic("p_yes", pt.clip(cell_p[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set`, `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_n_in = np.zeros((S, N_RULES))
    int_n_in = np.zeros((S, len(INT_LO)))
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        rule_n_in[i] = RULE_MEMBERSHIP[:, ex].sum(axis=1)
        int_n_in[i] = INT_MEMBERSHIP[:, ex].sum(axis=1)
    keys = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {k: i for i, k in enumerate(cells)}
    return {
        "rule_n_in": rule_n_in,
        "int_n_in": int_n_in,
        "cell_set": np.array([k[0] for k in cells], dtype="int64"),
        "cell_target": np.array([k[1] for k in cells], dtype="int64"),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
