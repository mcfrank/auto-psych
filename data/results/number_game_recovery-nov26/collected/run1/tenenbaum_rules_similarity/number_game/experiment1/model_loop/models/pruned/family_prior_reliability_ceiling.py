"""Refinement of learned_example_reliability_ceiling: Bayesian concept learning
with a learned example reliability, an endorsement ceiling, and a prior weight
per concept family.

People consider the 45 rules (odd, even, primes, squares, cubes, powers of
2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11 and 12, ending in
0-9, containing the digit 0-9) and every interval of 1..100. Each example is
drawn uniformly from the concept with probability `reliability` and uniformly
from 1..100 otherwise (size principle); "yes" is `ceiling` times the posterior
probability that the concept contains the target. The one change: the single
rule-vs-interval weight with equally likely rules is replaced by a prior mass
per concept family (parity; primes/squares/cubes/powers; multiples of k; ends
in d; contains digit d; intervals), shared equally by the family's members.
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


def family(name):
    if name in ("odd", "even"):
        return 0
    if name.startswith("multiples"):
        return 2
    if name.startswith("ends"):
        return 3
    if name.startswith("contains"):
        return 4
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
N_INTERVALS = len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
SIZE = MEMBERSHIP.sum(axis=1)
HYP_FAMILY = np.concatenate([[family(n) for n in RULE_NAMES], np.full(N_INTERVALS, 5)]).astype("int64")
FAMILY_SIZE = np.bincount(HYP_FAMILY).astype("float64")
LOG_FAMILY_SIZE = np.log(FAMILY_SIZE)[HYP_FAMILY]

with pm.Model() as model:
    n_in = pm.Data("n_in", np.zeros((1, MEMBERSHIP.shape[0])))  # (sets, hyps): examples inside h
    n_out = pm.Data("n_out", np.zeros((1, MEMBERSHIP.shape[0])))  # (sets, hyps): examples outside h
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the five rule families and of intervals.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(6))
    # Probability of endorsing a target believed to be in the concept.
    ceiling = pm.Beta("ceiling", 4, 2)
    # Believed probability that an example comes from the concept.
    reliability = pm.Beta("reliability", 18, 2)

    log_prior = pt.log(family_weight)[HYP_FAMILY] - LOG_FAMILY_SIZE  # (hyps,)
    log_in = pt.log(reliability / SIZE + (1 - reliability) / 100)  # (hyps,)
    log_out = pt.log1p(-reliability) - np.log(100)
    log_joint = log_prior + n_in * log_in + n_out * log_out  # (sets, hyps)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior @ pt.as_tensor(MEMBERSHIP)  # (sets, targets)

    membership = in_concept.reshape((-1,))[row_cell]
    p_yes = pm.Deterministic("p_yes", pt.clip(ceiling * membership, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    k_in = np.stack([MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique])
    k_total = np.array([len(s) for s in unique], dtype="float64")[:, None]
    return {
        "n_in": k_in,
        "n_out": k_total - k_in,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
