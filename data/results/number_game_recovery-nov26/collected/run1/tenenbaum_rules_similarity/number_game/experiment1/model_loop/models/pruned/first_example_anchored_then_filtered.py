"""First-example anchoring with consistency filtering.

People form their guess about the program from the first example alone: they
weigh 45 rules and the 5,050 intervals of 1..100 by a prior mass per concept
family and by the size principle applied to the FIRST example only (10% of
examples treated as uniform noise). Every later example is used only as a
consistency check: a concept that does not contain it is (nearly) discarded,
but a later example that fits brings no size-principle boost to tight concepts.
"Yes" is the posterior probability that the concept contains the target.
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
RULE_FAMILY = np.array([family(n) for n in RULE_NAMES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
N_INTERVALS = len(INTERVALS)
CONCEPT_FAMILY = np.concatenate([RULE_FAMILY, np.full(N_INTERVALS, 5)])
WITHIN_FAMILY_LOG = -np.log(np.concatenate([FAMILY_SIZE, [N_INTERVALS]]))[CONCEPT_FAMILY]
N_CONCEPTS = len(CONCEPT_FAMILY)
# Later examples: consistency check only (no 1/|h| size term), same noise rate.
FILTER_LOG_LIK = np.log(ALPHA * MEMBERSHIP + (1 - ALPHA) / 100)

with pm.Model() as model:
    # Per-set log-likelihood of the examples under every concept (parameter-free).
    set_log_lik = pm.Data("set_log_lik", np.zeros((1, N_CONCEPTS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_weight = pm.Dirichlet("family_weight", a=np.ones(6))

    log_prior = pt.log(family_weight)[CONCEPT_FAMILY] + WITHIN_FAMILY_LOG
    log_joint = log_prior[None, :] + set_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior @ pt.as_tensor(MEMBERSHIP)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.zeros((len(unique), N_CONCEPTS))
    for i, s in enumerate(unique):
        # Size principle for the first example; later examples only filter.
        set_log_lik[i] = EXAMPLE_LOG_LIK[:, s[0] - 1] + sum(FILTER_LOG_LIK[:, x - 1] for x in s[1:])
    return {
        "set_log_lik": set_log_lik,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
