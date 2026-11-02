"""First-example reliability in concept learning: people judge how far the first
example shown is a genuine sample of the hidden concept differently from the later
examples. The first example is drawn from the concept with an inferred probability
(otherwise it is a random number from 1..100), while each later example is drawn
from it with the usual probability 0.9. Everything else is the best model
(gamma_width_exception_family_concepts): 45 rules and all intervals of 1..100,
size principle, family-organised rule prior with exception rules as their own
family, and a gamma prior over interval width with inferred shape and mean. The
order of the examples matters through which one comes first.

Because the first example's likelihood is a1 * 1[x1 in h] / |h| + (1 - a1) / 100,
a set's likelihood is a1 * A(h) + (1 - a1) * B(h) with A, B parameter-free, so
prepare_observed sums A and B within each interval size (and over the intervals
containing each target) once, in linear space scaled by each set's maximum.
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
CONCEPT_SIZE = MEMBERSHIP.sum(axis=1, keepdims=True)
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / CONCEPT_SIZE + (1 - ALPHA) / 100)
FIRST_FROM_CONCEPT = MEMBERSHIP / CONCEPT_SIZE  # P(x1 | h) if x1 is a genuine sample
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_SIZE_IDX = (INTERVAL_MEMBERSHIP.sum(axis=1) - 1).astype("int64")  # L - 1
SIZE_ONEHOT = (INTERVAL_SIZE_IDX[:, None] == np.arange(100)[None, :]).astype("float64")
SIZES = np.arange(1, 101, dtype="float64")
LOG_N_OF_SIZE = np.log(101.0 - SIZES)  # number of intervals of each size


def rule_family(name):
    if name in ("odd", "even"):
        return 0
    if name == "primes":
        return 1
    if name in ("squares", "cubes"):
        return 2
    if name.startswith("powers of 2,"):
        return 7
    if name.startswith("powers"):
        return 3
    if name.startswith("multiples"):
        return 4
    if name.startswith("ends in"):
        return 5
    return 6


FAMILY = np.array([rule_family(n) for n in RULE_NAMES])
N_FAMILIES = FAMILY.max() + 1
FAMILY_SIZE = np.bincount(FAMILY)

with pm.Model() as model:
    # Scaled likelihoods of each set: A = first example genuine, B = first random.
    rule_a = pm.Data("rule_a", np.zeros((1, N_RULES)))  # (sets, rules)
    rule_b = pm.Data("rule_b", np.zeros((1, N_RULES)))
    size_a = pm.Data("size_a", np.zeros((1, 100)))  # (sets, sizes): sum over intervals
    size_b = pm.Data("size_b", np.zeros((1, 100)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_size_a = pm.Data("cell_size_a", np.zeros((1, 100)))  # (cells, sizes): target-containing sums
    cell_size_b = pm.Data("cell_size_b", np.zeros((1, 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    first_reliability = pm.Beta("first_reliability", 2.0, 1.0)  # P(first example is from the concept)
    lambda_0 = pm.Beta("lambda_0", 2, 2)
    mean_width = pm.LogNormal("mean_width", mu=np.log(30.0), sigma=1.0)
    shape = pm.LogNormal("shape", mu=np.log(2.0), sigma=0.75)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    rule_prior = lambda_0 * family_weights[FAMILY] / FAMILY_SIZE[FAMILY]

    log_size_prior = (shape - 1.0) * np.log(SIZES) - SIZES * shape / mean_width
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior)
    interval_prior = (1.0 - lambda_0) * pt.exp(log_size_prior - LOG_N_OF_SIZE)  # one interval of size L

    a1 = first_reliability
    rule_lik = a1 * rule_a + (1.0 - a1) * rule_b
    size_lik = a1 * size_a + (1.0 - a1) * size_b
    evidence = pt.dot(rule_lik, rule_prior) + pt.dot(size_lik, interval_prior)  # (sets,)
    cell_rule_lik = rule_lik[cell_set] * cell_rule_member
    cell_size_lik = a1 * cell_size_a + (1.0 - a1) * cell_size_b
    contains = pt.dot(cell_rule_lik, rule_prior) + pt.dot(cell_size_lik, interval_prior)
    in_concept = contains / evidence[cell_set]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`;
    the order matters, since the first example is treated differently."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    lik_a, lik_b = [], []
    for s in unique:
        rest = EXAMPLE_LOG_LIK[:, np.array(s[1:], dtype="int64") - 1].sum(axis=1)
        log_a = np.log(np.maximum(FIRST_FROM_CONCEPT[:, s[0] - 1], 1e-300)) + rest
        log_a[FIRST_FROM_CONCEPT[:, s[0] - 1] == 0] = -np.inf
        log_b = np.log(0.01) + rest
        scale = max(log_a.max(), log_b.max())
        lik_a.append(np.exp(log_a - scale))
        lik_b.append(np.exp(log_b - scale))
    lik_a, lik_b = np.array(lik_a), np.array(lik_b)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_s = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    member_t = INTERVAL_MEMBERSHIP[:, cell_t].T  # (cells, intervals)
    return {
        "rule_a": lik_a[:, :N_RULES],
        "rule_b": lik_b[:, :N_RULES],
        "size_a": lik_a[:, N_RULES:] @ SIZE_ONEHOT,
        "size_b": lik_b[:, N_RULES:] @ SIZE_ONEHOT,
        "cell_set": cell_s,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_t].T.copy(),
        "cell_size_a": (lik_a[cell_s, N_RULES:] * member_t) @ SIZE_ONEHOT,
        "cell_size_b": (lik_b[cell_s, N_RULES:] * member_t) @ SIZE_ONEHOT,
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
