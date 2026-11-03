"""People store each example as a separate exemplar and judge the target by its
similarity to each one, where the similarity of target t to exemplar x is how
likely a familiar number concept brought to mind by x alone (45 rules with a
family-organised prior and exception rules as their own family, intervals of
1..100 with a beta prior over the share of the line they cover, the size
principle) also contains t. Similarities combine multiplicatively (conjunctive
exemplar rule): the log-odds of "yes" is a criterion plus a gain times the mean
log-similarity of the target to the exemplars, so a target must resemble every
example to be accepted.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
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


_RULES = rule_concepts()
RULE_NAMES = list(_RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in _RULES.values()], dtype="float64")  # (rules, 100)
N_RULES = len(RULE_NAMES)
FAMILY = np.array([rule_family(n) for n in RULE_NAMES])
N_FAMILIES = FAMILY.max() + 1
FAMILY_SIZE = np.bincount(FAMILY)
# One-shot likelihood of exemplar x under each rule, and rule-mediated co-membership.
RULE_LIK = ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100  # (rules, x)

SIZES = np.arange(1, 101, dtype="float64")
LOG_N_OF_SIZE = np.log(101.0 - SIZES)
SHARE = (SIZES - 0.5) / 100.0
# Interval counts by size L (index L-1): BOTH[L, x, t] intervals containing x and t;
# ONE[L, t] intervals containing t; N_OF_SIZE[L] all intervals of size L.
_i = NUMBERS[:, None]
_j = NUMBERS[None, :]
_lo = np.minimum(_i, _j)
_hi = np.maximum(_i, _j)
BOTH = np.zeros((100, 100, 100))
for L in range(1, 101):
    # start s with s <= lo, s + L - 1 >= hi, 1 <= s <= 101 - L
    smin = np.maximum(1, _hi - L + 1)
    smax = np.minimum(_lo, 101 - L)
    BOTH[L - 1] = np.maximum(0, smax - smin + 1)
ONE = np.stack([np.diagonal(BOTH[L]) for L in range(100)])  # (sizes, t)
N_OF_SIZE = 101.0 - SIZES
# Interval-mediated terms per size, folded into the one-shot likelihood.
INT_NUM = ALPHA * BOTH / SIZES[:, None, None] + (1 - ALPHA) / 100 * ONE[:, None, :]  # (L, x, t)
INT_DEN = ALPHA * ONE / SIZES[:, None] + (1 - ALPHA) / 100 * N_OF_SIZE[:, None]  # (L, x)


with pm.Model() as model:
    ex_idx = pm.Data("ex_idx", np.zeros((1, MAX_EX), dtype="int64"))  # (cells, 4): example - 1
    ex_mask = pm.Data("ex_mask", np.zeros((1, MAX_EX)))  # (cells, 4)
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # (cells,): target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    mean_share = pm.Beta("mean_share", 2, 3)
    concentration = pm.LogNormal("concentration", mu=np.log(5.0), sigma=1.0)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    rule_prior = lambda_0 * family_weights[FAMILY] / FAMILY_SIZE[FAMILY]  # (rules,)

    log_size_prior = ((mean_share * concentration - 1.0) * np.log(SHARE)
                      + ((1.0 - mean_share) * concentration - 1.0) * np.log1p(-SHARE))
    size_prior = pt.exp(log_size_prior - pt.logsumexp(log_size_prior))
    per_interval = (1.0 - lambda_0) * size_prior / N_OF_SIZE  # (sizes,)

    # Concept-mediated similarity matrix S[x, t] = P(t in concept | exemplar x alone).
    num = (pt.dot((RULE_LIK * rule_prior[:, None]).T, RULE_MEMBERSHIP)
           + pt.tensordot(per_interval, INT_NUM, axes=1))
    den = pt.dot(rule_prior, RULE_LIK) + pt.dot(per_interval, INT_DEN)
    sim = pt.clip(num / den[:, None], 1e-6, 1.0)
    log_sim = pt.log(sim)

    ex_log_sim = log_sim[ex_idx, cell_target[:, None]]  # (cells, 4)
    mean_log_sim = (ex_log_sim * ex_mask).sum(axis=1) / ex_mask.sum(axis=1)

    criterion = pm.Normal("criterion", mu=0.0, sigma=2.0)
    gain = pm.LogNormal("gain", mu=np.log(2.0), sigma=0.7)
    logit_yes = criterion + gain * mean_log_sim
    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(logit_yes)[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    keys = [(tuple(int(x) for x in str(r["number_set"]).split(",")), int(r["target"])) for r in rows]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    idx = np.zeros((len(cells), MAX_EX), dtype="int64")
    mask = np.zeros((len(cells), MAX_EX))
    for i, (s, _) in enumerate(cells):
        idx[i, :len(s)] = np.array(s) - 1
        mask[i, :len(s)] = 1.0
    return {
        "ex_idx": idx,
        "ex_mask": mask,
        "cell_target": np.array([c[1] - 1 for c in cells], dtype="int64"),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
