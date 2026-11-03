"""Boundary examples anchor ranges: people do the same Bayesian concept learning
as beta_share_width_exception_concepts (45 rules with a family-organised prior and
exception rules as their own family, all intervals of 1..100 with a beta prior over
the share of the line a range covers, size principle with 90% of examples from the
concept). The single change is attention when a magnitude range is evaluated: the
range is judged by its boundary-setting examples (the smallest and largest), and
the examples in between, which say nothing new about the edges, count only with
an inferred weight w as evidence about the range. Rules use every example fully.
w < 1 means interior examples narrow range generalisation less than the size
principle demands; w = 1 is the ideal learner.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
NEG = -1e4  # stands in for log(0) where an (interval size, class) is empty


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
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
INTERVAL_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in INTERVALS], dtype="float64")
INTERVAL_LOG_LIK = np.log(ALPHA * INTERVAL_MEMBERSHIP / INTERVAL_MEMBERSHIP.sum(axis=1, keepdims=True)
                          + (1 - ALPHA) / 100)
INTERVAL_SIZE_IDX = (INTERVAL_MEMBERSHIP.sum(axis=1) - 1).astype("int64")  # L - 1
SIZES = np.arange(1, 101, dtype="float64")
LOG_N_OF_SIZE = np.log(101.0 - SIZES)
SHARE = (SIZES - 0.5) / 100.0
N_CLASS = 3  # number of interior examples inside the interval: 0, 1 or 2
CLASS = np.arange(N_CLASS, dtype="float64")
LOG_IN = np.log(ALPHA / SIZES + (1 - ALPHA) / 100)  # interior example inside, by size
LOG_OUT = np.log((1 - ALPHA) / 100)


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
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    edge_log_ev = pm.Data("edge_log_ev", np.zeros((1, 100, N_CLASS)))  # (sets, sizes, classes)
    n_interior = pm.Data("n_interior", np.zeros(1))  # (sets,)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_range_member = pm.Data("cell_range_member", np.zeros((1, 100 * N_CLASS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    mean_share = pm.Beta("mean_share", 2, 3)
    concentration = pm.LogNormal("concentration", mu=np.log(5.0), sigma=1.0)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    # weight of an interior (non-boundary) example as evidence about a range, in (0, 2)
    interior_weight = pm.Deterministic("interior_weight", 2.0 * pm.Beta("interior_weight_half", 2.0, 2.0))
    rule_log_prior = pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])

    log_size_prior = ((mean_share * concentration - 1.0) * np.log(SHARE)
                      + ((1.0 - mean_share) * concentration - 1.0) * np.log1p(-SHARE))
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior)
    log_per_interval = log_size_prior - LOG_N_OF_SIZE

    # interior examples' log-likelihood: c inside (each 1/L), the rest outside (noise)
    interior_ll = (CLASS[None, None, :] * LOG_IN[None, :, None]
                   + (n_interior[:, None, None] - CLASS[None, None, :]) * LOG_OUT)
    log_ranges = (pt.log1p(-lambda_0) + log_per_interval[None, :, None]
                  + edge_log_ev + interior_weight * interior_ll)
    log_rules = pt.log(lambda_0) + rule_log_prior[None, :] + rule_log_lik
    log_joint = pt.concatenate([log_rules, log_ranges.reshape((log_ranges.shape[0], 100 * N_CLASS))], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_post = posterior[cell_set]
    in_concept = ((cell_post[:, :N_RULES] * cell_rule_member).sum(axis=1)
                  + (cell_post[:, N_RULES:] * cell_range_member).sum(axis=1))

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    edge_ev = np.full((n_sets, 100, N_CLASS), NEG)
    member = np.zeros((n_sets, 100, N_CLASS, 100))
    n_int = np.zeros(n_sets)
    for i, s in enumerate(unique):
        arr = np.array(s)
        lo, hi = int(np.argmin(arr)), int(np.argmax(arr))
        edges = sorted({lo, hi})
        interior = [j for j in range(len(arr)) if j not in edges]
        n_int[i] = len(interior)
        ll = INTERVAL_LOG_LIK[:, arr[edges] - 1].sum(axis=1)
        n_in = (INTERVAL_MEMBERSHIP[:, arr[interior] - 1].sum(axis=1) if interior
                else np.zeros(len(INTERVALS))).astype("int64")
        for L in range(100):
            for c in range(N_CLASS):
                sel = (INTERVAL_SIZE_IDX == L) & (n_in == c)
                if not sel.any():
                    continue
                ev = logsumexp(ll[sel])
                edge_ev[i, L, c] = ev
                member[i, L, c] = np.exp(ll[sel] - ev) @ INTERVAL_MEMBERSHIP[sel]
    rule_ll = np.stack([RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_s = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    return {
        "rule_log_lik": rule_ll,
        "edge_log_ev": edge_ev,
        "n_interior": n_int,
        "cell_set": cell_s,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_t].T.copy(),
        "cell_range_member": np.clip(member[cell_s, :, :, cell_t], 0.0, 1.0).reshape(len(cells), 100 * N_CLASS),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
