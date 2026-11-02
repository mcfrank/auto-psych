"""Exemplar retrieval of stored number concepts.

People hold number concepts in memory as stored sets of member numbers (the 45
Bigelow-Piantadosi rules -- odd, even, primes, squares, cubes, powers, multiples,
final digit, contained digit -- and every magnitude range of 1-100) and judge by
exemplar similarity on the number line, s(x, y) = r^|x - y|. A stored concept is
retrieved in proportion to its prior weight times, for each example, the
example's mean similarity to the concept's stored members (GCM-style normalised
summed similarity; 10% of examples are treated as unrelated, as in the
incumbent), so an example close to but outside a small concept still partly
supports it. A target is endorsed by its similarity to the nearest stored
member of each retrieved concept, so endorsement falls off gradedly just
outside a concept. The single free mechanism is the similarity gradient r
(r -> 0 recovers crisp membership).

The 45 rules are computed exactly in the graph. The 5050 ranges depend on the
parameters only through r, so prepare_observed tabulates, on a fine grid of
logit(r), each set's summed range evidence and each (set, target) cell's
evidence-weighted endorsement share; the model interpolates them smoothly.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
ALPHA = 0.9
GRID = np.linspace(-8.0, 3.0, 221)  # logit(r)
GRID_STEP = GRID[1] - GRID[0]


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
ABS_DIFF = np.abs(NUMBERS[:, None] - NUMBERS[None, :]).astype("float64")
# Distance from each number to the nearest member of each rule: (rules, numbers).
RULE_DMIN = np.array([np.where(m[None, :] > 0, ABS_DIFF, np.inf).min(axis=1) for m in RULE_MEMBERSHIP])

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0


def range_tables(counts, cell_set, cell_target):
    """Per grid value of logit(r): log summed range evidence per set, and per
    cell the evidence share of ranges weighted by target-to-range similarity."""
    lo = np.array([m for m in NUMBERS for n in range(m, 101)], dtype="float64")[:, None]
    hi = np.array([n for m in NUMBERS for n in range(m, 101)], dtype="float64")[:, None]
    size = hi - lo + 1
    x = NUMBERS[None, :].astype("float64")
    inside = (x >= lo) & (x <= hi)
    dmin = np.maximum(0.0, np.maximum(lo - x, x - hi))
    log_z = np.zeros((len(GRID), counts.shape[0]))
    share = np.zeros((len(GRID), len(cell_set)))
    for g, logit_r in enumerate(GRID):
        r = 1.0 / (1.0 + np.exp(-logit_r))
        geo = r / (1 - r)
        summed = np.where(
            inside,
            1 + geo * (2 - r ** np.where(inside, x - lo, 0) - r ** np.where(inside, hi - x, 0)),
            r**dmin * (1 - r**size) / (1 - r),
        )
        ll = counts @ np.log(ALPHA * summed / size + (1 - ALPHA) / 100).T  # (sets, ranges)
        top = ll.max(axis=1, keepdims=True)
        w = np.exp(ll - top)
        z = w.sum(axis=1)
        log_z[g] = top[:, 0] + np.log(z)
        endorse = (w @ r**dmin) / z[:, None]  # (sets, numbers)
        share[g] = endorse[cell_set, cell_target]
    return log_z, share


with pm.Model() as model:
    example_counts = pm.Data("example_counts", np.zeros((1, 100)))  # (sets, numbers)
    range_log_z = pm.Data("range_log_z", np.zeros((len(GRID), 1)))  # (grid, sets)
    range_share = pm.Data("range_share", np.zeros((len(GRID), 1)))  # (grid, cells)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    # Similarity gradient on the number line: s(x, y) = r^|x - y|.
    r_logit = pm.TruncatedNormal("r_logit", mu=-2.5, sigma=1.5, lower=GRID[0], upper=GRID[-1])
    r = pm.Deterministic("r", pm.math.sigmoid(r_logit))
    log_r = pt.log(r)

    # Rules: summed similarity of each number to the rule's members.
    rule_sum = pt.dot(RULE_MEMBERSHIP, pt.exp(log_r * ABS_DIFF))  # (rules, numbers)
    rule_ll = pt.dot(example_counts, pt.log(ALPHA * rule_sum / RULE_SIZE[:, None] + (1 - ALPHA) / 100).T)
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_ll  # (sets, rules)

    # Ranges: smooth (Gaussian-kernel) interpolation of the tabulated evidence in logit(r).
    kernel = pt.exp(-0.5 * ((r_logit - GRID) / GRID_STEP) ** 2)
    tent = kernel / kernel.sum()  # (grid,)
    log_ranges = pt.log1p(-lambda_0) - 2 * np.log(100) + pt.dot(tent, range_log_z)  # (sets,)
    share = pt.dot(tent, range_share)  # (cells,)

    log_joint = pt.concatenate([log_rules, log_ranges[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    range_post = pt.exp(log_ranges - log_norm)

    # Endorsement: similarity of the target to the nearest member of each concept.
    rule_endorse = pt.exp(log_r * RULE_DMIN)  # (rules, numbers)
    p_cell = (rule_post[cell_set] * rule_endorse.T[cell_target]).sum(axis=1) + range_post[cell_set] * share
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell, 1e-6, 1 - 1e-6)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    counts = np.zeros((len(unique), 100))
    for i, s in enumerate(unique):
        for x in s:
            counts[i, x - 1] += 1
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_set = np.array([i for i, _ in cells], dtype="int64")
    cell_target = np.array([t - 1 for _, t in cells], dtype="int64")
    log_z, share = range_tables(counts, cell_set, cell_target)
    return {
        "example_counts": counts,
        "range_log_z": log_z,
        "range_share": share,
        "cell_set": cell_set,
        "cell_target": cell_target,
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
