"""Exception-tolerant tempered concept learning over arithmetic rules and place-value grid regions.

People do exception-tolerant, tempered Bayesian concept learning over arithmetic
rules, but they represent magnitude on a decimal place-value grid (tens digit x
ones digit) rather than on a continuous number line: their non-arithmetic
concepts are rectangles of that grid -- a range of tens digits crossed with a
range of final digits ("the 40s-90s ending in 2", "the 20s and 30s") -- weighed
by the size principle. Same 45 rules as the exception-tolerant mixture (prior
lambda_0 split equally), grid rectangles share 1 - lambda_0 equally; examples
are drawn from the concept with exception rate eps, evidence tempered by beta.
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


RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in rule_concepts().values()], dtype="float64")
N_RULES = RULE_MEMBERSHIP.shape[0]
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)

# Place-value grid: tens digit 0..10 (100 -> tens 10), ones digit 0..9.
TENS = NUMBERS // 10
ONES = NUMBERS % 10
_rects = []
for a in range(11):
    for b in range(a, 11):
        for c in range(10):
            for d in range(c, 10):
                m = (TENS >= a) & (TENS <= b) & (ONES >= c) & (ONES <= d)
                if m.any():
                    _rects.append(m)
RECT_MEMBERSHIP = np.array(_rects, dtype="float64")
N_RECTS = RECT_MEMBERSHIP.shape[0]
RECT_SIZE = RECT_MEMBERSHIP.sum(axis=1)
# Rectangles with the same size and the same number of examples inside share a
# likelihood, so they are pooled into (examples inside, size) groups per set.
SIZES = np.unique(RECT_SIZE)
SIZE_ID = np.searchsorted(SIZES, RECT_SIZE)
MAX_K = 4
GROUP_K = np.repeat(np.arange(MAX_K + 1), len(SIZES)).astype("float64")
GROUP_SIZE = np.tile(SIZES, MAX_K + 1)
N_GROUPS = len(GROUP_K)


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples inside each rule
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups): log #rects
    cell_group_count = pm.Data("cell_group_count", np.zeros((1, N_GROUPS)))  # (cells, groups): #rects holding target
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    set_n = pm.Data("set_n", np.ones(1))  # (sets,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)  # sharpness of the size principle
    eps = pm.Beta("eps", 2, 5)  # believed rate of off-concept examples

    log_noise = pt.log(eps) - np.log(100)
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)
    group_in = pt.log((1 - eps) / GROUP_SIZE + eps / 100)
    log_rules = (
        pt.log(lambda_0) - np.log(N_RULES)
        + beta * (rule_k * rule_in[None, :] + (set_n[:, None] - rule_k) * log_noise)
    )
    log_groups = pt.log1p(-lambda_0) - np.log(N_RECTS) + beta * (
        GROUP_K[None, :] * group_in[None, :] + (set_n[:, None] - GROUP_K[None, :]) * log_noise
    )  # (sets, groups): log weight of one rectangle in each group
    shift = pt.maximum(pt.max(log_rules, axis=1), pt.max(log_groups + group_log_count, axis=1))
    w_rules = pt.exp(log_rules - shift[:, None])
    w_one = pt.exp(log_groups - shift[:, None])
    z = w_rules.sum(axis=1) + pt.exp(log_groups + group_log_count - shift[:, None]).sum(axis=1)
    numer = (w_rules[cell_set] * pt.as_tensor(RULE_MEMBERSHIP.T)[cell_target]).sum(axis=1) + (
        w_one[cell_set] * cell_group_count
    ).sum(axis=1)
    in_concept = numer / z[cell_set]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    trial_cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(trial_cells))
    cell_index = {c: i for i, c in enumerate(cells)}
    # Group of each rectangle for each set: (examples inside, size).
    rect_group = [
        np.minimum(RECT_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1), MAX_K).astype(int) * len(SIZES) + SIZE_ID
        for s in unique
    ]
    counts = np.stack([np.bincount(g, minlength=N_GROUPS) for g in rect_group])
    cell_counts = np.stack(
        [np.bincount(rect_group[i], weights=RECT_MEMBERSHIP[:, t - 1], minlength=N_GROUPS) for i, t in cells]
    )
    return {
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "group_log_count": np.where(counts > 0, np.log(np.maximum(counts, 1)), -1e4),
        "cell_group_count": cell_counts.astype("float64"),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
