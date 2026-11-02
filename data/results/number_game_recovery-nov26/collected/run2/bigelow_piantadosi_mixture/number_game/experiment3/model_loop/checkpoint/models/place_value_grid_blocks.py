"""Place-value grid blocks replace number-line ranges.

People do not picture stretches of numbers on a continuous number line; they
read each number as a two-digit numeral in a tens-by-ones place-value grid, and
the magnitude-like concepts they entertain are blocks of that grid (a band of
tens digits combined with a band of ones digits, e.g. "the sixties" or "fifties
and sixties ending in 0-4"), weighed by the size principle alongside the 45
arithmetic and digit rules and the 18 digit-sum concepts (inferred family
share). Examples are distinct draws without replacement with 10% strays, and
there is a guessing lapse. Generalization from a cluster spreads within its
decade and to the same ones positions of neighbouring decades, but drops at
decade boundaries.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
TENS = NUMBERS // 10  # 100 sits in a tens row of its own (10)
ONES = NUMBERS % 10


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
    return [np.isin(NUMBERS, c) for c in rules.values()]


def grid_blocks():
    """Distinct non-empty blocks: tens digit in [a, b] and ones digit in [c, d]."""
    seen = {}
    for a in range(11):
        for b in range(a, 11):
            for c in range(10):
                for d in range(c, 10):
                    m = (TENS >= a) & (TENS <= b) & (ONES >= c) & (ONES <= d)
                    if m.any():
                        seen.setdefault(m.tobytes(), m)
    return list(seen.values())


RULES = rule_concepts()
N_RULES = len(RULES)
DIGIT_SUM = np.array([sum(int(ch) for ch in str(n)) for n in NUMBERS])
SUMS = [DIGIT_SUM == s for s in range(1, 19)]
N_SUMS = len(SUMS)
BLOCKS = grid_blocks()
N_BLOCKS = len(BLOCKS)
MEMBERSHIP = np.array(RULES + SUMS + BLOCKS, dtype="float64")
SIZE = MEMBERSHIP.sum(axis=1)
N_FINITE = N_RULES + N_SUMS


def set_log_lik(examples):
    """log P(examples in order | h) for every hypothesis, sampling members without replacement."""
    total = np.zeros(len(MEMBERSHIP))
    shown = np.zeros(len(MEMBERSHIP))
    for x in examples:
        member = MEMBERSHIP[:, x - 1]
        remaining = np.maximum(SIZE - shown, 1.0)
        total += np.log(ALPHA * member / remaining + (1 - ALPHA) / 100)
        shown += member
    return total


with pm.Model() as model:
    rule_log_evidence = pm.Data("rule_log_evidence", np.zeros(1))  # (sets,)
    sum_log_evidence = pm.Data("sum_log_evidence", np.zeros(1))  # (sets,)
    block_log_evidence = pm.Data("block_log_evidence", np.zeros(1))  # (sets,)
    rule_in_cell = pm.Data("rule_in_cell", np.zeros(1))  # (cells,)
    sum_in_cell = pm.Data("sum_in_cell", np.zeros(1))  # (cells,)
    block_in_cell = pm.Data("block_in_cell", np.zeros(1))  # (cells,)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)  # prior share of symbolic rules versus grid blocks
    log_sum_share = pm.Normal("log_sum_share", -3.0, 1.5)

    log_rules_only = rule_log_evidence - np.log(N_RULES)
    log_sums_only = sum_log_evidence - np.log(N_SUMS) + log_sum_share
    p_sum = pm.math.sigmoid(log_sums_only - log_rules_only)[cell_set]
    finite_in_cell = (1 - p_sum) * rule_in_cell + p_sum * sum_in_cell
    log_finite = (
        pt.log(lambda_0) + pt.logaddexp(log_rules_only, log_sums_only) - pt.logaddexp(0.0, log_sum_share)
    )
    log_blocks = pt.log1p(-lambda_0) + block_log_evidence - np.log(N_BLOCKS)
    p_rule = pm.math.sigmoid(log_finite - log_blocks)[cell_set]
    in_concept = p_rule * finite_in_cell + (1 - p_rule) * block_in_cell

    p_concept = pt.clip(in_concept[row_cell], 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    keys = [(set_index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    n_sets = len(unique)
    ev = np.zeros((3, n_sets))
    in_set = np.zeros((3, n_sets, 100))
    bounds = [(0, N_RULES), (N_RULES, N_FINITE), (N_FINITE, len(MEMBERSHIP))]
    for i, s in enumerate(unique):
        ll = set_log_lik(s)
        for f, (lo, hi) in enumerate(bounds):
            ev[f, i] = logsumexp(ll[lo:hi])
            in_set[f, i] = np.exp(ll[lo:hi] - ev[f, i]) @ MEMBERSHIP[lo:hi]
    cell_set = np.array([c[0] for c in cells], dtype="int64")
    cell_target = np.array([c[1] - 1 for c in cells], dtype="int64")
    cell_index = {c: i for i, c in enumerate(cells)}
    return {
        "rule_log_evidence": ev[0],
        "sum_log_evidence": ev[1],
        "block_log_evidence": ev[2],
        "rule_in_cell": in_set[0, cell_set, cell_target],
        "sum_in_cell": in_set[1, cell_set, cell_target],
        "block_in_cell": in_set[2, cell_set, cell_target],
        "cell_set": cell_set,
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
