"""Pedagogical (helpful-teacher) sampling in Bayesian concept learning.

People treat the examples as chosen on purpose by a helpful teacher: under a
candidate concept (one of 45 rules or an interval of 1..100) an example is
likely in proportion to its membership AND how diagnostic it is of that
concept for a naive learner, P_teach(x | h) ∝ [x in h] * L0(h | x)^beta, where
L0 is a literal learner with a flat prior (half mass on rules, half on
intervals) and the size-principle likelihood. beta (fitted) is the strength of
pedagogical expectation; beta = 0 recovers random sampling from the concept
(the size principle). 10% of examples are noise from 1..100. The learner has a
prior weight per rule (Dirichlet) and a rule share lambda_0 (each interval
(1 - lambda_0) / 100^2), and "yes" is the posterior probability that the
concept contains the target.

Within a concept every member has the same literal likelihood, so the
teacher's preference reduces to w(x) = Z(x)^-beta, Z(x) = sum_h P0(h) P0(x|h)
being how many concepts x fits (its ambiguity): P_teach(x | h) = w(x) / W(h),
W(h) the summed w of h's members (beta = 0: W(h) = |h|).

The interval hypotheses depend on beta alone, so their total evidence and
in-concept probabilities are computed per set on a fine beta grid in
prepare_observed and read by cubic (Catmull-Rom) interpolation; the rules are
computed exactly.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

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
N_INT = len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INT_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INT_M = np.array([c[0] - 1 for c in INTERVALS])  # 0-based endpoints
INT_N = np.array([c[-1] - 1 for c in INTERVALS])

# Literal learner's ambiguity of each number, Z(x).
_lik0 = ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100
_prior0 = np.concatenate([np.full(N_RULES, 0.5 / N_RULES), np.full(N_INT, 0.5 / N_INT)])
_log_z = np.log(_prior0 @ _lik0)
LOG_Z = _log_z - _log_z.mean()  # (100,)

# beta grid for the interval block (beta's prior puts ~e^-18 mass beyond 6).
GRID_STEP = 0.02
GRID = np.arange(-2, 302) * GRID_STEP  # -0.04 .. 6.02
N_GRID = len(GRID)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0
MAX_EX = 4


def _interval_tables(sets, row_set, row_target):
    """Per beta grid point: interval log evidence per set and in-concept share per row."""
    values = np.unique(np.concatenate([np.array(s) - 1 for s in sets]))
    counts = np.zeros((len(sets), len(values)))
    for i, s in enumerate(sets):
        for x in s:
            counts[i, np.searchsorted(values, x - 1)] += 1
    n_ex = counts.sum(axis=1)
    inside = (INT_M[None, :] <= values[:, None]) & (values[:, None] <= INT_N[None, :])
    evidence = np.zeros((N_GRID, len(sets)))
    share = np.zeros((N_GRID, len(row_set)))
    for g, b in enumerate(GRID):
        w = np.exp(-b * LOG_Z)
        c = np.concatenate([[0.0], np.cumsum(w)])
        w_int = c[INT_N + 1] - c[INT_M]
        gain = inside * np.log1p((ALPHA * 100 / (1 - ALPHA)) * w[values][:, None] / w_int[None, :])
        ll = counts @ gain + n_ex[:, None] * np.log((1 - ALPHA) / 100)  # (sets, 5050)
        ev = logsumexp(ll, axis=1)
        evidence[g] = ev
        post = np.exp(ll - ev[:, None])
        share[g] = np.einsum("rh,rh->r", post[row_set], INT_MEMBERSHIP[:, row_target].T)
    return evidence, share


def _interp(table, b):
    """Catmull-Rom interpolation of table (grid, ...) at scalar b."""
    u = (b - GRID[0]) / GRID_STEP
    i = pt.clip(pt.floor(u), 1, N_GRID - 3).astype("int64")
    t = u - i
    p0, p1, p2, p3 = table[i - 1], table[i], table[i + 1], table[i + 2]
    return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2
                  + (-p0 + 3 * p1 - 3 * p2 + p3) * t**3)


with pm.Model() as model:
    ex_idx = pm.Data("ex_idx", np.zeros((1, MAX_EX), dtype="int64"))  # (sets, 4): example - 1
    ex_mask = pm.Data("ex_mask", np.zeros((1, MAX_EX)))  # 1 for a real example
    int_evidence = pm.Data("int_evidence", np.zeros((N_GRID, 1)))  # (grid, sets)
    int_share = pm.Data("int_share", np.zeros((N_GRID, 1)))  # (grid, rows)
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))  # target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    beta = pm.HalfNormal("beta", sigma=1.0)
    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    # Teacher's preference for unambiguous examples; rules: P(x | r) =
    # 0.9 [x in r] w(x) / W(r) + 0.1 / 100.
    w = pt.exp(-beta * pt.as_tensor(LOG_Z))  # (100,)
    mem = pt.as_tensor(RULE_MEMBERSHIP)
    w_rule = mem @ w  # (45,)
    in_rule = mem.T[ex_idx]  # (sets, 4, 45)
    rule_ll = (pt.log(ALPHA * in_rule * w[ex_idx][:, :, None] / w_rule[None, None, :] + (1 - ALPHA) / 100)
               * ex_mask[:, :, None]).sum(axis=1)  # (sets, 45)

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_ll
    log_ints = pt.log1p(-lambda_0) - 2 * np.log(100) + _interp(int_evidence, beta)  # (sets,)
    log_joint = pt.concatenate([log_rules, log_ints[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, 46)
    rule_part = (posterior[row_set, :N_RULES] * mem.T[row_target]).sum(axis=1)
    in_concept = rule_part + posterior[row_set, N_RULES] * _interp(int_share, beta)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    ex_idx = np.zeros((len(unique), MAX_EX), dtype="int64")
    ex_mask = np.zeros((len(unique), MAX_EX))
    for i, s in enumerate(unique):
        ex_idx[i, :len(s)] = np.array(s) - 1
        ex_mask[i, :len(s)] = 1.0
    row_set = np.array([index[s] for s in sets], dtype="int64")
    row_target = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    evidence, share = _interval_tables(unique, row_set, row_target)
    return {
        "ex_idx": ex_idx,
        "ex_mask": ex_mask,
        "int_evidence": evidence,
        "int_share": share,
        "row_set": row_set,
        "row_target": row_target,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
