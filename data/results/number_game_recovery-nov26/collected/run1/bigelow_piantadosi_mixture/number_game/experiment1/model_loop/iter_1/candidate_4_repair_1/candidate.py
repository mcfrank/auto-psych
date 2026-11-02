"""Refinement of bigelow_piantadosi_independent: Bayesian concept learning
over 45 rules and all intervals [m, n] of 1-100 (size principle, 10% of
examples uniform on 1-100, a prior weight inferred for each rule, a share
lambda_0 for the rules as a group), but the interval's two endpoints are drawn
from a power-law (log-compressed number line) distribution P(x) ~ x^-gamma
over 1..100 instead of uniformly, with gamma fitted: interval [m, n] has prior
(1 - lambda_0) * (m n)^-gamma / (sum_x x^-gamma)^2 (a pair with m > n is the
empty set, no concept). gamma = 0 is the incumbent. "Yes" is the posterior
probability that the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
LOG_X = np.log(NUMBERS.astype("float64"))


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
RULE_EXAMPLE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

# Interval [m, n] lives at cell (m-1, n-1) of a 100 x 100 grid; m > n cells
# are empty. Its prior depends on gamma only through (m n)^-gamma, so per set
# prepare_observed groups intervals into N_BINS bins of log(m n) and stores,
# for each target t (and, in row 100, for all intervals), the summed evidence
# of intervals containing t in each bin times delta^j / j!, j = 0, 1, 2, with
# delta = log(m n) - bin centre. Then sum_I P(D | I) (m n)^-gamma is
# sum_k exp(-gamma c_k) sum_j (-gamma)^j H_kj: a second-order expansion within
# each bin, exact to ~1e-5 relative for |gamma| <= 2.
M_GRID, N_GRID = np.meshgrid(NUMBERS, NUMBERS, indexing="ij")
VALID = M_GRID <= N_GRID
SIZE_GRID = np.where(VALID, N_GRID - M_GRID + 1, 1)
LOG_MN = np.log(M_GRID * N_GRID.astype("float64"))
N_BINS = 64
EDGES = np.linspace(0.0, np.log(10000.0) + 1e-9, N_BINS + 1)
CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])
BIN = np.clip(np.digitize(LOG_MN, EDGES) - 1, 0, N_BINS - 1)
DELTA = LOG_MN - CENTRES[BIN]


def interval_log_lik(examples):
    ll = np.zeros((100, 100))
    for x in examples:
        inside = (M_GRID <= x) & (x <= N_GRID)
        ll += np.log(ALPHA * inside / SIZE_GRID + (1 - ALPHA) / 100)
    return ll


def interval_moments(examples):
    """(101, N_BINS * 3) moments and the log scale of the set's evidence."""
    ll = interval_log_lik(examples)
    scale = ll[VALID].max()
    evidence = np.where(VALID, np.exp(ll - scale), 0.0)
    out = np.zeros((101, N_BINS, 3))
    diag = np.arange(100)
    for j, fact in enumerate([1.0, 1.0, 2.0]):
        weighted = evidence * DELTA**j / fact
        for k in range(N_BINS):
            w = np.where(BIN == k, weighted, 0.0)
            # sum over m <= t <= n via cumsums over n (from the top) and m.
            contains = np.cumsum(np.cumsum(w[:, ::-1], axis=1)[:, ::-1], axis=0)[diag, diag]
            out[:100, k, j] = contains
            out[100, k, j] = w.sum()
    return out.reshape(101, N_BINS * 3), scale


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    interval_mass = pm.Data("interval_mass", np.zeros((1, 101, N_BINS * 3)))  # (sets, targets + all, bins*3)
    interval_log_scale = pm.Data("interval_log_scale", np.zeros(1))  # (sets,)
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
    # Power-law exponent of the endpoint distribution P(x) ~ x^-gamma (0 = uniform).
    gamma = pm.Normal("gamma", mu=0.0, sigma=1.0)

    log_z_endpoint = pt.logsumexp(-gamma * LOG_X)
    bin_weight = pt.exp(-gamma * CENTRES - 2 * log_z_endpoint)  # (bins,)
    w = pt.stack([bin_weight, -gamma * bin_weight, gamma**2 * bin_weight], axis=1).reshape((-1,))
    n_sets = interval_mass.shape[0]
    mass = pt.dot(interval_mass.reshape((n_sets * 101, N_BINS * 3)), w).reshape((n_sets, 101))
    total_mass = mass[:, 100]

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik  # (S, R)
    log_interval = pt.log1p(-lambda_0) + pt.log(total_mass) + interval_log_scale  # (S,)
    log_joint = pt.concatenate([log_rules, log_interval[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1, keepdims=True)
    post_rules = pt.exp(log_rules - log_norm)
    post_interval = pt.exp(log_interval[:, None] - log_norm)  # (S, 1)
    in_concept = post_rules @ pt.as_tensor(RULE_MEMBERSHIP) + post_interval * mass[:, :100] / total_mass[:, None]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-9, 1.0 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    moments = [interval_moments(s) for s in unique]
    return {
        "rule_log_lik": np.stack([RULE_EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "interval_mass": np.stack([m for m, _ in moments]),
        "interval_log_scale": np.array([sc for _, sc in moments]),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
