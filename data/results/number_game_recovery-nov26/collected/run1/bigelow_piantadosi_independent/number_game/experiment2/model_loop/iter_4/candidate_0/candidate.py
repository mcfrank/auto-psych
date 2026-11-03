"""Adjacent-number confusion of the target in exception-tolerant tempered concept learning.

People infer the program's concept like the exception-tolerant, tempered Bayesian
learner of `final_digit_salient_rules` (45 rules, final-digit rules boosted by
kappa; Erlang-width intervals on a compressed number line), but they hold the
number asked about only approximately: with fitted probability nu they judge an
adjacent number (target - 1 or target + 1, equally often; the one existing
neighbour at 1 or 100) instead of the target. So a target next to any example
borrows that example's endorsement whatever its parity or rule membership, and
people endorse examples' immediate neighbours more than membership implies.
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


RULE_NAMES = list(rule_concepts().keys())
RULES = list(rule_concepts().values())
# Rules read straight off the final digit: parity and "ends in d".
RULE_SALIENT = np.array([n in ("odd", "even") or n.startswith("ends in") for n in RULE_NAMES], dtype="float64")
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
# All intervals [a, b] of 1..100.
INT_A = np.array([a for a in NUMBERS for b in NUMBERS if b >= a])
INT_B = np.array([b for a in NUMBERS for b in NUMBERS if b >= a])
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
INT_SIZE_IDX = INT_B - INT_A  # width - 1
SIZES = np.arange(1, 101, dtype="float64")
LOG_SIZE_COUNT = np.log(101 - SIZES)  # intervals of each width
INT_FLAT = (INT_A - 1) * 100 + (INT_B - 1)  # position of [a, b] in a 100 x 100 grid


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples inside the rule
    set_n = pm.Data("set_n", np.ones(1))  # (sets,): number of examples
    int_k = pm.Data("int_k", np.zeros((1, len(INT_A))))  # (sets, intervals): examples inside
    int_log_x = pm.Data("int_log_x", np.zeros((1, len(INT_A))))  # sum of log x over those examples
    # Cells (set, target) whose concept membership is needed.
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.ones(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    # Per observed (set, target): the cell of the target itself and of its two neighbours.
    obs_self = pm.Data("obs_self", np.zeros(1, dtype="int64"))
    obs_lo = pm.Data("obs_lo", np.zeros(1, dtype="int64"))
    obs_hi = pm.Data("obs_hi", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # row -> observed (set, target)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)  # sharpness of the size principle
    eps = pm.Beta("eps", 2, 5)  # believed rate of off-concept examples
    gamma = pm.HalfNormal("gamma", sigma=1.0)  # compression of the mental number line
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)  # typical interval width
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)  # log prior boost of final-digit rules
    nu = pm.Beta("nu", 1, 4)  # probability of judging an adjacent number instead of the target

    n_sets = set_n.shape[0]
    log_noise = pt.log(eps) - np.log(100)
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)  # (rules,)
    rule_log_lik = rule_k * rule_in[None, :] + (set_n[:, None] - rule_k) * log_noise
    log_rule_prior = kappa * RULE_SALIENT - pm.math.logsumexp(kappa * RULE_SALIENT)
    log_rules = pt.log(lambda_0) + log_rule_prior[None, :] + beta * rule_log_lik
    shift = pt.max(log_rules, axis=1)  # per-set stabiliser
    w_rules = pt.exp(log_rules - shift[:, None])

    # Interval members are drawn with weight x^-gamma: P(x | [a, b]) = x^-gamma / W(a, b).
    cum_w = pt.concatenate([pt.zeros(1), pt.cumsum(pt.exp(-gamma * LOG_NUMBERS))])
    log_w = pt.log(cum_w[INT_B] - cum_w[INT_A - 1])  # (intervals,)
    int_lik = int_k * (pt.log1p(-eps) - log_w[None, :]) - gamma * int_log_x + (set_n[:, None] - int_k) * log_noise
    # Erlang width prior: log P(interval of width s) = log s - s / sigma - log Z.
    log_width = pt.log(SIZES) - SIZES / sigma
    log_z = pm.math.logsumexp(LOG_SIZE_COUNT + log_width)
    log_size_prior = pt.log1p(-lambda_0) + log_width - log_z  # (widths,)
    w_int = pt.exp(log_size_prior[INT_SIZE_IDX][None, :] + beta * int_lik - shift[:, None])  # (sets, intervals)

    # Membership of t in intervals: sum of w over a <= t <= b, via 2D cumulative sums on the (a, b) grid.
    grid = pt.set_subtensor(pt.zeros((n_sets, 10000))[:, INT_FLAT], w_int).reshape((n_sets, 100, 100))
    above = pt.cumsum(grid[:, :, ::-1], axis=2)[:, :, ::-1]  # sum over b' >= b
    covered = pt.cumsum(above, axis=1).reshape((n_sets, 10000))  # sum over a' <= a
    int_member = covered[cell_set, (cell_target - 1) * 101]

    z = w_rules.sum(axis=1) + w_int.sum(axis=1)  # (sets,)
    in_concept = ((w_rules[cell_set] * cell_rule_member).sum(axis=1) + int_member) / z[cell_set]

    judged = (1 - nu) * in_concept[obs_self] + nu * 0.5 * (in_concept[obs_lo] + in_concept[obs_hi])
    p_yes = pm.Deterministic("p_yes", pt.clip(judged[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _neighbours(t):
    lo, hi = t - 1, t + 1
    if lo < 1:
        lo = hi
    if hi > 100:
        hi = lo
    return lo, hi


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    trial_cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    observed = list(dict.fromkeys(trial_cells))
    obs_index = {c: i for i, c in enumerate(observed)}
    needed = []
    for i, t in observed:
        lo, hi = _neighbours(t)
        needed += [(i, t), (i, lo), (i, hi)]
    cells = list(dict.fromkeys(needed))
    cell_index = {c: i for i, c in enumerate(cells)}

    int_k, int_log_x = [], []
    for st in unique:
        ex = np.array(st)
        inside = (INT_A[:, None] <= ex[None, :]) & (INT_B[:, None] >= ex[None, :])
        int_k.append(inside.sum(axis=1).astype("float64"))
        int_log_x.append((inside * np.log(ex)[None, :]).sum(axis=1))

    return {
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "int_k": np.stack(int_k),
        "int_log_x": np.stack(int_log_x),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_target": np.array([t for _, t in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "obs_self": np.array([cell_index[c] for c in observed], dtype="int64"),
        "obs_lo": np.array([cell_index[(i, _neighbours(t)[0])] for i, t in observed], dtype="int64"),
        "obs_hi": np.array([cell_index[(i, _neighbours(t)[1])] for i, t in observed], dtype="int64"),
        "row_cell": np.array([obs_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
