"""Spread-coverage expectation (refines the best model's concept learner).

People expect the program's examples to be spread out across the whole concept
rather than bunched in one corner of it. Over the same hypothesis space as
digit_sum_open_ended_range_discount (45 arithmetic and digit rules, 18
digit-sum concepts, every interval of 1..100 with a width-graded and
open-ended-discounted prior; distinct examples drawn without replacement, 10%
strays, a guessing lapse), each concept's likelihood is tilted by how much of
it the examples cover: with r = (rank span of the member examples) / (size - 1),
the examples' configuration has probability proportional to r**kappa under
the without-replacement range distribution, exactly normalised for the
concept's size and the number of member examples. kappa > 0 credits concepts
the examples reach across; kappa = 0 is the best model. Single examples are
unaffected.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import gammaln, logsumexp

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
    return list(rules.values())


RULES = rule_concepts()
N_RULES = len(RULES)
DIGIT_SUM = np.array([sum(int(c) for c in str(n)) for n in NUMBERS])
SUM_CONCEPTS = [list(NUMBERS[DIGIT_SUM == s]) for s in range(1, 19)]
N_SUMS = len(SUM_CONCEPTS)
N_FINITE = N_RULES + N_SUMS
FINITE_MEMBERS = np.array([np.isin(NUMBERS, c) for c in RULES + SUM_CONCEPTS], dtype="float64")
FINITE_SIZE = FINITE_MEMBERS.sum(axis=1)
FINITE_RANK = np.cumsum(FINITE_MEMBERS, axis=1)
INTERVALS = [(m, n) for m in NUMBERS for n in range(m, 101)]
_LO = np.array([a for a, _ in INTERVALS])
_HI = np.array([b for _, b in INTERVALS])
_WIDTH = _HI - _LO + 1
_OPEN = ((_LO == 1) | (_HI == 100)).astype(int)
_GROUP_KEYS = sorted(set(zip(_WIDTH.tolist(), _OPEN.tolist())))
_GROUP_OF = {g: i for i, g in enumerate(_GROUP_KEYS)}
INTERVAL_GROUP = np.array([_GROUP_OF[(s, o)] for s, o in zip(_WIDTH, _OPEN)])
N_GROUPS = len(_GROUP_KEYS)
GROUP_WIDTH = np.array([g[0] for g in _GROUP_KEYS], dtype="float64")
GROUP_OPEN = np.array([g[1] for g in _GROUP_KEYS], dtype="float64")
GROUP_COUNT = np.bincount(INTERVAL_GROUP, minlength=N_GROUPS).astype("float64")
INTERVAL_MEMBERS = ((NUMBERS[None, :] >= _LO[:, None]) & (NUMBERS[None, :] <= _HI[:, None])).astype("float64")

# Without-replacement distribution of the rank span d of n distinct draws from a
# concept of size S: P(d) = (S - d) C(d-1, n-2) / C(S, n). Table axes (S, n, d).
MAX_N = 4


def _log_choose(a, b):
    return gammaln(a + 1) - gammaln(b + 1) - gammaln(a - b + 1)


SPAN_LOGP = np.full((101, MAX_N + 1, 100), -1e4)
SPAN_LOGR = np.zeros((101, MAX_N + 1, 100))
for _s in range(2, 101):
    for _n in range(2, min(MAX_N, _s) + 1):
        _d = np.arange(_n - 1, _s)
        SPAN_LOGP[_s, _n, _d] = np.log(_s - _d) + _log_choose(_d - 1, _n - 2) - _log_choose(_s, _n)
        SPAN_LOGR[_s, _n, _d] = np.log(_d / (_s - 1))
SPAN_ACTIVE = (SPAN_LOGP.max(axis=2) > -1e3).astype("float64")  # (S, n) pairs with a coverage term


def wor_log_lik(examples, members, size):
    """log P(examples | h) for the rows of `members`, without replacement with 10% strays."""
    total = np.zeros(len(members))
    shown = np.zeros(len(members))
    for x in examples:
        member = members[:, x - 1]
        total += np.log(ALPHA * member / np.maximum(size - shown, 1.0) + (1 - ALPHA) / 100)
        shown += member
    return total


with pm.Model() as model:
    # Finite concepts (rules then digit sums): base log-lik, log coverage and table index.
    finite_ll = pm.Data("finite_ll", np.zeros((1, N_FINITE)))  # (sets, 63)
    finite_logr = pm.Data("finite_logr", np.zeros((1, N_FINITE)))
    finite_tidx = pm.Data("finite_tidx", np.zeros((1, N_FINITE), dtype="int64"))
    finite_in = pm.Data("finite_in", np.zeros((1, N_FINITE)))  # (cells, 63) target membership
    # Intervals collapsed per set into (width-group, covered-run) columns.
    col_ll = pm.Data("col_ll", np.zeros((1, 1)))  # (sets, cols) log-sum of base lik in column
    col_logr = pm.Data("col_logr", np.zeros((1, 1)))
    col_tidx = pm.Data("col_tidx", np.zeros((1, 1), dtype="int64"))
    col_group = pm.Data("col_group", np.zeros((1, 1), dtype="int64"))
    col_in = pm.Data("col_in", np.zeros((1, 1)))  # (cells, cols) within-column membership
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 1.0)
    log_sum_share = pm.Normal("log_sum_share", -3.0, 1.5)
    log_open = pm.Normal("log_open", 0.0, 1.0)
    # Strength of the expectation that examples spread across the concept.
    kappa = pm.Normal("kappa", 0.0, 0.5)

    # Log normaliser E[r**kappa] for each (size, n); zero where no coverage term applies.
    log_norm = pt.logsumexp(SPAN_LOGP + kappa * SPAN_LOGR, axis=2) * SPAN_ACTIVE
    log_norm_flat = log_norm.reshape((-1,))

    # Finite concepts.
    f_ll = finite_ll + kappa * finite_logr - log_norm_flat[finite_tidx]
    rule_ll = f_ll[:, :N_RULES]
    sum_ll = f_ll[:, N_RULES:]
    rule_ev = pt.logsumexp(rule_ll, axis=1)
    sum_ev = pt.logsumexp(sum_ll, axis=1)
    rule_post = pt.exp(rule_ll - rule_ev[:, None])[cell_set]
    sum_post = pt.exp(sum_ll - sum_ev[:, None])[cell_set]
    rule_in_concept = (rule_post * finite_in[:, :N_RULES]).sum(axis=1)
    sum_in_concept = (sum_post * finite_in[:, N_RULES:]).sum(axis=1)

    # Intervals.
    log_w = gamma * np.log(GROUP_WIDTH) + log_open * GROUP_OPEN
    log_prior = log_w - pt.logsumexp(log_w + np.log(GROUP_COUNT))
    weighted = col_ll + kappa * col_logr - log_norm_flat[col_tidx] + log_prior[col_group]
    interval_ev = pt.logsumexp(weighted, axis=1)
    col_post = pt.exp(weighted - interval_ev[:, None])
    interval_in_concept = (col_post[cell_set] * col_in).sum(axis=1)

    log_rules_only = rule_ev - np.log(N_RULES)
    log_sums_only = sum_ev - np.log(N_SUMS) + log_sum_share
    p_sum = pm.math.sigmoid(log_sums_only - log_rules_only)[cell_set]
    finite_in_concept = (1 - p_sum) * rule_in_concept + p_sum * sum_in_concept
    log_finite = (
        pt.log(lambda_0) + pt.logaddexp(log_rules_only, log_sums_only) - pt.logaddexp(0.0, log_sum_share)
    )
    log_intervals = pt.log1p(-lambda_0) + interval_ev
    p_rule = pm.math.sigmoid(log_finite - log_intervals)[cell_set]
    in_concept = p_rule * finite_in_concept + (1 - p_rule) * interval_in_concept

    p_concept = pt.clip(in_concept[row_cell], 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _coverage(examples, members, rank, size):
    """Per concept: log span fraction of the member examples and the (size, n) table index."""
    x = np.array(examples) - 1
    mem = members[:, x] > 0
    n = mem.sum(axis=1)
    rk = rank[:, x]
    d = np.where(n > 0, np.where(mem, rk, -1).max(axis=1) - np.where(mem, rk, 1e9).min(axis=1), 0)
    active = (n >= 2) & (size >= 2)
    logr = np.where(active, np.log(np.maximum(d, 1.0) / np.maximum(size - 1, 1.0)), 0.0)
    tidx = np.where(active, size.astype(int) * (MAX_N + 1) + np.minimum(n, MAX_N), 0)
    return logr, tidx


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    flat = [index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)]
    cells = sorted(set(flat))
    cell_of = {c: i for i, c in enumerate(cells)}
    c_set = np.array([c // 100 for c in cells], dtype="int64")
    c_tgt = np.array([c % 100 for c in cells], dtype="int64")

    f_ll, f_logr, f_tidx = [], [], []
    columns = []  # per set: list of (ll, logr, tidx, group, membership over 100)
    for s in unique:
        f_ll.append(wor_log_lik(s, FINITE_MEMBERS, FINITE_SIZE))
        lr_, ti_ = _coverage(s, FINITE_MEMBERS, FINITE_RANK, FINITE_SIZE)
        f_logr.append(lr_)
        f_tidx.append(ti_)
        ill = wor_log_lik(s, INTERVAL_MEMBERS, _WIDTH.astype(float))
        x = np.array(s)
        mem = (x[None, :] >= _LO[:, None]) & (x[None, :] <= _HI[:, None])
        n = mem.sum(axis=1)
        span = np.where(mem, x, -1).max(axis=1) - np.where(mem, x, 1000).min(axis=1)
        active = (n >= 2) & (_WIDTH >= 2)
        span = np.where(active, span, 0).astype(int)
        n_key = np.where(active, n, 0)
        keys = INTERVAL_GROUP * 10000 + n_key * 100 + span
        cols = []
        for k in np.unique(keys):
            mask = keys == k
            g, nk, sp = k // 10000, (k // 100) % 100, k % 100
            w = GROUP_WIDTH[g]
            ev = logsumexp(ill[mask])
            member = np.exp(ill[mask] - ev) @ INTERVAL_MEMBERS[mask]
            if nk >= 2:
                cols.append((ev, np.log(sp / (w - 1)), int(w) * (MAX_N + 1) + min(nk, MAX_N), g, member))
            else:
                cols.append((ev, 0.0, 0, g, member))
        columns.append(cols)
    n_cols = max(len(c) for c in columns)
    S = len(unique)
    col_ll = np.full((S, n_cols), -1e4)
    col_logr = np.zeros((S, n_cols))
    col_tidx = np.zeros((S, n_cols), dtype="int64")
    col_group = np.zeros((S, n_cols), dtype="int64")
    col_member = np.zeros((S, n_cols, 100))
    for i, cols in enumerate(columns):
        for j, (ev, lr_, ti_, g, member) in enumerate(cols):
            col_ll[i, j], col_logr[i, j], col_tidx[i, j], col_group[i, j] = ev, lr_, ti_, g
            col_member[i, j] = member
    return {
        "finite_ll": np.array(f_ll),
        "finite_logr": np.array(f_logr),
        "finite_tidx": np.array(f_tidx, dtype="int64"),
        "finite_in": FINITE_MEMBERS[:, c_tgt].T.copy(),
        "col_ll": col_ll,
        "col_logr": col_logr,
        "col_tidx": col_tidx,
        "col_group": col_group,
        "col_in": col_member[c_set, :, c_tgt],
        "cell_set": c_set,
        "row_cell": np.array([cell_of[c] for c in flat], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


SAMPLER_SETTINGS = {"target_accept": 0.9}
