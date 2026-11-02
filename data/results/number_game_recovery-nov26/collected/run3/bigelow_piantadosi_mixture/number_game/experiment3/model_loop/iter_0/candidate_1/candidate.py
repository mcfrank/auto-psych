"""Bounded arithmetic progressions as candidate concepts.

People do the same Bayesian concept learning as `rule_count_familiar_strays`:
45 equally plausible mathematical rules and every magnitude interval of 1-100,
the size principle, an uncertain belief about how reliable the program is
(Beta over the per-example reliability, integrated out per set), and strays
that are either uniform numbers or familiar numbers (familiarity = how many
rules contain the number). The one change is the concept space: people also
entertain bounded arithmetic progressions, "every d-th number from a to b"
(step d >= 2, at least two members, inside 1-100, e.g. 20, 40, 60, 80 or
61, 64, 67). These programs jointly take a stated a-priori share eta
(Beta(1, 19), mean 5%) of the prior, split equally among all of them; rules
get lambda_0 (1 - eta) and intervals (1 - lambda_0)(1 - eta). Because a
progression stops at its ends, it predicts that people extend a regular
pattern to the gaps between the examples more than beyond their range.
"Yes" is the posterior probability that the concept contains the target.
"""

import math

import numpy as np
import pymc as pm
import pytensor.sparse as ps
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4


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
INT_LO = np.array([m for m in NUMBERS for n in range(m, 101)])
INT_HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INT_SIZE_IDX = INT_HI - INT_LO  # size - 1
RULE_MEMBER_RATE = RULE_MEMBERSHIP / RULE_SIZE[:, None]  # (rules, 100): pick a member of the rule
# Interval prior predictive: an interval drawn uniformly, then one of its members.
STRAY_INTERVAL_GEN = np.array([((INT_LO <= x) & (x <= INT_HI)) @ (1.0 / (INT_SIZE_IDX + 1)) for x in NUMBERS]) / len(INT_LO)


SUBSET_IN = np.array([[(m >> i) & 1 for i in range(MAX_N)] for m in range(2**MAX_N)], dtype="float64")
RULE_SIZE_IDX = RULE_SIZE.astype("int64") - 1


def progressions():
    """Every bounded arithmetic progression a, a+d, ..., b in 1-100 with d >= 2 and >= 2 members."""
    rows = []
    for d in range(2, 100):
        for a in range(1, 101 - d):
            for b in range(a + d, 101, d):
                m = np.zeros(100, dtype=bool)
                m[a - 1 : b : d] = True
                rows.append(m)
    return np.array(rows)


AP_MEMBERSHIP = progressions()  # (progressions, 100)
N_AP = AP_MEMBERSHIP.shape[0]
AP_SIZE_IDX = AP_MEMBERSHIP.sum(axis=1).astype("int64") - 1

# Runs of sorted examples an interval can contain: (first, last) with first <= last.
RUNS = [(a, b) for a in range(MAX_N) for b in range(a, MAX_N)]
N_RUNS = len(RUNS)
RUN_INDEX = {r: i for i, r in enumerate(RUNS)}
RUN_SUBSET = np.array([sum(1 << i for i in range(a, b + 1)) for a, b in RUNS] + [0])


_N = np.arange(1, MAX_N + 1, dtype="float64")[:, None]
_J = np.arange(MAX_N + 1, dtype="float64")[None, :]
_VALID = _J <= _N
_N_MINUS_J = np.maximum(_N - _J, 0.0)
_SIZES = np.arange(1, 101, dtype="float64")
_SIZE_POW = _SIZES[:, None] ** -np.arange(MAX_N + 1)[None, :]  # (size, j)


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


def _elementary(weights):
    """Elementary symmetric polynomials e_0..e_4 over the last axis (4 examples)."""
    e = [pt.ones_like(weights[..., 0])] + [pt.zeros_like(weights[..., 0])] * MAX_N
    for i in range(MAX_N):
        w = weights[..., i]
        e = [e[0]] + [e[j] + w * e[j - 1] for j in range(1, MAX_N + 1)]
    return pt.stack(e, axis=-1)


def _csr_data(name):
    """Sparse (rows, sets * subsets * sizes) count matrix bound through three pm.Data arrays."""
    data = pm.Data(f"{name}_data", np.zeros(1))
    indices = pm.Data(f"{name}_indices", np.zeros(1, dtype="int32"))
    indptr = pm.Data(f"{name}_indptr", np.array([0, 1], dtype="int32"))
    n_cols = pm.Data(f"{name}_ncols", np.ones(1, dtype="int64"))
    return ps.CSR(data, indices, indptr, pt.stack([indptr.shape[0] - 1, n_cols[0]]))


def _csr(prefix, rows, cols, vals, n_rows, n_cols):
    order = np.lexsort((cols, rows))
    rows, cols, vals = rows[order], cols[order], vals[order]
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=n_rows))])
    return {
        f"{prefix}_data": vals.astype("float64"),
        f"{prefix}_indices": cols.astype("int32"),
        f"{prefix}_indptr": indptr.astype("int32"),
        f"{prefix}_ncols": np.array([n_cols], dtype="int64"),
    }


with pm.Model() as model:
    example_idx = pm.Data("example_idx", np.zeros((1, MAX_N), dtype="int64"))  # (sets, 4): x - 1, sorted
    example_mask = pm.Data("example_mask", np.zeros((1, MAX_N)))  # (sets, 4): 1 if a real example
    rule_sub = pm.Data("rule_sub", np.zeros((1, N_RULES), dtype="int64"))  # (sets, rules): subset of examples in rule
    set_n_idx = pm.Data("set_n_idx", np.zeros(1, dtype="int64"))  # (sets,): n - 1
    run_pow = pm.Data("run_pow", np.zeros((1, N_RUNS + 1, MAX_N + 1)))  # (sets, run, j): sum over intervals of size^-j
    cell_run_pow = pm.Data("cell_run_pow", np.zeros((1, N_RUNS + 1, MAX_N + 1)))  # same, intervals containing target
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    # Progression counts by (set, subset of examples contained, size), flattened: (sets, .) and (cells, .).
    ap_count = _csr_data("apc")
    cell_ap_count = _csr_data("capc")
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    eta = pm.Beta("eta", 1, 19)  # stated a-priori share of arithmetic-progression programs
    # Equal rule prior: every rule gets lambda_0 / 45.
    rule_weights = pt.constant(np.full(N_RULES, 1.0 / N_RULES))
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa

    # Stray distribution: uniform, or a familiar number, with familiarity = how many rules contain it.
    mix_logit = pm.Normal("mix_logit", mu=0.0, sigma=1.5)
    mix = pm.Deterministic("mix", pm.math.sigmoid(mix_logit))
    q = (1 - mix) * 0.01 + mix * (RULE_MEMBERSHIP.sum(axis=0) / RULE_MEMBERSHIP.sum())
    inv_q = example_mask / q[example_idx]  # (sets, 4)

    # Reliability-marginal weights c_j = B(u + j, v + n - j) / B(u, v) for each n.
    c_table = pt.where(_VALID, pt.exp(_betaln(u + _J, v + _N_MINUS_J) - _betaln(u, v)), 0.0)
    c_set = c_table[set_n_idx]  # (sets, j)

    # e_j for every subset of a set's examples, folded with c_j and size^-j.
    e_sub = _elementary(SUBSET_IN[None, :, :] * inv_q[:, None, :])  # (sets, subsets, j)
    sub_size_lik = pt.dot(e_sub * c_set[:, None, :], _SIZE_POW.T)  # (sets, subsets, size)

    # Rules: the subset of examples each rule contains, and its size.
    rule_lik = sub_size_lik[pt.arange(rule_sub.shape[0])[:, None], rule_sub, RULE_SIZE_IDX[None, :]]
    log_rules = pt.log1p(-eta) + pt.log(lambda_0) + pt.log(rule_weights)[None, :] + pt.log(rule_lik)

    # Intervals: each contains a contiguous run of sorted examples (or none);
    # size enters only through size^-j, pre-summed per run in prepare_observed.
    run_coef = (e_sub * c_set[:, None, :])[:, RUN_SUBSET, :]  # (sets, runs, j)
    int_ev = (run_pow * run_coef).sum(axis=(1, 2))
    log_intervals = pt.log1p(-eta) + pt.log1p(-lambda_0) - 2 * np.log(100) + pt.log(int_ev)

    # Progressions: counts by (subset of examples contained, size), pre-tabulated.
    lik_flat = sub_size_lik.reshape((-1, 1))
    ap_ev = ps.structured_dot(ap_count, lik_flat)[:, 0]
    log_ap = pt.log(eta) - np.log(N_AP) + pt.log(ap_ev)

    log_joint = pt.concatenate([log_rules, log_intervals[:, None], log_ap[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    interval_post = pt.exp(log_intervals - log_norm)
    ap_post = pt.exp(log_ap - log_norm)

    cell_int_share = (cell_run_pow * run_coef[cell_set]).sum(axis=(1, 2)) / int_ev[cell_set]
    cell_ap_share = ps.structured_dot(cell_ap_count, lik_flat)[:, 0] / ap_ev[cell_set]
    in_concept = (
        (rule_post[cell_set] * cell_rule_member).sum(axis=1)
        + interval_post[cell_set] * cell_int_share
        + ap_post[cell_set] * cell_ap_share
    )
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-9, 1 - 1e-9)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _interval_runs(ex):
    """Index of the run of sorted examples each interval contains (N_RUNS if none)."""
    inside = (ex[None, :] >= INT_LO[:, None]) & (ex[None, :] <= INT_HI[:, None])
    runs = np.full(len(INT_LO), N_RUNS, dtype="int64")
    any_in = inside.any(axis=1)
    first = inside.argmax(axis=1)
    last = len(ex) - 1 - inside[:, ::-1].argmax(axis=1)
    for i in np.flatnonzero(any_in):
        runs[i] = RUN_INDEX[(first[i], last[i])]
    return runs


def _set_triplets(tabs):
    rows, cols, vals = [], [], []
    for i, tab in enumerate(tabs):
        nz = np.flatnonzero(tab)
        rows.append(np.full(len(nz), i)), cols.append(nz), vals.append(tab[nz])
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    example_idx = np.zeros((S, MAX_N), dtype="int64")
    example_mask = np.zeros((S, MAX_N))
    set_n_idx = np.zeros(S, dtype="int64")
    run_count = np.zeros((S, N_RUNS + 1, 100))
    set_runs = []
    n_flat = 2**MAX_N * 100
    ap_tabs = []
    set_ap_sub = []
    for i, s in enumerate(unique):
        ex = np.sort(np.array(s))
        n = len(ex)
        set_n_idx[i] = n - 1
        example_idx[i, :n] = ex - 1
        example_mask[i, :n] = 1.0
        runs = _interval_runs(ex)
        set_runs.append(runs)
        np.add.at(run_count[i], (runs, INT_SIZE_IDX), 1.0)
        ap_sub = AP_MEMBERSHIP[:, ex - 1].astype("int64") @ (2 ** np.arange(n))
        set_ap_sub.append(ap_sub)
        ap_tabs.append(np.bincount(i * n_flat + ap_sub * 100 + AP_SIZE_IDX))
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_run_count = np.zeros((len(cells), N_RUNS + 1, 100))
    c_rows, c_cols, c_vals = [], [], []
    for j, (i, t) in enumerate(cells):
        has_t = AP_MEMBERSHIP[:, t - 1]
        tab = np.bincount(set_ap_sub[i][has_t] * 100 + AP_SIZE_IDX[has_t])
        nz = np.flatnonzero(tab)
        c_rows.append(np.full(len(nz), j)), c_cols.append(i * n_flat + nz), c_vals.append(tab[nz])
        contains = (INT_LO <= t) & (t <= INT_HI)
        np.add.at(cell_run_count[j], (set_runs[i][contains], INT_SIZE_IDX[contains]), 1.0)
    return {
        "example_idx": example_idx,
        "example_mask": example_mask,
        "rule_sub": (
            (RULE_MEMBERSHIP[:, example_idx].transpose(1, 0, 2) * example_mask[:, None, :])
            @ (2 ** np.arange(MAX_N))
        ).astype("int64"),
        "set_n_idx": set_n_idx,
        "run_pow": run_count @ _SIZE_POW,
        "cell_run_pow": cell_run_count @ _SIZE_POW,
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        **_csr("apc", *_set_triplets(ap_tabs), S, S * n_flat),
        **_csr("capc", np.concatenate(c_rows), np.concatenate(c_cols), np.concatenate(c_vals), len(cells), S * n_flat),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
