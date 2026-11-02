"""Cluster-cued range salience.

Hypothesis: which kind of program comes to mind depends on how the examples look.
People do Bayesian concept learning over equally plausible mathematical rules,
rules held within a magnitude range, and magnitude ranges of 1-100 (size
principle, uncertain reliability, strays sometimes familiar numbers by rule
count), as in range_limited_rule_familiar_strays. The one change is that a
magnitude range is brought to mind more readily the more tightly the examples
cluster on the number line: the prior weight of the range hypotheses is
multiplied by exp(beta * tightness), tightness = 1 - span/99 for sets of two or
more examples (a single number shows no cluster, so tightness 0). With beta > 0,
when examples sharing a rule sit close together, people fall back on "numbers
around here" and accept the rule's non-members inside the span more than
the size principle alone implies, while widely spread sets keep the rule.
"""

import math

import numpy as np
import pymc as pm
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


with pm.Model() as model:
    example_idx = pm.Data("example_idx", np.zeros((1, MAX_N), dtype="int64"))  # (sets, 4): x - 1, sorted
    example_mask = pm.Data("example_mask", np.zeros((1, MAX_N)))  # (sets, 4): 1 if a real example
    rule_sub = pm.Data("rule_sub", np.zeros((1, N_RULES), dtype="int64"))  # (sets, rules): subset of examples in rule
    set_n_idx = pm.Data("set_n_idx", np.zeros(1, dtype="int64"))  # (sets,): n - 1
    run_pow = pm.Data("run_pow", np.zeros((1, N_RUNS + 1, MAX_N + 1)))  # (sets, run, j): sum over intervals of size^-j
    cell_run_pow = pm.Data("cell_run_pow", np.zeros((1, N_RUNS + 1, MAX_N + 1)))  # same, intervals containing target
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    rr_pow = pm.Data("rr_pow", np.zeros((1, 2**MAX_N, MAX_N + 1)))  # (sets, subset, j): restricted rules, sum of size^-j
    cell_rr_pow = pm.Data("cell_rr_pow", np.zeros((1, 2**MAX_N, MAX_N + 1)))  # same, containing the target
    set_tight = pm.Data("set_tight", np.zeros(1))  # (sets,): 1 - span/99 for n >= 2, else 0
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
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
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + pt.log(rule_lik)

    # Intervals: each contains a contiguous run of sorted examples (or none);
    # size enters only through size^-j, pre-summed per run in prepare_observed.
    run_coef = (e_sub * c_set[:, None, :])[:, RUN_SUBSET, :]  # (sets, runs, j)
    int_ev = (run_pow * run_coef).sum(axis=(1, 2))
    beta_tight = pm.Normal("beta_tight", mu=0.0, sigma=2.0)
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + pt.log(int_ev) + beta_tight * set_tight

    # Rules held within a range: share rho of the rule prior, each rule then each range of 1-100.
    rho_logit = pm.Normal("rho_logit", mu=-2.0, sigma=1.0)
    rho = pm.Deterministic("rho", pm.math.sigmoid(rho_logit))
    log_rules = log_rules + pt.log1p(-rho)
    sub_coef = e_sub * c_set[:, None, :]  # (sets, subsets, j)
    rr_ev = (rr_pow * sub_coef).sum(axis=(1, 2))
    log_rr = pt.log(lambda_0) + pt.log(rho) - np.log(N_RULES * len(INT_LO)) + pt.log(rr_ev)

    log_joint = pt.concatenate([log_rules, log_intervals[:, None], log_rr[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    interval_post = pt.exp(log_intervals - log_norm)
    rr_post = pt.exp(log_rr - log_norm)

    cell_int_share = (cell_run_pow * run_coef[cell_set]).sum(axis=(1, 2)) / int_ev[cell_set]
    in_concept = (rule_post[cell_set] * cell_rule_member).sum(axis=1) + interval_post[cell_set] * cell_int_share
    cell_rr_share = (cell_rr_pow * sub_coef[cell_set]).sum(axis=(1, 2)) / rr_ev[cell_set]
    in_concept = in_concept + rr_post[cell_set] * cell_rr_share
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


def _restricted_blocks():
    """Every proper, non-empty block of a rule's sorted members a range of 1-100 can cut out."""
    rule, lo, hi, size, mult = [], [], [], [], []
    for r, c in enumerate(RULES):
        m = np.sort(np.array(c))
        k = len(m)
        gaps = np.diff(np.concatenate([[0], m, [101]]))  # gaps[i]: choices of boundary before member i
        for a in range(k):
            for b in range(a, k):
                if a == 0 and b == k - 1:
                    continue  # the whole rule is the plain rule
                rule.append(r)
                lo.append(m[a])
                hi.append(m[b])
                size.append(b - a + 1)
                mult.append(gaps[a] * gaps[b + 1])
    return (np.array(rule), np.array(lo), np.array(hi), np.array(size) - 1, np.array(mult, dtype="float64"))


BLK_RULE, BLK_LO, BLK_HI, BLK_SIZE_IDX, BLK_MULT = _restricted_blocks()


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
    rr_count = np.zeros((S, 2**MAX_N, 100))
    set_blk_sub = []
    for i, s in enumerate(unique):
        ex = np.sort(np.array(s))
        n = len(ex)
        set_n_idx[i] = n - 1
        example_idx[i, :n] = ex - 1
        example_mask[i, :n] = 1.0
        runs = _interval_runs(ex)
        set_runs.append(runs)
        np.add.at(run_count[i], (runs, INT_SIZE_IDX), 1.0)
        sub = np.zeros(len(BLK_RULE), dtype="int64")
        for e_i, x in enumerate(ex):
            inb = (RULE_MEMBERSHIP[BLK_RULE, x - 1] > 0) & (BLK_LO <= x) & (x <= BLK_HI)
            sub += inb.astype("int64") << e_i
        set_blk_sub.append(sub)
        np.add.at(rr_count[i], (sub, BLK_SIZE_IDX), BLK_MULT)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_run_count = np.zeros((len(cells), N_RUNS + 1, 100))
    cell_rr_count = np.zeros((len(cells), 2**MAX_N, 100))
    for j, (i, t) in enumerate(cells):
        hit = (RULE_MEMBERSHIP[BLK_RULE, t - 1] > 0) & (BLK_LO <= t) & (t <= BLK_HI)
        np.add.at(cell_rr_count[j], (set_blk_sub[i][hit], BLK_SIZE_IDX[hit]), BLK_MULT[hit])
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
        "rr_pow": rr_count @ _SIZE_POW,
        "cell_rr_pow": cell_rr_count @ _SIZE_POW,
        "set_tight": np.array([(1 - (max(u) - min(u)) / 99.0) if len(u) > 1 else 0.0 for u in unique]),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
