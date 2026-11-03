"""Refinement of uncertain_reliability_noise_concepts: Bayesian concept learning
over 45 rules (each with its own inferred prior weight, a share lambda_0 for
the rules as a group) and every interval of 1-100, with the size principle and
an uncertain belief over the program's reliability a (Beta with mean alpha and
concentration kappa, integrated out per example set) -- but a stray example is
not a uniform draw from 1-100. People think of a stray as a number produced by
some *other* mathematical rule: a rule picked by their own rule prior, then a
member of it picked at random, q(x) = sum_r w_r 1[x in r] / |r|. So numbers
that many small rules produce (1, 2, 4, 8, 16, 64, 81, 100) are easily excused
as strays, while numbers few rules produce (e.g. 58, 87, 94) are not, and a
concept must account for them. "Yes" is the posterior probability that the
concept contains the target.

p(X | h) = E_a prod_i (a 1[x_i in h] / s + (1 - a) q(x_i))
         = Q(X) sum_j s^-j e_j(h) B(u + j, v + n - j) / B(u, v),
with Q(X) = prod_i q(x_i) (common to all concepts, so dropped), and e_j(h) the
j-th elementary symmetric polynomial of 1 / q(x_i) over the examples h
contains. Every interval contains a contiguous run of the sorted examples, so
prepare_observed counts intervals by (run, size) per set and per (set, target).
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
RULE_PER_MEMBER = RULE_MEMBERSHIP / RULE_SIZE[:, None]  # (rules, 100)

# Runs of sorted examples an interval can contain: (first, last) with first <= last.
RUNS = [(a, b) for a in range(MAX_N) for b in range(a, MAX_N)]
N_RUNS = len(RUNS)
RUN_INDEX = {r: i for i, r in enumerate(RUNS)}

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

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
    rule_in = pm.Data("rule_in", np.zeros((1, N_RULES, MAX_N)))  # (sets, rules, 4): rule contains example
    set_n_idx = pm.Data("set_n_idx", np.zeros(1, dtype="int64"))  # (sets,): n - 1
    run_count = pm.Data("run_count", np.zeros((1, N_RUNS + 1, 100)))  # (sets, run or none, size)
    cell_run_count = pm.Data("cell_run_count", np.zeros((1, N_RUNS + 1, 100)))  # containing target
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
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
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa

    # Stray distribution: a random member of a rule drawn from the learner's rule prior.
    q = pt.dot(rule_weights, RULE_PER_MEMBER)
    inv_q = example_mask / q[example_idx]  # (sets, 4)

    # Reliability-marginal weights c_j = B(u + j, v + n - j) / B(u, v) for each n.
    c_table = pt.where(_VALID, pt.exp(_betaln(u + _J, v + _N_MINUS_J) - _betaln(u, v)), 0.0)
    c_set = c_table[set_n_idx]  # (sets, j)

    # Rules: e_j over the examples each rule contains.
    e_rules = _elementary(rule_in * inv_q[:, None, :])  # (sets, rules, j)
    rule_lik = (e_rules * c_set[:, None, :] * RULE_SIZE[None, :, None] ** -_J[None]).sum(axis=2)
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + pt.log(rule_lik)

    # Intervals: e_j for each contiguous run of sorted examples (plus the empty run).
    run_in = np.zeros((N_RUNS + 1, MAX_N))
    for (a, b), i in RUN_INDEX.items():
        run_in[i, a:b + 1] = 1.0
    e_runs = _elementary(run_in[None, :, :] * inv_q[:, None, :])  # (sets, runs, j)
    run_size_lik = pt.dot(e_runs * c_set[:, None, :], _SIZE_POW.T)  # (sets, runs, size)
    int_ev = (run_count * run_size_lik).sum(axis=(1, 2))
    log_intervals = pt.log1p(-lambda_0) - pt.log(float(len(INT_LO))) + pt.log(int_ev)

    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    interval_post = pt.exp(log_intervals - log_norm)

    cell_int_share = (cell_run_count * run_size_lik[cell_set]).sum(axis=(1, 2)) / int_ev[cell_set]
    in_concept = (rule_post[cell_set] * cell_rule_member).sum(axis=1) + interval_post[cell_set] * cell_int_share
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
    for i, s in enumerate(unique):
        ex = np.sort(np.array(s))
        n = len(ex)
        set_n_idx[i] = n - 1
        example_idx[i, :n] = ex - 1
        example_mask[i, :n] = 1.0
        runs = _interval_runs(ex)
        set_runs.append(runs)
        np.add.at(run_count[i], (runs, INT_SIZE_IDX), 1.0)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_run_count = np.zeros((len(cells), N_RUNS + 1, 100))
    for j, (i, t) in enumerate(cells):
        contains = (INT_LO <= t) & (t <= INT_HI)
        np.add.at(cell_run_count[j], (set_runs[i][contains], INT_SIZE_IDX[contains]), 1.0)
    return {
        "example_idx": example_idx,
        "example_mask": example_mask,
        "rule_in": RULE_MEMBERSHIP[:, example_idx].transpose(1, 0, 2) * example_mask[:, None, :],
        "set_n_idx": set_n_idx,
        "run_count": run_count,
        "cell_run_count": cell_run_count,
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
