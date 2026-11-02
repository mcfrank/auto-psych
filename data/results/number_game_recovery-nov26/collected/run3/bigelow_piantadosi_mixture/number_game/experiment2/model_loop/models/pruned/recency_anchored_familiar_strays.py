"""Recency-anchored concept learning with familiar-number strays.

Hypothesis: the most recent example anchors the concept. People trust the last
number shown to come from the program's concept at its own, inferred level of
trust (a_last), while the earlier examples, fading in memory, are pooled into an
uncertain belief about how often the program produces a stray. Otherwise as in
familiar_strays_equal_rule_prior: Bayesian concept learning over 45 equally
plausible rules (share lambda_0) and every interval of 1-100 ((1 - lambda_0) /
100^2 each), the size principle, and strays that are with inferred probability
mix a "familiar-rule number" (an equally chosen rule, then one of its members)
and otherwise uniform on 1-100:
q(x) = (1 - mix) / 100 + mix * mean_r 1[x in r] / |r|.
So an off-pattern number shown last constrains the concept differently from the
same number shown earlier, and the same set in another order generalises
differently.

p(X | h) / Q(X) = [(1 - a_last) + a_last 1[x_n in h] / (s q(x_n))]
                  * sum_j s^-j e_j(h, earlier) B(u + j, v + m - j) / B(u, v),
m = n - 1, e_j the elementary symmetric polynomial of 1 / q(x_i) over the
earlier examples h contains. Examples are stored sorted (intervals contain
contiguous runs of them) with a one-hot flag for the last-shown example.
"Yes" is the posterior probability that the concept contains the target.
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
STRAY_RULE_GEN = (RULE_MEMBERSHIP / RULE_SIZE[:, None]).mean(axis=0)  # (100,): equal rules, then a member

SUBSET_IN = np.array([[(m >> i) & 1 for i in range(MAX_N)] for m in range(2**MAX_N)], dtype="float64")
RULE_SIZE_IDX = RULE_SIZE.astype("int64") - 1

# Runs of sorted examples an interval can contain: (first, last) with first <= last.
RUNS = [(a, b) for a in range(MAX_N) for b in range(a, MAX_N)]
N_RUNS = len(RUNS)
RUN_INDEX = {r: i for i, r in enumerate(RUNS)}
RUN_SUBSET = np.array([sum(1 << i for i in range(a, b + 1)) for a, b in RUNS] + [0])


_N = np.arange(0, MAX_N, dtype="float64")[:, None]  # m = earlier (non-anchor) examples
_J = np.arange(MAX_N + 1, dtype="float64")[None, :]
_VALID = _J <= _N
_N_MINUS_J = np.maximum(_N - _J, 0.0)
_SIZES = np.arange(1, 101, dtype="float64")
_SIZE_POW = _SIZES[:, None] ** -np.arange(MAX_N + 2)[None, :]  # (size, j): j = 0..5
_SIZE_POW_J = _SIZE_POW[:, : MAX_N + 1]


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
    anchor_onehot = pm.Data("anchor_onehot", np.zeros((1, MAX_N)))  # (sets, 4): sorted slot of last-shown example
    rule_sub = pm.Data("rule_sub", np.zeros((1, N_RULES), dtype="int64"))  # (sets, rules): subset of examples in rule
    set_n_idx = pm.Data("set_n_idx", np.zeros(1, dtype="int64"))  # (sets,): m = n - 1 earlier examples
    run_pow = pm.Data("run_pow", np.zeros((1, N_RUNS + 1, MAX_N + 2)))  # (sets, run, j): sum over intervals of size^-j
    cell_run_pow = pm.Data("cell_run_pow", np.zeros((1, N_RUNS + 1, MAX_N + 2)))  # same, intervals containing target
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
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

    # Stray distribution: a random member of a rule drawn from the learner's rule prior.
    mix_logit = pm.Normal("mix_logit", mu=0.0, sigma=1.5)
    mix = pm.Deterministic("mix", pm.math.sigmoid(mix_logit))
    q = (1 - mix) * 0.01 + mix * STRAY_RULE_GEN
    inv_q = example_mask / q[example_idx]  # (sets, 4)
    inv_q_rest = inv_q * (1.0 - anchor_onehot)
    inv_q_anchor = (inv_q * anchor_onehot).sum(axis=1)  # (sets,)

    # Trust in the last-shown (most recent, anchoring) example.
    anchor_logit = pm.Normal("anchor_logit", mu=2.2, sigma=1.5)
    a1 = pm.Deterministic("a_last", pm.math.sigmoid(anchor_logit))

    # Reliability-marginal weights over the m earlier examples.
    c_table = pt.where(_VALID, pt.exp(_betaln(u + _J, v + _N_MINUS_J) - _betaln(u, v)), 0.0)
    c_set = c_table[set_n_idx]  # (sets, j)

    # e_j over the earlier examples in every subset, folded with c_j.
    e_sub = _elementary(SUBSET_IN[None, :, :] * inv_q_rest[:, None, :])  # (sets, subsets, j)
    coef = e_sub * c_set[:, None, :]
    anchor_in_sub = pt.dot(anchor_onehot, SUBSET_IN.T)  # (sets, subsets)
    rest_size = pt.dot(coef, _SIZE_POW_J.T)  # (sets, subsets, size)
    sub_size_lik = rest_size * (
        (1 - a1) + a1 * (anchor_in_sub * inv_q_anchor[:, None])[:, :, None] / _SIZES[None, None, :]
    )

    rule_lik = sub_size_lik[pt.arange(rule_sub.shape[0])[:, None], rule_sub, RULE_SIZE_IDX[None, :]]
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + pt.log(rule_lik)

    # Intervals: run_pow[.., j] = sum size^-j; the anchor adds one more size^-1.
    run_coef = coef[:, RUN_SUBSET, :]  # (sets, runs, j)
    run_anchor = (anchor_in_sub[:, RUN_SUBSET] * inv_q_anchor[:, None])[:, :, None]  # (sets, runs, 1)

    def _int_ev(pow_tab, rc, rf):
        return ((1 - a1) * (pow_tab[:, :, : MAX_N + 1] * rc) + a1 * rf * (pow_tab[:, :, 1:] * rc)).sum(axis=(1, 2))

    int_ev = _int_ev(run_pow, run_coef, run_anchor)
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + pt.log(int_ev)

    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    interval_post = pt.exp(log_intervals - log_norm)

    cell_int_share = _int_ev(cell_run_pow, run_coef[cell_set], run_anchor[cell_set]) / int_ev[cell_set]
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
    anchor_onehot = np.zeros((S, MAX_N))
    set_n_idx = np.zeros(S, dtype="int64")
    run_count = np.zeros((S, N_RUNS + 1, 100))
    set_runs = []
    for i, s in enumerate(unique):
        ex = np.sort(np.array(s))
        n = len(ex)
        set_n_idx[i] = n - 1
        example_idx[i, :n] = ex - 1
        example_mask[i, :n] = 1.0
        anchor_onehot[i, int(np.flatnonzero(ex == s[-1])[0])] = 1.0
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
        "anchor_onehot": anchor_onehot,
        "rule_sub": (
            (RULE_MEMBERSHIP[:, example_idx].transpose(1, 0, 2) * example_mask[:, None, :])
            @ (2 ** np.arange(MAX_N))
        ).astype("int64"),
        "set_n_idx": set_n_idx,
        "run_pow": run_count @ _SIZE_POW,
        "cell_run_pow": cell_run_count @ _SIZE_POW,
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
