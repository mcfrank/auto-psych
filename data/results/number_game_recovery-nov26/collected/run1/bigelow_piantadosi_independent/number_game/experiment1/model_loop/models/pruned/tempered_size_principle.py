"""Bayesian concept learning with a tempered size principle.

People infer the hidden concept by Bayesian reasoning over mathematical rules
and the intervals of 1..100, but each example favours a concept containing it
only by |h|^(-beta) (beta free, beta = 1 is the full size principle), so broad
rules like "even" stay strong against narrow rules and tight intervals. An
example outside a concept has fixed small probability EPS (examples are
occasionally noise). "Yes" is the posterior probability that the concept
contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPS = 1e-3
NUMBERS = np.arange(1, 101)


def _rules():
    rules = [
        [n for n in NUMBERS if n % 2 == 1],
        list(range(2, 101, 2)),
        [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        [n * n for n in range(1, 11)],
        [n**3 for n in range(1, 5)],
        [2, 4, 8, 16, 64],
        [2, 4, 8, 16, 32, 37, 64],
    ]
    rules += [list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rules += [[m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)]
    rules += [[n for n in NUMBERS if n % 10 == d] for d in range(10)]
    rules += [[n for n in NUMBERS if str(d) in str(n)] for d in range(10)]
    return np.array([np.isin(NUMBERS, r) for r in rules], dtype="float64")


RULE_MEM = _rules()
N_RULES = RULE_MEM.shape[0]
RULE_LOGSIZE = np.log(RULE_MEM.sum(axis=1))
LOGSIZE = np.log(np.arange(1, 101, dtype="float64"))  # interval sizes 1..100
KS = np.arange(5, dtype="float64")  # number of examples outside the concept


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_out = np.zeros((S, N_RULES))
    n_ex = np.zeros(S)
    # interval groups by (size, k_out): count of intervals, and per target count containing it
    counts = np.zeros((S, 100, 5))
    tgt_counts = np.zeros((S, 100, 100, 5))  # set, target, size, k
    lo = np.arange(1, 101)[:, None]
    hi = np.arange(1, 101)[None, :]
    valid = hi >= lo
    size_idx = (hi - lo).clip(0)
    for i, s in enumerate(unique):
        ex = np.array(s)
        n_ex[i] = len(ex)
        rule_out[i] = len(ex) - RULE_MEM[:, ex - 1].sum(axis=1)
        inside = sum(((lo <= x) & (x <= hi)).astype(int) for x in ex)
        k = len(ex) - inside
        for a, b in zip(*np.nonzero(valid)):
            sz, kk = size_idx[a, b], k[a, b]
            counts[i, sz, kk] += 1
            tgt_counts[i, a : b + 1, sz, kk] += 1
    frac = np.where(counts[:, None] > 0, tgt_counts / np.maximum(counts[:, None], 1), 0.0)
    set_idx = np.array([index[s] for s in sets])
    tgt = np.array([int(r["target"]) - 1 for r in rows])
    return {
        "n_ex": n_ex,
        "rule_out": rule_out,
        "rule_member_t": RULE_MEM[:, tgt].T,
        "log_counts": np.where(counts > 0, np.log(np.maximum(counts, 1)), -1e4).reshape(S, -1),
        "interval_frac_t": frac[set_idx, tgt].reshape(len(rows), -1),
        "set_idx": set_idx.astype("int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    n_ex = pm.Data("n_ex", np.zeros(1))
    rule_out = pm.Data("rule_out", np.zeros((1, N_RULES)))
    rule_member_t = pm.Data("rule_member_t", np.zeros((1, N_RULES)))
    log_counts = pm.Data("log_counts", np.zeros((1, 500)))
    interval_frac_t = pm.Data("interval_frac_t", np.zeros((1, 500)))
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lam = pm.Beta("lambda_rules", 2, 2)
    beta = pm.Gamma("beta", alpha=2.0, beta=2.0)

    # Rules: log prior + in-concept examples at |h|^-beta + out-of-concept at EPS.
    n_in_rule = n_ex[:, None] - rule_out
    log_rule = (pt.log(lam) - np.log(N_RULES) - beta * n_in_rule * RULE_LOGSIZE[None, :]
                + rule_out * np.log(EPS))
    # Intervals grouped by (size, k_out); each interval has prior (1 - lam) / 5050.
    n_in_int = n_ex[:, None, None] - KS[None, None, :]
    log_int = (pt.log1p(-lam) - np.log(5050.0) - beta * n_in_int * LOGSIZE[None, :, None]
               + KS[None, None, :] * np.log(EPS)).reshape((n_ex.shape[0], 500)) + log_counts
    log_joint = pt.concatenate([log_rule, log_int], axis=1)
    post = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))[set_idx]
    p = (post[:, :N_RULES] * rule_member_t).sum(axis=1) + (post[:, N_RULES:] * interval_frac_t).sum(axis=1)
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
