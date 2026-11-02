"""Pedagogical sampling: people do Bayesian concept learning over the 45
Bigelow-Piantadosi rules (each with its own inferred prior weight) and every
interval of 1-100, but they assume the examples were chosen by a helpful
teacher rather than drawn at random from the concept. The teacher prefers
examples that point a naive learner to the intended concept and away from
other concepts, so an example that belongs to few other concepts (64) is
strong evidence for the concepts containing it and one that belongs to many
(16, 50) is weak evidence. The one change from random (size-principle)
sampling is the inferred pedagogical strength tau.

Formally, a naive learner (rules as a group and intervals as a group equally
likely, equal weight within each) believes after one example x that
P0(h | x) = prior0(h) [x in h] / |h| / Z0(x); the teacher picks x in h with
probability proportional to P0(h | x)^tau, i.e. w(x) / W(h) with
w(x) = Z0(x)^-tau and W(h) the sum of w over h (tau = 0 is the size
principle). As in the base model, each example is a stray draw from 1-100 with
probability 0.1. "Yes" is the posterior probability that the concept contains
the target.

The rule part is computed exactly in the graph. The interval part depends only
on tau, so prepare_observed tabulates, on a grid of tau, each set's log summed
interval evidence and, per target, the share of it from intervals containing
the target; the graph interpolates with normalised Gaussian kernel weights.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4
ALPHA = 0.9
NOISE = (1 - ALPHA) / 100.0
TAU_MAX = 4.0
TAU_GRID = np.linspace(0.0, TAU_MAX, 81)
TAU_STEP = TAU_GRID[1] - TAU_GRID[0]


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
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")  # (rules, 100)
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
INT_LO = np.array([m for m in NUMBERS for n in range(m, 101)])
INT_HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INT_SIZE = (INT_HI - INT_LO + 1).astype("float64")
INT_MEMBERSHIP = ((NUMBERS[None, :] >= INT_LO[:, None]) & (NUMBERS[None, :] <= INT_HI[:, None])).astype("float64")
# Naive learner's normaliser Z0(x): rules and intervals each half the prior mass.
Z0 = 0.5 * (RULE_MEMBERSHIP / RULE_SIZE[:, None]).mean(axis=0) + 0.5 * (INT_MEMBERSHIP / INT_SIZE[:, None]).mean(axis=0)
LOG_Z0 = np.log(Z0) - np.log(Z0).mean()

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0


def teacher_weights(tau):
    return np.exp(-tau * LOG_Z0)


with pm.Model() as model:
    ex_idx = pm.Data("ex_idx", np.zeros((1, MAX_N), dtype="int64"))  # (sets, 4): example - 1
    ex_mask = pm.Data("ex_mask", np.ones((1, MAX_N)))  # (sets, 4): 1 = real example
    rule_in = pm.Data("rule_in", np.zeros((1, MAX_N, N_RULES)))  # (sets, 4, rules)
    int_log_ev = pm.Data("int_log_ev", np.zeros((1, len(TAU_GRID))))  # (sets, grid)
    cell_int_share = pm.Data("cell_int_share", np.zeros((1, len(TAU_GRID))))  # (cells, grid)
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
    # Pedagogical strength, kept inside the tabulated range (0, TAU_MAX).
    tau_raw = pm.Normal("tau_raw", mu=-1.0, sigma=1.0)
    tau = pm.Deterministic("tau", TAU_MAX * pm.math.sigmoid(tau_raw))

    # Rules: exact teacher likelihood w(x) / W(rule) mixed with stray draws.
    w = pt.exp(-tau * LOG_Z0)  # (100,)
    W_rules = pt.dot(RULE_MEMBERSHIP, w)  # (rules,)
    p_rule = rule_in * ALPHA * w[ex_idx][:, :, None] / W_rules[None, None, :] + NOISE
    rule_ll = (ex_mask[:, :, None] * pt.log(p_rule)).sum(axis=1)  # (sets, rules)
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_ll

    # Intervals: smooth interpolation of the tabulated tau tables.
    kern = pt.exp(-0.5 * ((tau - TAU_GRID) / TAU_STEP) ** 2)
    kern = kern / kern.sum()
    log_ints = pt.log1p(-lambda_0) - 2 * np.log(100) + pt.dot(int_log_ev, kern)  # (sets,)

    log_norm = pt.logsumexp(pt.concatenate([log_rules, log_ints[:, None]], axis=1), axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    int_post = pt.exp(log_ints - log_norm)
    share = pt.dot(cell_int_share, kern)  # (cells,)
    in_concept = (rule_post[cell_set] * cell_rule_member).sum(axis=1) + int_post[cell_set] * share
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-9, 1 - 1e-9)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S, G = len(unique), len(TAU_GRID)
    ex_idx = np.zeros((S, MAX_N), dtype="int64")
    ex_mask = np.zeros((S, MAX_N))
    rule_in = np.zeros((S, MAX_N, N_RULES))
    int_in = np.zeros((S, MAX_N, len(INT_LO)), dtype=bool)
    for i, s in enumerate(unique):
        for j, x in enumerate(s):
            ex_idx[i, j] = x - 1
            ex_mask[i, j] = 1.0
            rule_in[i, j] = RULE_MEMBERSHIP[:, x - 1]
            int_in[i, j] = INT_MEMBERSHIP[:, x - 1] > 0
    log_ev = np.zeros((S, G))
    share_all = np.zeros((S, 100, G))
    for g, tau in enumerate(TAU_GRID):
        w = teacher_weights(tau)
        cw = np.concatenate([[0.0], np.cumsum(w)])
        W_int = cw[INT_HI] - cw[INT_LO - 1]
        ll = np.zeros((S, len(INT_LO)))
        for j in range(MAX_N):
            p = np.where(int_in[:, j], ALPHA * w[ex_idx[:, j]][:, None] / W_int[None, :], 0.0) + NOISE
            ll += ex_mask[:, j][:, None] * np.log(p)
        top = ll.max(axis=1, keepdims=True)
        e = np.exp(ll - top)
        tot = e.sum(axis=1)
        log_ev[:, g] = top[:, 0] + np.log(tot)
        share_all[:, :, g] = (e @ INT_MEMBERSHIP) / tot[:, None]
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    return {
        "ex_idx": ex_idx,
        "ex_mask": ex_mask,
        "rule_in": rule_in,
        "int_log_ev": log_ev,
        "cell_int_share": np.stack([share_all[i, t - 1] for i, t in cells]),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
