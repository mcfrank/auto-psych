"""Gist memory for examples (fuzzy-trace concept learning).

People infer the hidden program by Bayesian concept learning over equally plausible
mathematical rules, rules held within a magnitude range and magnitude ranges of
1-100 (size principle, an uncertain belief about how often an example is a stray,
strays sometimes familiar numbers). But they do not keep every example verbatim:
each example is retained exactly with an inferred probability, and otherwise only
its gist, its approximate place on the number line (within about 10 either side). A gist
example carries magnitude information but not its arithmetic identity, so it
supports a range covering that stretch as well as ever, but supports a rule only
as much as the rule happens to have members in that stretch. When some examples
are held only as gist, clustered examples point to a narrow range, and people fill
in the rule's non-members inside the examples' span.

Computation: for each memory pattern P (which examples are verbatim) and subset T
of examples taken as genuine, the parameter-free coefficient
sum_C prod_{i in T} a_i^P(C) / |C|^|T| is tabulated in prepare_observed for each
concept family (rules one by one, all intervals, all restricted-rule blocks), with
a_i = membership (verbatim) or overlap of C with the gist window (gist). The model
weights these by the reliability marginal c_|T| and by the stray probabilities, and
averages the per-pattern posterior membership over patterns.
"""

import numpy as np
import pymc as pm
import pytensor.sparse as ps
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4
GIST_W = 10


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
RULES = list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
RULE_CUM = np.concatenate([np.zeros((N_RULES, 1)), np.cumsum(RULE_MEMBERSHIP, axis=1)], axis=1)  # count in [1, n]
INT_LO = np.array([m for m in NUMBERS for n in range(m, 101)])
INT_HI = np.array([n for m in NUMBERS for n in range(m, 101)])
N_INT = len(INT_LO)
FAMILIARITY = RULE_MEMBERSHIP.sum(axis=0) / RULE_MEMBERSHIP.sum()

N_SUB = 2**MAX_N
SUBSET_IN = np.array([[(m >> i) & 1 for i in range(MAX_N)] for m in range(N_SUB)], dtype="float64")  # (16, 4)
SUBSET_POP = SUBSET_IN.sum(axis=1).astype("int64")
# Memory patterns use the same bit layout: bit i = 1 means example i is held verbatim.
PATTERN_IN = SUBSET_IN

_N = np.arange(1, MAX_N + 1, dtype="float64")[:, None]
_J = np.arange(MAX_N + 1, dtype="float64")[None, :]
_VALID = _J <= _N
_N_MINUS_J = np.maximum(_N - _J, 0.0)


def _restricted_blocks():
    rule, lo, hi, size, mult = [], [], [], [], []
    for r, c in enumerate(RULES):
        m = np.sort(np.array(c))
        k = len(m)
        gaps = np.diff(np.concatenate([[0], m, [101]]))
        for a in range(k):
            for b in range(a, k):
                if a == 0 and b == k - 1:
                    continue
                rule.append(r)
                lo.append(m[a])
                hi.append(m[b])
                size.append(b - a + 1)
                mult.append(gaps[a] * gaps[b + 1])
    return np.array(rule), np.array(lo), np.array(hi), np.array(size, dtype="float64"), np.array(mult, dtype="float64")


BLK_RULE, BLK_LO, BLK_HI, BLK_SIZE, BLK_MULT = _restricted_blocks()


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


with pm.Model() as model:
    example_idx = pm.Data("example_idx", np.zeros((1, MAX_N), dtype="int64"))  # x - 1 (0 if absent)
    example_mask = pm.Data("example_mask", np.zeros((1, MAX_N)))
    gist_unif = pm.Data("gist_unif", np.zeros((1, MAX_N)))  # uniform stray mass in the gist window
    gist_fam = pm.Data("gist_fam", np.zeros((1, MAX_N)))  # familiar stray mass in the gist window
    set_n_idx = pm.Data("set_n_idx", np.zeros(1, dtype="int64"))
    rule_coef = pm.Data("rule_coef", np.zeros((1, N_SUB, N_RULES, N_SUB)))  # (sets, P, rules, T)
    int_coef = pm.Data("int_coef", np.zeros((1, N_SUB, N_SUB)))  # (sets, P, T)
    rr_coef = pm.Data("rr_coef", np.zeros((1, N_SUB, N_SUB)))
    # Distinct (set, target) cells: (cells, P, 2T) interval then restricted-rule coefficients.
    cell_range_coef = pm.Data("cell_range_coef", np.zeros((1, N_SUB, 2 * N_SUB)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_flat = pm.Data("cell_flat", np.zeros(1, dtype="int64"))  # set * 100 + target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa
    mix_logit = pm.Normal("mix_logit", mu=0.0, sigma=1.5)
    mix = pm.Deterministic("mix", pm.math.sigmoid(mix_logit))
    rho_logit = pm.Normal("rho_logit", mu=-2.0, sigma=1.0)
    rho = pm.Deterministic("rho", pm.math.sigmoid(rho_logit))
    # Probability an example is retained verbatim rather than as magnitude gist.
    verbatim_logit = pm.Normal("verbatim_logit", mu=1.5, sigma=1.5)
    g = pm.Deterministic("verbatim", pm.math.sigmoid(verbatim_logit))

    # Stray probability of each example's memory trace: exact value, or the gist window.
    q = (1 - mix) * 0.01 + mix * FAMILIARITY
    q_verb = q[example_idx]  # (sets, 4)
    q_gist = (1 - mix) * gist_unif + mix * gist_fam
    bits = PATTERN_IN[None, :, :]  # (1, P, 4)
    log_q = bits * pt.log(q_verb)[:, None, :] + (1 - bits) * pt.log(pt.maximum(q_gist, 1e-12))[:, None, :]
    log_inv_q = -log_q * example_mask[:, None, :]  # (sets, P, 4); absent examples contribute 1
    # Product over each subset T of examples: (sets, P, T)
    iq = pt.exp(pt.dot(log_inv_q, SUBSET_IN.T))

    c_table = pt.where(_VALID, pt.exp(_betaln(u + _J, v + _N_MINUS_J) - _betaln(u, v)), 0.0)
    c_T = c_table[set_n_idx][:, SUBSET_POP]  # (sets, T)
    wt = iq * c_T[:, None, :]  # (sets, P, T)

    # Memory-pattern probabilities (absent examples count as verbatim with probability 1).
    m = example_mask[:, None, :]
    pat_p = pt.prod(m * pt.where(bits > 0, g, 1 - g) + (1 - m) * bits, axis=-1)  # (sets, P)

    # Prior-weighted evidence for each concept family under each pattern.
    n_sets = wt.shape[0]
    rule_ev = pt.batched_dot(
        rule_coef.reshape((n_sets * N_SUB, N_RULES, N_SUB)), wt.reshape((n_sets * N_SUB, N_SUB))
    ).reshape((n_sets, N_SUB, N_RULES))
    ev_rule = lambda_0 * (1 - rho) / N_RULES * rule_ev  # (sets, P, rules)
    ev_int = (1 - lambda_0) / N_INT * (int_coef * wt).sum(axis=-1)  # (sets, P)
    ev_rr = lambda_0 * rho / (N_RULES * N_INT) * (rr_coef * wt).sum(axis=-1)
    z = ev_rule.sum(axis=2) + ev_int + ev_rr  # (sets, P)

    # Rule evidence for every possible target 1-100, then the asked-about cells.
    rule_t = pt.dot(ev_rule.reshape((n_sets * N_SUB, N_RULES)), RULE_MEMBERSHIP)  # (sets*P, 100)
    rule_t = (rule_t.reshape((n_sets, N_SUB, 100)) / z[:, :, None]).dimshuffle(0, 2, 1).reshape((n_sets * 100, N_SUB))
    wt2 = pt.concatenate([(1 - lambda_0) / N_INT * wt, lambda_0 * rho / (N_RULES * N_INT) * wt], axis=-1) / z[:, :, None]
    # Sparse cell-by-set one-hot: copies each set's weights to its cells (cheap gradient).
    n_cells = cell_set.shape[0]
    onehot = ps.CSR(pt.ones(n_cells), cell_set, pt.arange(n_cells + 1), pt.stack([n_cells, n_sets]))
    per_cell = ps.structured_dot(onehot, pt.concatenate([wt2.reshape((n_sets, -1)), pat_p], axis=1))
    wt2_c = per_cell[:, : N_SUB * 2 * N_SUB].reshape((n_cells, N_SUB, 2 * N_SUB))
    num = rule_t[cell_flat] + (cell_range_coef * wt2_c).sum(axis=-1)  # (cells, P)
    in_concept = (per_cell[:, N_SUB * 2 * N_SUB :] * num).sum(axis=-1)
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-9, 1 - 1e-9)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _window(x):
    return max(1, x - GIST_W), min(100, x + GIST_W)


def _coef(verb, gist, size):
    """verb, gist: (N, 4) traces' fit to N concepts; size: (N,). Returns (N, P, T)."""
    a = np.where(PATTERN_IN[None, :, :] > 0, verb[:, None, :], gist[:, None, :])  # (N, P, 4)
    prod = np.prod(a[:, :, None, :] ** SUBSET_IN[None, None, :, :], axis=-1)  # (N, P, T)
    return prod * (size[:, None, None] ** -SUBSET_POP[None, None, :].astype(float))


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    example_idx = np.zeros((S, MAX_N), dtype="int64")
    example_mask = np.zeros((S, MAX_N))
    gist_unif = np.zeros((S, MAX_N))
    gist_fam = np.zeros((S, MAX_N))
    set_n_idx = np.zeros(S, dtype="int64")
    rule_coef = np.zeros((S, N_SUB, N_RULES, N_SUB))
    int_coef = np.zeros((S, N_SUB, N_SUB))
    rr_coef = np.zeros((S, N_SUB, N_SUB))
    cells_of = {}
    for i, t in cells:
        cells_of.setdefault(i, []).append(t)
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_range_coef = np.zeros((len(cells), N_SUB, 2 * N_SUB))
    int_size = (INT_HI - INT_LO + 1).astype(float)
    for i, s in enumerate(unique):
        ex = np.array(s)
        n = len(ex)
        set_n_idx[i] = n - 1
        example_idx[i, :n] = ex - 1
        example_mask[i, :n] = 1.0
        lo_w = np.array([_window(x)[0] for x in ex])
        hi_w = np.array([_window(x)[1] for x in ex])
        gist_unif[i, :n] = (hi_w - lo_w + 1) / 100.0
        gist_fam[i, :n] = [FAMILIARITY[a - 1 : b].sum() for a, b in zip(lo_w, hi_w)]
        # Rules
        verb = np.zeros((N_RULES, MAX_N))
        gist = np.zeros((N_RULES, MAX_N))
        verb[:, :n] = RULE_MEMBERSHIP[:, ex - 1]
        gist[:, :n] = RULE_CUM[:, hi_w] - RULE_CUM[:, lo_w - 1]
        rule_coef[i] = _coef(verb, gist, RULE_SIZE).transpose(1, 0, 2)
        # Intervals
        verb = np.zeros((N_INT, MAX_N))
        gist = np.zeros((N_INT, MAX_N))
        verb[:, :n] = (ex[None, :] >= INT_LO[:, None]) & (ex[None, :] <= INT_HI[:, None])
        gist[:, :n] = np.maximum(
            0, np.minimum(INT_HI[:, None], hi_w[None, :]) - np.maximum(INT_LO[:, None], lo_w[None, :]) + 1
        )
        ic = _coef(verb, gist, int_size).reshape(N_INT, -1)
        int_coef[i] = ic.sum(axis=0).reshape(N_SUB, N_SUB)
        # Restricted-rule blocks
        verb = np.zeros((len(BLK_RULE), MAX_N))
        gist = np.zeros((len(BLK_RULE), MAX_N))
        verb[:, :n] = (RULE_MEMBERSHIP[BLK_RULE][:, ex - 1] > 0) & (ex[None, :] >= BLK_LO[:, None]) & (
            ex[None, :] <= BLK_HI[:, None]
        )
        glo = np.maximum(BLK_LO[:, None], lo_w[None, :])
        ghi = np.minimum(BLK_HI[:, None], hi_w[None, :])
        cnt = RULE_CUM[BLK_RULE[:, None], np.clip(ghi, 0, 100)] - RULE_CUM[BLK_RULE[:, None], np.clip(glo - 1, 0, 100)]
        gist[:, :n] = np.where(ghi >= glo, cnt, 0.0)
        bc = _coef(verb, gist, BLK_SIZE).reshape(len(BLK_RULE), -1) * BLK_MULT[:, None]
        rr_coef[i] = bc.sum(axis=0).reshape(N_SUB, N_SUB)
        # Cells of this set: concepts containing the target.
        ts = np.array(cells_of.get(i, []), dtype="int64")
        js = [cell_index[(i, t)] for t in ts]
        if len(ts):
            in_int = ((ts[:, None] >= INT_LO[None, :]) & (ts[:, None] <= INT_HI[None, :])).astype(float)
            cell_range_coef[js, :, :N_SUB] = (in_int @ ic).reshape(len(ts), N_SUB, N_SUB)
            in_blk = (
                (RULE_MEMBERSHIP[BLK_RULE][:, ts - 1].T > 0)
                & (ts[:, None] >= BLK_LO[None, :])
                & (ts[:, None] <= BLK_HI[None, :])
            ).astype(float)
            cell_range_coef[js, :, N_SUB:] = (in_blk @ bc).reshape(len(ts), N_SUB, N_SUB)
    return {
        "example_idx": example_idx,
        "example_mask": example_mask,
        "gist_unif": gist_unif,
        "gist_fam": gist_fam,
        "set_n_idx": set_n_idx,
        "rule_coef": rule_coef,
        "int_coef": int_coef,
        "rr_coef": rr_coef,
        "cell_range_coef": cell_range_coef,
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_flat": np.array([i * 100 + t - 1 for i, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
