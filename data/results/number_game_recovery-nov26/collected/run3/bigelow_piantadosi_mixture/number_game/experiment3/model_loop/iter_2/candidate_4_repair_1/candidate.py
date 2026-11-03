"""Variable rule-versus-range expectation with a two-source account of strays.

Refinement of variable_rule_range_expectation (Bayesian concept learning over 45
equally plausible rules, rules held within a range (share rho of the rule prior)
and every range of 1-100; size principle; uncertain reliability
a ~ Beta(alpha*kappa, (1-alpha)*kappa) integrated out per set; the prior share of
rule-based programs lambda varies from judgement to judgement,
logit(lambda) ~ Normal(lambda_mu, lambda_sd), integrated with 9-node
Gauss-Hermite quadrature). The one change, taken from
two_source_range_limited_rules, is what the off-pattern examples are: they are not only
strays written off, but the output of one secondary
source per set, either a second familiar rule c2 (prior mix/45 each) whose
other members people then also endorse, or glitches (prior 1-mix) that are
uniform or, with probability g, familiar numbers (rule-count familiarity) and
license no generalisation.

For examples X (n), main concept c1, source src and genuine subset G (|G| = j):
  p(X, G | c1, src) = B(u+j, v+n-j)/B(u,v) * prod_G 1[x in c1]/|c1| * prod_{X\\G} q_src(x).
Per-class (full rule, range, range-limited rule) evidence and yes-mass tables
are built in prepare_observed; lambda enters only through the class weights, so
each quadrature node is a (3 x K) product.
"""

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


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
RULE_MEM = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")  # (rules, 100)
RULE_SIZE = RULE_MEM.sum(axis=1)
INT_LO = np.array([m for m in NUMBERS for n in range(m, 101)])
INT_HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INT_SIZE = (INT_HI - INT_LO + 1).astype("float64")
INT_MEM = ((NUMBERS[None, :] >= INT_LO[:, None]) & (NUMBERS[None, :] <= INT_HI[:, None])).astype("float64")


def _restricted_blocks():
    """Every proper, non-empty block of a rule's sorted members that a range of 1-100 can cut out,
    with the number of ranges that cut it out (rule then range picked uniformly)."""
    rule, lo, hi, size, mult = [], [], [], [], []
    for r, c in enumerate(RULES):
        m = np.sort(np.array(c))
        k = len(m)
        gaps = np.diff(np.concatenate([[0], m, [101]]))
        for a in range(k):
            for b in range(a, k):
                if a == 0 and b == k - 1:
                    continue  # the whole rule is the plain rule
                rule.append(r); lo.append(m[a]); hi.append(m[b]); size.append(b - a + 1)
                mult.append(gaps[a] * gaps[b + 1])
    return np.array(rule), np.array(lo), np.array(hi), np.array(size, dtype="float64"), np.array(mult, dtype="float64")


BLK_RULE, BLK_LO, BLK_HI, BLK_SIZE, BLK_MULT = _restricted_blocks()
BLK_MEM = (RULE_MEM[BLK_RULE] * ((NUMBERS[None, :] >= BLK_LO[:, None]) & (NUMBERS[None, :] <= BLK_HI[:, None]))).astype("float64")
N_CLASS = 3  # main concept: full rule, interval, range-limited rule

_N = np.arange(1, MAX_N + 1, dtype="float64")[:, None]
_J = np.arange(MAX_N + 1, dtype="float64")[None, :]
_VALID = _J <= _N
_N_MINUS_J = np.maximum(_N - _J, 0.0)
_GH_Z, _GH_W = np.polynomial.hermite_e.hermegauss(9)
_GH_W = _GH_W / _GH_W.sum()  # standard-normal quadrature
N_SUB = 2**MAX_N
SUBSETS = np.array([[(m >> i) & 1 for i in range(MAX_N)] for m in range(N_SUB)], dtype=bool)  # (16, 4)
SUBSET_SIZE = SUBSETS.sum(axis=1)
RULE_COUNT_FAMILIARITY = RULE_MEM.sum(axis=0) / RULE_MEM.sum()  # (100,)


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


with pm.Model() as model:
    ev_rule = pm.Data("ev_rule", np.zeros((1, N_CLASS, MAX_N + 1)))  # (sets, class, j): second-rule source
    yes_rule = pm.Data("yes_rule", np.zeros((1, N_CLASS, MAX_N + 1)))  # (cells, class, j)
    ev_glitch = pm.Data("ev_glitch", np.zeros((1, N_CLASS, N_SUB)))  # (sets, class, subset G): glitch source
    yes_glitch = pm.Data("yes_glitch", np.zeros((1, N_CLASS, N_SUB)))  # (cells, class, subset G)
    example_idx = pm.Data("example_idx", np.zeros((1, MAX_N), dtype="int64"))  # x - 1, shown order
    example_mask = pm.Data("example_mask", np.zeros((1, MAX_N)))
    set_n_idx = pm.Data("set_n_idx", np.zeros(1, dtype="int64"))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Judgement-to-judgement variable belief that the program is rule-based.
    lambda_mu = pm.Normal("lambda_mu", mu=0.0, sigma=1.5)
    lambda_sd = pm.HalfNormal("lambda_sd", sigma=1.0)
    lam_k = pm.math.sigmoid(lambda_mu + lambda_sd * _GH_Z)  # (K,)
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa
    mix_logit = pm.Normal("mix_logit", mu=0.0, sigma=1.5)
    mix = pm.Deterministic("mix", pm.math.sigmoid(mix_logit))  # prior that the second source is a rule
    # Glitches: uniform, or with probability g a familiar number (rule-count familiarity).
    g_logit = pm.Normal("g_logit", mu=0.0, sigma=1.5)
    g = pm.Deterministic("g", pm.math.sigmoid(g_logit))
    q_rel = (1 - g) + g * 100.0 * RULE_COUNT_FAMILIARITY  # (100,): 100 q(x)
    q_ex = pt.where(example_mask > 0, q_rel[example_idx], 1.0)  # (sets, 4)
    stray_fac = pt.prod(pt.where(SUBSETS[None, :, :], 1.0, q_ex[:, None, :]), axis=2)  # (sets, 16)

    c_table = pt.where(_VALID, pt.exp(_betaln(u + _J, v + _N_MINUS_J) - _betaln(u, v)), 0.0)
    c_set = c_table[set_n_idx]  # (sets, j)
    # Share rho of the rule prior goes to rules held within a range of 1-100.
    rho_logit = pm.Normal("rho_logit", mu=-2.0, sigma=1.0)
    rho = pm.Deterministic("rho", pm.math.sigmoid(rho_logit))
    # Class-level evidence and yes-mass (without the lambda-dependent class weights).
    w_rule = mix * c_set[:, None, :]  # (sets, 1, j)
    w_glitch = (1 - mix) * (c_set[:, SUBSET_SIZE] * stray_fac)[:, None, :]  # (sets, 1, 16)
    ev_cls = (w_rule * ev_rule).sum(axis=2) + (w_glitch * ev_glitch).sum(axis=2)  # (sets, class)
    yes_cls = (w_rule[cell_set] * yes_rule).sum(axis=2) + (w_glitch[cell_set] * yes_glitch).sum(axis=2)  # (cells, class)
    w_class = pt.stack([lam_k * (1 - rho), 1 - lam_k, lam_k * rho])  # (class, K)
    ev_k = pt.dot(ev_cls, w_class)  # (sets, K)
    yes_k = pt.dot(yes_cls, w_class)  # (cells, K)
    p_cell = pt.dot(yes_k / ev_k[cell_set], _GH_W)
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell, 1e-9, 1 - 1e-9)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _set_tables(ex):
    """Second-rule tables (class, j) and glitch tables (class, subset) for one set, with yes-mass per target."""
    n = len(ex)
    n_sub = 2**n
    sub_in = np.array([[(m >> i) & 1 for i in range(n)] for m in range(n_sub)], dtype=bool)  # (G, n)
    j_of = sub_in.sum(axis=1)
    full = n_sub - 1
    onehot_j = np.zeros((n_sub, MAX_N + 1))
    onehot_j[np.arange(n_sub), j_of] = 1.0

    def genuine_w(mem, size, prior):  # G must lie in c1: weight prior * size^-j
        ex_in = mem[:, ex - 1] > 0  # (concepts, n)
        ok = ~(sub_in[None, :, :] & ~ex_in[:, None, :]).any(axis=2)  # (concepts, G)
        return ok * size[:, None] ** -j_of[None, :] * prior

    w1 = [
        genuine_w(RULE_MEM, RULE_SIZE, 1.0 / N_RULES),
        genuine_w(INT_MEM, INT_SIZE, 1.0 / 100**2),
        genuine_w(BLK_MEM, BLK_SIZE, BLK_MULT[:, None] / (N_RULES * 5050.0)),
    ]
    w1_tot = [w.sum(axis=0) for w in w1]  # (G,)
    w1_out = [RULE_MEM_NOT.T @ w1[0], INT_MEM_NOT.T @ w1[1], BLK_MEM_NOT.T @ w1[2]]  # (100, G): c1 lacks target
    # Second-rule source: strays X\\G must lie in c2.
    ex_in2 = RULE_MEM[:, ex - 1] > 0
    ok2 = ~((~sub_in)[None, :, :] & ~ex_in2[:, None, :]).any(axis=2)
    w2 = ok2 * RULE_SIZE[:, None] ** -(n - j_of)[None, :] / N_RULES  # (rules, G)
    w2_tot = w2.sum(axis=0)
    w2_out = RULE_MEM_NOT.T @ w2  # (100, G): c2 lacks target
    w2_out[:, full] = w2_tot[full]  # no strays: the second rule produced nothing, no generalisation
    w_unif = 0.01 ** (n - j_of)

    ev_r = np.zeros((N_CLASS, MAX_N + 1))
    yes_r = np.zeros((100, N_CLASS, MAX_N + 1))
    ev_g = np.zeros((N_CLASS, N_SUB))  # subsets beyond 2^n stay zero
    yes_g = np.zeros((100, N_CLASS, N_SUB))
    for k in range(N_CLASS):
        ev_r[k] = (w1_tot[k] * w2_tot) @ onehot_j
        yes_r[:, k] = (w1_tot[k] * w2_tot - w1_out[k] * w2_out) @ onehot_j
        ev_g[k, :n_sub] = w1_tot[k] * w_unif
        yes_g[:, k, :n_sub] = (w1_tot[k] - w1_out[k]) * w_unif
    scale = max(ev_r.max(), ev_g.max())
    return ev_r / scale, yes_r / scale, ev_g / scale, yes_g / scale


RULE_MEM_NOT = 1.0 - RULE_MEM
INT_MEM_NOT = 1.0 - INT_MEM
BLK_MEM_NOT = 1.0 - BLK_MEM


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    tables = [_set_tables(np.array(s)) for s in unique]
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: k for k, c in enumerate(cells)}
    S = len(unique)
    example_idx = np.zeros((S, MAX_N), dtype="int64")
    example_mask = np.zeros((S, MAX_N))
    for i, s in enumerate(unique):
        example_idx[i, : len(s)] = np.array(s) - 1
        example_mask[i, : len(s)] = 1.0
    return {
        "ev_rule": np.stack([t[0] for t in tables]),
        "yes_rule": np.stack([tables[i][1][t - 1] for i, t in cells]),
        "ev_glitch": np.stack([t[2] for t in tables]),
        "yes_glitch": np.stack([tables[i][3][t - 1] for i, t in cells]),
        "example_idx": example_idx,
        "example_mask": example_mask,
        "set_n_idx": np.array([len(s) - 1 for s in unique], dtype="int64"),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
