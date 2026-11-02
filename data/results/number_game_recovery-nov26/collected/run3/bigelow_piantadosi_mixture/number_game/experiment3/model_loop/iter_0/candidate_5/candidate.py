"""Two-source program whose second source is drawn from the same program prior.

Refinement of two_source_program_concepts. People think the hidden program may
run two sources at once: a main concept c1 (one of 45 rules, prior lambda_0/45
each, or a magnitude range of 1-100, prior (1-lambda_0)/100^2 each), from which
each example is a size-principle draw with uncertain reliability
a ~ Beta(alpha*kappa, (1-alpha)*kappa), and one secondary source per set that
produced the remaining (stray) examples: random glitches (prior 1-mix) or a
second program c2 (prior mix). The one change: the second program is drawn from
the learner's own prior over programs -- a rule with prior mix*lambda_0/45 or a
magnitude range with prior mix*(1-lambda_0)/100^2 -- rather than always being a
rule. So off-pattern examples that cluster in magnitude (e.g. 91, 95 next to
13, 23, 43) are read as a second, range-based source and license "yes" to
nearby numbers, just as off-pattern members of a small rule license its other
members. No parameter is added.

"Yes" to t: t in c1, or c2 actually produced at least one example and t in c2.
For a set X (n examples), split into genuine G (|G| = j) and strays X\\G:
  p(X, G | c1, src) = B(u+j, v+n-j)/B(u,v) * prod_G 1[x in c1]/|c1| * prod_{X\\G} q_src(x).
All parameter-free sums go into prepare_observed as E[set, class1, j, type]
and Y[cell, class1, j, type] with type = (second rule, second range, glitches).
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

_N = np.arange(1, MAX_N + 1, dtype="float64")[:, None]
_J = np.arange(MAX_N + 1, dtype="float64")[None, :]
_VALID = _J <= _N
_N_MINUS_J = np.maximum(_N - _J, 0.0)


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


with pm.Model() as model:
    ev_table = pm.Data("ev_table", np.zeros((1, 2, MAX_N + 1, 3)))  # (sets, class, j, type)
    yes_table = pm.Data("yes_table", np.zeros((1, 2, MAX_N + 1, 3)))  # (cells, class, j, type)
    set_n_idx = pm.Data("set_n_idx", np.zeros(1, dtype="int64"))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa
    mix_logit = pm.Normal("mix_logit", mu=0.0, sigma=1.5)
    mix = pm.Deterministic("mix", pm.math.sigmoid(mix_logit))  # prior that the second source is a rule

    c_table = pt.where(_VALID, pt.exp(_betaln(u + _J, v + _N_MINUS_J) - _betaln(u, v)), 0.0)
    c_set = c_table[set_n_idx]  # (sets, j)
    w_class = pt.stack([lambda_0, 1 - lambda_0])
    # Second source: a program from the same prior (rule or range), or glitches.
    w_type = pt.stack([mix * lambda_0, mix * (1 - lambda_0), 1 - mix])
    w = w_class[None, :, None, None] * c_set[:, None, :, None] * w_type[None, None, None, :]
    evidence = (w * ev_table).sum(axis=(1, 2, 3))
    yes_mass = (w[cell_set] * yes_table).sum(axis=(1, 2, 3))
    p_cell = yes_mass / evidence[cell_set]
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell, 1e-9, 1 - 1e-9)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _set_tables(ex):
    """Evidence (class, j, type) and yes-mass (100 targets, class, j, type) for one set."""
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

    w1 = [genuine_w(RULE_MEM, RULE_SIZE, 1.0 / N_RULES), genuine_w(INT_MEM, INT_SIZE, 1.0 / 100**2)]
    w1_tot = [w.sum(axis=0) for w in w1]  # (G,)
    w1_out = [RULE_MEM_NOT.T @ w1[0], INT_MEM_NOT.T @ w1[1]]  # (100, G): c1 lacks target
    # Second-program source (rule or range): strays X\\G must lie in c2.
    def second_w(mem, size, prior):
        ex_in2 = mem[:, ex - 1] > 0
        ok2 = ~((~sub_in)[None, :, :] & ~ex_in2[:, None, :]).any(axis=2)
        w2 = ok2 * size[:, None] ** -(n - j_of)[None, :] * prior  # (concepts, G)
        w2_tot = w2.sum(axis=0)
        w2_out = (1.0 - mem).T @ w2  # (100, G): c2 lacks target
        w2_out[:, full] = w2_tot[full]  # no strays: c2 produced nothing, no generalisation
        return w2_tot, w2_out

    w2 = [second_w(RULE_MEM, RULE_SIZE, 1.0 / N_RULES), second_w(INT_MEM, INT_SIZE, 1.0 / 100**2)]
    w_unif = 0.01 ** (n - j_of)

    ev = np.zeros((2, MAX_N + 1, 3))
    yes = np.zeros((100, 2, MAX_N + 1, 3))
    for k in range(2):
        for s, (w2_tot, w2_out) in enumerate(w2):
            ev[k, :, s] = (w1_tot[k] * w2_tot) @ onehot_j
            yes[:, k, :, s] = (w1_tot[k] * w2_tot - w1_out[k] * w2_out) @ onehot_j
        ev[k, :, 2] = (w1_tot[k] * w_unif) @ onehot_j
        yes[:, k, :, 2] = ((w1_tot[k] - w1_out[k]) * w_unif) @ onehot_j
    scale = ev.max()
    return ev / scale, yes / scale


RULE_MEM_NOT = 1.0 - RULE_MEM
INT_MEM_NOT = 1.0 - INT_MEM


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    tables = [_set_tables(np.array(s)) for s in unique]
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: k for k, c in enumerate(cells)}
    return {
        "ev_table": np.stack([t[0] for t in tables]),
        "yes_table": np.stack([tables[i][1][t - 1] for i, t in cells]),
        "set_n_idx": np.array([len(s) - 1 for s in unique], dtype="int64"),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
