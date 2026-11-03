"""I chose large_number_biased_sampling and refined it. People do the same Bayesian
concept learning (45 independently weighted rules plus multiples of 13-50, intervals
of 1..100 with an Erlang width prior, an inferred share of random examples, examples
drawn without replacement), and they believe that when the program draws from a range
of magnitudes it favours the larger members (an in-range output n is produced in
proportion to n**gamma relative to the range's average weight, gamma inferred), but
that rule concepts are sampled uniformly. The one change is that the magnitude bias
applies to ranges only: a range stretching below the smallest example is cheap under
the size principle, so "yes" extends below examples that fit no rule more readily than
above them, while generalisation within rules is unchanged (gamma = 0 recovers
extended_multiples_concepts). "Yes" is the posterior probability that the concept
contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_EX = 4


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
N_BASE = len(_RULES)
EXT_M = list(range(13, 51))  # less-familiar multiples rules
N_EXT = len(EXT_M)
_RULES.update({f"multiples of {m}": list(range(m, 101, m)) for m in EXT_M})
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")  # (R, 100)
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)

# Every interval [lo, hi] of 1..100.
INT_LO, INT_HI = np.array([(a, b) for a in range(1, 101) for b in range(a, 101)]).T
N_INT = len(INT_LO)
INT_SIZE = (INT_HI - INT_LO + 1).astype("float64")
SIZES_F = np.arange(1, 101).astype("float64")
N_PER_SIZE = 101.0 - SIZES_F
INT_MEMBERSHIP = ((NUMBERS[None, :] >= INT_LO[:, None]) & (NUMBERS[None, :] <= INT_HI[:, None])).astype("float64")
LOG_N = np.log(NUMBERS / 50.0)

# Uniform without-replacement table over (examples shown, interval size, examples inside).
TAB_SIZE = np.tile(np.repeat(SIZES_F, MAX_EX + 1), MAX_EX)
TAB_NIN = np.tile(np.arange(MAX_EX + 1), 100 * MAX_EX).astype("float64")
TAB_SIZE_IDX = TAB_SIZE.astype("int64") - 1


def tab_index(n_ex, size, n_in):
    return ((n_ex - 1) * 100 + (size - 1)) * (MAX_EX + 1) + n_in


END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_BASE - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))
    n_examples = pm.Data("n_examples", np.ones(1))
    z_tab = pm.Data("z_tab", np.zeros((1, 100), dtype="int64"))  # table index, empty intervals by size
    z_count = pm.Data("z_count", np.zeros((1, 100)))  # number of empty intervals by size
    # One entry per (set, interval containing at least one example).
    e_int = pm.Data("e_int", np.zeros(1, dtype="int64"))
    e_tab = pm.Data("e_tab", np.zeros(1, dtype="int64"))
    e_nin = pm.Data("e_nin", np.zeros(1))
    e_logx = pm.Data("e_logx", np.zeros(1))  # sum of log(x / 50) over the examples inside
    e_lo = pm.Data("e_lo", np.zeros(1, dtype="int64"))  # set * 100 + lo - 1
    e_hi = pm.Data("e_hi", np.zeros(1, dtype="int64"))  # set * 100 + hi - 1
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_z_frac = pm.Data("cell_z_frac", np.zeros((1, 100)))  # share of empty intervals of each size containing t
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    noise = pm.Beta("noise", 1.5, 6.0)  # share of examples believed random
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    ext_share = pm.Beta("ext_share", 1.0, 4.0)
    all_weights = pt.concatenate([(1 - ext_share) * rule_weights, pt.ones(N_EXT) * ext_share / N_EXT])

    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    log_erlang = pt.log(SIZES_F) - SIZES_F / sigma
    log_per_interval = log_erlang - pt.logsumexp(log_erlang + np.log(N_PER_SIZE))

    log_out = pt.log(noise / 100.0)
    rule_ll = (n_examples[:, None] - rule_n_in) * log_out
    for j in range(MAX_EX):
        rule_term = pt.log((1 - noise) / np.maximum(RULE_SIZE - j, 1.0) + noise / 100.0)
        rule_ll = rule_ll + pt.gt(rule_n_in, j) * rule_term[None, :]
    # Work in linear space relative to n_examples * log_out: every term is bounded
    # (rules/intervals at most (100 / noise)**4 times the all-noise likelihood).
    rule_rel = rule_ll - n_examples[:, None] * log_out
    u_rules = lambda_0 * all_weights[None, :] * pt.exp(rule_rel)  # (S, R)

    # Uniform without-replacement interval likelihood times prior, by (n_ex, size, n_in).
    tab = -TAB_NIN * log_out + pt.log1p(-lambda_0) + log_per_interval[TAB_SIZE_IDX]
    for j in range(MAX_EX):
        tab = tab + (TAB_NIN > j) * pt.log((1 - noise) / np.maximum(TAB_SIZE - j, 1.0) + noise / 100.0)
    w_tab = pt.exp(tab)

    # The one change: within a range, output n is produced in proportion to n**gamma
    # relative to the range's average weight; rules sample their members uniformly.
    gamma = pm.Normal("gamma", mu=0.0, sigma=1.0)
    cum_w = pt.concatenate([pt.zeros(1), pt.cumsum(pt.exp(gamma * LOG_N))])
    log_rel_mass = pt.log((cum_w[INT_HI] - cum_w[INT_LO - 1]) / INT_SIZE)  # log mean weight, (I,)
    u_e = w_tab[e_tab] * pt.exp(gamma * e_logx - e_nin * log_rel_mass[e_int])
    n_sets = n_examples.shape[0]
    u_z = w_tab[z_tab] * z_count  # (S, 100) empty intervals by size
    # Interval mass containing t = mass with lo <= t minus mass with hi < t.
    lo_mass = pt.inc_subtensor(pt.zeros(n_sets * 100)[e_lo], u_e).reshape((n_sets, 100))
    hi_mass = pt.inc_subtensor(pt.zeros(n_sets * 100)[e_hi], u_e).reshape((n_sets, 100))
    contain = pt.cumsum(lo_mass, axis=1) - pt.cumsum(hi_mass, axis=1) + hi_mass  # (S, 100)
    z_total = u_rules.sum(axis=1) + u_z.sum(axis=1) + lo_mass.sum(axis=1)
    in_mass = (
        (u_rules[cell_set] * cell_rule_member).sum(axis=1)
        + (u_z[cell_set] * cell_z_frac).sum(axis=1)
        + contain[cell_set, cell_target]
    )
    in_concept = in_mass / z_total[cell_set]
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set`, `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_n_in = np.zeros((S, N_RULES))
    n_ex = np.zeros(S)
    z_tab = np.zeros((S, 100), dtype="int64")
    z_count = np.zeros((S, 100))
    z_contain = np.zeros((S, 100, 100))
    es, ei, et, en, ex_, el, eh = [], [], [], [], [], [], []
    for i, s in enumerate(unique):
        s = s[:MAX_EX]
        k = len(s)
        n_ex[i] = k
        xs = np.array(s)
        rule_n_in[i] = RULE_MEMBERSHIP[:, xs - 1].sum(axis=1)
        inside = (xs[None, :] >= INT_LO[:, None]) & (xs[None, :] <= INT_HI[:, None])  # (I, k)
        nin = inside.sum(axis=1)
        logx = (inside * np.log(xs / 50.0)[None, :]).sum(axis=1)
        z_tab[i] = tab_index(k, np.arange(1, 101), 0)
        empty = np.nonzero(nin == 0)[0]
        z_count[i] = np.bincount(INT_SIZE[empty].astype("int64") - 1, minlength=100)
        np.add.at(z_contain[i], INT_SIZE[empty].astype("int64") - 1, INT_MEMBERSHIP[empty])
        full = np.nonzero(nin > 0)[0]
        ei.append(full)
        et.append(tab_index(k, INT_SIZE[full].astype("int64"), nin[full]))
        en.append(nin[full].astype("float64"))
        ex_.append(logx[full])
        el.append(i * 100 + INT_LO[full] - 1)
        eh.append(i * 100 + INT_HI[full] - 1)
    z_frac = np.where(z_count[:, :, None] > 0, z_contain / np.maximum(z_count, 1)[:, :, None], 0.0)
    keys = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cs = np.array([c[0] for c in cells], dtype="int64")
    ct = np.array([c[1] for c in cells], dtype="int64")
    return {
        "rule_n_in": rule_n_in,
        "n_examples": n_ex,
        "z_tab": z_tab,
        "z_count": z_count,
        "e_int": np.concatenate(ei).astype("int64"),
        "e_tab": np.concatenate(et).astype("int64"),
        "e_nin": np.concatenate(en),
        "e_logx": np.concatenate(ex_),
        "e_lo": np.concatenate(el).astype("int64"),
        "e_hi": np.concatenate(eh).astype("int64"),
        "cell_set": cs,
        "cell_target": ct,
        "cell_rule_member": RULE_MEMBERSHIP[:, ct].T.copy(),
        "cell_z_frac": z_frac[cs, :, ct],
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r.get("rating", 0) or 0) for r in rows], dtype="int64"),
    }
