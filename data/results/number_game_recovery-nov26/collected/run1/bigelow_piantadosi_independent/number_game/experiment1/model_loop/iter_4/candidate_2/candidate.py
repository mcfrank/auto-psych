"""Costly compound concepts: numerical ranges restricted to a parity.

People can build compound concepts that join a numerical range with a parity
("even numbers between 20 and 60"), but considering such a two-part concept
carries a cognitive cost, so these conjunctions get only a fitted share rho of
the range prior. Otherwise the learner is the exception-tolerant, tempered
Bayesian concept learner with the same 45 rules and all intervals of 1..100,
members of a range sampled along a compressed number line
(P(x | range) = x^-gamma / sum of k^-gamma over the range's members). Once
considered, the size principle favours these small compound concepts, so people
fill in the range around the examples but reject immediate neighbours of the
wrong parity far more than a learner with only plain ranges.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


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
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
INT_A = np.array([a for a in NUMBERS for b in NUMBERS if b >= a])
INT_B = np.array([b for a in NUMBERS for b in NUMBERS if b >= a])
N_INT = len(INT_A)
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
LOG_INT_PRIOR = -2 * np.log(100)
# Range concepts: kind 0 = plain range, kind 1 = its odd members only, kind 2 = its even members only.
PARITY_MASK = np.stack([np.ones(100), (NUMBERS % 2 == 1).astype(float), (NUMBERS % 2 == 0).astype(float)])
# A parity-restricted range must contain at least one number of that parity.
_COUNT = np.stack([np.concatenate([[0], np.cumsum(m)]) for m in PARITY_MASK])
VALID = np.stack([(_COUNT[k][INT_B] - _COUNT[k][INT_A - 1]) > 0 for k in range(3)])  # (3, N_INT)
VALID_INDEX = np.nonzero(VALID.reshape(-1))[0]
# Position of each (kind, interval) concept in the valid-concept list.
CONCEPT_POS = np.full(3 * N_INT, -1, dtype="int64")
CONCEPT_POS[VALID_INDEX] = np.arange(len(VALID_INDEX))
COMPOUND = (VALID_INDEX >= N_INT).astype("float64")  # 1 for parity-restricted ranges


def _members(kind, x):
    """Whether numbers x have range-kind `kind`'s parity (ignoring the range bounds)."""
    x = np.asarray(x)
    return np.ones(x.shape, bool) if kind == 0 else (x % 2 == (1 if kind == 1 else 0))


def _segment_sum(values, ends):
    """Sums of consecutive runs of `values`; run j ends (exclusive) at ends[j]."""
    c = pt.concatenate([pt.zeros(1), pt.cumsum(values)])
    return c[ends] - c[pt.concatenate([pt.zeros(1, dtype="int64"), ends[:-1]])]


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples inside the rule
    set_n = pm.Data("set_n", np.ones(1))  # (sets,): number of examples
    # Range concepts holding >= 1 example, one entry per (set, concept).
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))
    set_end = pm.Data("set_end", np.ones(1, dtype="int64"))  # pairs are grouped by set
    pair_conc = pm.Data("pair_conc", np.zeros(1, dtype="int64"))  # index into valid concepts
    pair_k = pm.Data("pair_k", np.zeros(1))  # examples inside the concept
    pair_log_x = pm.Data("pair_log_x", np.zeros(1))  # sum of log x over those examples
    # Range concepts holding no example all share one likelihood: lumped per set.
    empty_log_plain = pm.Data("empty_log_plain", np.zeros(1))
    empty_log_comp = pm.Data("empty_log_comp", np.zeros(1))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_empty_plain = pm.Data("cell_empty_plain", np.zeros(1))  # share of empty concepts containing target
    cell_empty_comp = pm.Data("cell_empty_comp", np.zeros(1))
    hit_pair = pm.Data("hit_pair", np.zeros(1, dtype="int64"))
    cell_end = pm.Data("cell_end", np.zeros(1, dtype="int64"))  # hits are grouped by cell
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)  # sharpness of the size principle
    eps = pm.Beta("eps", 2, 5)  # believed rate of off-concept examples
    gamma = pm.HalfNormal("gamma", sigma=1.0)  # compression of the mental number line
    rho = pm.Beta("rho", 1, 3)  # share of range belief left for costly compound concepts

    log_noise = pt.log(eps) - np.log(100)
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)
    rule_log_lik = rule_k * rule_in[None, :] + (set_n[:, None] - rule_k) * log_noise
    log_rules = pt.log(lambda_0) - np.log(N_RULES) + beta * rule_log_lik
    shift = pt.max(log_rules, axis=1)  # per-set stabiliser
    w_rules = pt.exp(log_rules - shift[:, None])

    # Range members are drawn with weight x^-gamma among the concept's members.
    weights = pt.exp(-gamma * LOG_NUMBERS)[None, :] * PARITY_MASK  # (3, 100)
    cum_w = pt.concatenate([pt.zeros((3, 1)), pt.cumsum(weights, axis=1)], axis=1)
    log_w = pt.log((cum_w[:, INT_B] - cum_w[:, INT_A - 1]).reshape((-1,))[VALID_INDEX])
    log_plain_prior = pt.log1p(-lambda_0) + pt.log1p(-rho) + LOG_INT_PRIOR
    log_comp_prior = pt.log1p(-lambda_0) + pt.log(rho) - np.log(2.0) + LOG_INT_PRIOR
    # Tempered log weight of a concept holding k of the n examples (their log x summing to L):
    # prior + beta * (k * (log(1 - eps) - log W) - gamma * L + (n - k) * log noise), split into
    # a per-concept slope on k, a per-concept prior and a per-set all-noise term.
    conc_slope = beta * (pt.log1p(-eps) - log_noise - log_w)
    conc_prior = log_plain_prior + COMPOUND * (log_comp_prior - log_plain_prior)
    all_noise = beta * set_n * log_noise - shift
    w_pairs = pt.exp(
        pair_k * conc_slope[pair_conc] + conc_prior[pair_conc] - (beta * gamma) * pair_log_x + all_noise[pair_set]
    )
    w_empty_plain = pt.exp(log_plain_prior + empty_log_plain + all_noise)
    w_empty_comp = pt.exp(log_comp_prior + empty_log_comp + all_noise)

    z = w_rules.sum(axis=1) + _segment_sum(w_pairs, set_end) + w_empty_plain + w_empty_comp
    numer = (
        (w_rules[cell_set] * cell_rule_member).sum(axis=1)
        + _segment_sum(w_pairs[hit_pair], cell_end)
        + w_empty_plain[cell_set] * cell_empty_plain
        + w_empty_comp[cell_set] * cell_empty_comp
    )
    in_concept = numer / z[cell_set]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    trial_cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(trial_cells))
    cell_index = {c: i for i, c in enumerate(cells)}

    pair_set, pair_conc, pair_k, pair_log_x = [], [], [], []
    empty_plain, empty_comp = [], []
    set_pairs, set_empty = [], []
    offset = 0
    for i, st in enumerate(unique):
        ex = np.array(st)
        in_range = (INT_A[:, None] <= ex[None, :]) & (INT_B[:, None] >= ex[None, :])
        lists, empties = [], []
        for kind in range(3):
            inside = in_range & _members(kind, ex)[None, :]
            k = inside.sum(axis=1)
            listed = np.nonzero((k > 0) & VALID[kind])[0]
            empty = np.nonzero((k == 0) & VALID[kind])[0]
            pair_set += [i] * len(listed)
            pair_conc += list(CONCEPT_POS[kind * N_INT + listed])
            pair_k += list(k[listed].astype(float))
            pair_log_x += list((inside[listed] * np.log(ex)[None, :]).sum(axis=1))
            lists.append((offset, kind, listed))
            empties.append((kind, empty))
            offset += len(listed)
        n_plain = len(empties[0][1])
        n_comp = len(empties[1][1]) + len(empties[2][1])
        empty_plain.append(np.log(n_plain) if n_plain else -1e4)
        empty_comp.append(np.log(n_comp) if n_comp else -1e4)
        set_pairs.append(lists)
        set_empty.append(empties)

    hit_pair, cell_end, cell_empty_plain, cell_empty_comp = [], [], [], []
    for c, (i, t) in enumerate(cells):
        for start, kind, listed in set_pairs[i]:
            if not _members(kind, t):
                continue
            has_t = np.nonzero((INT_A[listed] <= t) & (INT_B[listed] >= t))[0]
            hit_pair += list(start + has_t)
        cell_end.append(len(hit_pair))
        empties = set_empty[i]
        e0 = empties[0][1]
        cell_empty_plain.append(((INT_A[e0] <= t) & (INT_B[e0] >= t)).mean() if len(e0) else 0.0)
        n_comp = len(empties[1][1]) + len(empties[2][1])
        hits = sum(
            int(((INT_A[e] <= t) & (INT_B[e] >= t)).sum()) for kind, e in empties[1:] if _members(kind, t)
        )
        cell_empty_comp.append(hits / n_comp if n_comp else 0.0)

    return {
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "pair_set": np.array(pair_set, dtype="int64"),
        "set_end": np.cumsum(np.bincount(pair_set, minlength=len(unique))).astype("int64"),
        "pair_conc": np.array(pair_conc, dtype="int64"),
        "pair_k": np.array(pair_k, dtype="float64"),
        "pair_log_x": np.array(pair_log_x, dtype="float64"),
        "empty_log_plain": np.array(empty_plain, dtype="float64"),
        "empty_log_comp": np.array(empty_comp, dtype="float64"),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_empty_plain": np.array(cell_empty_plain, dtype="float64"),
        "cell_empty_comp": np.array(cell_empty_comp, dtype="float64"),
        "hit_pair": np.array(hit_pair, dtype="int64"),
        "cell_end": np.array(cell_end, dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
