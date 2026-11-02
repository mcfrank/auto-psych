"""Example-cued generation of rule-plus-exception concepts.

People generate "rule plus exception" concepts bottom-up from the examples:
while checking a rule against the examples, an example the rule fails to cover
captures attention, and the learner entertains the rule together with a small
fuzzy region around that noticed example ("ends in 2, and also around 27"),
giving these example-cued exception concepts a fitted share alpha of the rule's
belief (split over the rule's off-rule examples, radius h = 0..3 with
P(h) proportional to exp(-rho h)), instead of the tiny share an exception at an
arbitrary number would get. Rules that cover every example spawn no exception
concepts and keep their full prior. Otherwise: the same 45 rules (final-digit
rules boosted by kappa), Erlang-width intervals with compressed-number-line
sampling, tempered size principle (beta) and exception rate eps.
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


RULE_NAMES = list(rule_concepts().keys())
RULES = list(rule_concepts().values())
# Rules read straight off the final digit: parity and "ends in d".
RULE_SALIENT = np.array([n in ("odd", "even") or n.startswith("ends in") for n in RULE_NAMES], dtype="float64")
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
# All intervals [a, b] of 1..100.
INT_A = np.array([a for a in NUMBERS for b in NUMBERS if b >= a])
INT_B = np.array([b for a in NUMBERS for b in NUMBERS if b >= a])
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
INT_SIZE_IDX = INT_B - INT_A  # width - 1
H_MAX = 3  # largest exception-neighbourhood radius
H_RADII = np.arange(H_MAX + 1, dtype="float64")
SIZES = np.arange(1, 101, dtype="float64")
LOG_SIZE_COUNT = np.log(101 - SIZES)  # intervals of each width


def _scatter_sum(n, index, values):
    return pt.inc_subtensor(pt.zeros(n)[index], values)


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples inside the rule
    rule_has_exc = pm.Data("rule_has_exc", np.zeros((1, N_RULES)))  # 1 if some example is off the rule
    set_n = pm.Data("set_n", np.ones(1))  # (sets,): number of examples
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))
    pair_int = pm.Data("pair_int", np.zeros(1, dtype="int64"))
    pair_k = pm.Data("pair_k", np.zeros(1))
    pair_log_x = pm.Data("pair_log_x", np.zeros(1))
    empty_size_count = pm.Data("empty_size_count", np.zeros((1, 100)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_empty_size_count = pm.Data("cell_empty_size_count", np.zeros((1, 100)))
    hit_pair = pm.Data("hit_pair", np.zeros(1, dtype="int64"))
    hit_cell = pm.Data("hit_cell", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    # Example-cued exception concepts: rule R plus the radius-h region around an off-rule example.
    exc_set = pm.Data("exc_set", np.zeros(1, dtype="int64"))
    exc_rule = pm.Data("exc_rule", np.zeros(1, dtype="int64"))
    exc_h = pm.Data("exc_h", np.zeros(1, dtype="int64"))
    exc_size = pm.Data("exc_size", np.ones(1))
    exc_k = pm.Data("exc_k", np.zeros(1))
    exc_log_m = pm.Data("exc_log_m", np.zeros(1))  # log number of off-rule examples cueing R
    # Sparse (cell, concept): the concept's region holds a target outside its rule.
    ehit_cell = pm.Data("ehit_cell", np.zeros(1, dtype="int64"))
    ehit_exc = pm.Data("ehit_exc", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)  # sharpness of the size principle
    eps = pm.Beta("eps", 2, 5)  # believed rate of off-concept examples
    gamma = pm.HalfNormal("gamma", sigma=1.0)  # compression of the mental number line
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)  # typical interval width
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)  # log prior boost of final-digit rules
    alpha = pm.Beta("alpha", 2, 2)  # share of a violated rule's belief moved to example-cued exceptions
    rho = pm.LogNormal("rho", mu=0.0, sigma=1.0)  # how fast prior falls with the exception's radius

    n_sets = set_n.shape[0]
    n_cells = cell_set.shape[0]
    log_noise = pt.log(eps) - np.log(100)
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)
    rule_log_lik = rule_k * rule_in[None, :] + (set_n[:, None] - rule_k) * log_noise
    log_rule_prior = kappa * RULE_SALIENT - pm.math.logsumexp(kappa * RULE_SALIENT)
    log_rules = (
        pt.log(lambda_0) + pt.log1p(-alpha * rule_has_exc) + log_rule_prior[None, :] + beta * rule_log_lik
    )
    shift = pt.max(log_rules, axis=1)
    w_rules = pt.exp(log_rules - shift[:, None])

    log_h_prior = -rho * H_RADII - pm.math.logsumexp(-rho * H_RADII)
    exc_in = pt.log((1 - eps) / exc_size + eps / 100)
    log_exc = (
        pt.log(lambda_0) + pt.log(alpha) + log_rule_prior[exc_rule] + log_h_prior[exc_h] - exc_log_m
        + beta * (exc_k * exc_in + (set_n[exc_set] - exc_k) * log_noise)
    )
    w_exc = pt.exp(log_exc - shift[exc_set])
    w_aug = _scatter_sum(n_sets * N_RULES, exc_set * N_RULES + exc_rule, w_exc).reshape((n_sets, N_RULES))

    cum_w = pt.concatenate([pt.zeros(1), pt.cumsum(pt.exp(-gamma * LOG_NUMBERS))])
    log_w = pt.log(cum_w[INT_B] - cum_w[INT_A - 1])
    pair_lik = (
        pair_k * (pt.log1p(-eps) - log_w[pair_int])
        - gamma * pair_log_x
        + (set_n[pair_set] - pair_k) * log_noise
    )
    log_width = pt.log(SIZES) - SIZES / sigma
    log_z = pm.math.logsumexp(LOG_SIZE_COUNT + log_width)
    log_size_prior = pt.log1p(-lambda_0) + log_width - log_z
    w_pairs = pt.exp(log_size_prior[pt.as_tensor(INT_SIZE_IDX)[pair_int]] + beta * pair_lik - shift[pair_set])
    size_prior = pt.exp(log_size_prior)
    empty_scale = pt.exp(beta * set_n * log_noise - shift)
    w_empty = empty_scale * (empty_size_count * size_prior[None, :]).sum(axis=1)

    z = w_rules.sum(axis=1) + w_aug.sum(axis=1) + _scatter_sum(n_sets, pair_set, w_pairs) + w_empty
    numer = (
        ((w_rules + w_aug)[cell_set] * cell_rule_member).sum(axis=1)
        + _scatter_sum(n_cells, ehit_cell, w_exc[ehit_exc])
        + _scatter_sum(n_cells, hit_cell, w_pairs[hit_pair])
        + empty_scale[cell_set] * (cell_empty_size_count * size_prior[None, :]).sum(axis=1)
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

    pair_set, pair_int, pair_k, pair_log_x, empty_size_count = [], [], [], [], []
    set_pairs, set_empty = [], []
    offset = 0
    for i, st in enumerate(unique):
        ex = np.array(st)
        inside = (INT_A[:, None] <= ex[None, :]) & (INT_B[:, None] >= ex[None, :])
        k = inside.sum(axis=1)
        listed = np.nonzero(k > 0)[0]
        empty = np.nonzero(k == 0)[0]
        pair_set += [i] * len(listed)
        pair_int += list(listed)
        pair_k += list(k[listed].astype(float))
        pair_log_x += list((inside[listed] * np.log(ex)[None, :]).sum(axis=1))
        empty_size_count.append(np.bincount(INT_SIZE_IDX[empty], minlength=100).astype("float64"))
        set_pairs.append((offset, listed))
        set_empty.append(empty)
        offset += len(listed)

    hit_pair, hit_cell, cell_empty_size_count = [], [], []
    for c, (i, t) in enumerate(cells):
        start, listed = set_pairs[i]
        has_t = np.nonzero((INT_A[listed] <= t) & (INT_B[listed] >= t))[0]
        hit_pair += list(start + has_t)
        hit_cell += [c] * len(has_t)
        empty = set_empty[i]
        holds = empty[(INT_A[empty] <= t) & (INT_B[empty] >= t)]
        cell_empty_size_count.append(np.bincount(INT_SIZE_IDX[holds], minlength=100).astype("float64"))

    exc_rows = []  # (set, rule, h, size, k, log m)
    exc_extra = []  # per concept: set of non-rule numbers in its region
    set_exc = []  # per set: list of concept indices
    rule_has_exc = np.zeros((len(unique), N_RULES))
    for i, st in enumerate(unique):
        ex_count = np.bincount(np.array(st), minlength=101)[1:]
        mine = []
        for r in range(N_RULES):
            in_rule = RULE_MEMBERSHIP[r] > 0
            off = sorted({x for x in st if not in_rule[x - 1]})
            if not off:
                continue
            rule_has_exc[i, r] = 1.0
            k_rule = ex_count[in_rule].sum()
            for x in off:
                for h in range(H_MAX + 1):
                    extra = np.arange(max(1, x - h), min(100, x + h) + 1)
                    extra = extra[~in_rule[extra - 1]]
                    mine.append(len(exc_rows))
                    exc_rows.append((i, r, h, RULE_SIZE[r] + len(extra),
                                     k_rule + ex_count[extra - 1].sum(), np.log(len(off))))
                    exc_extra.append(set(extra.tolist()))
        set_exc.append(mine)
    ehit_cell, ehit_exc = [], []
    for c, (i, t) in enumerate(cells):
        for j in set_exc[i]:
            if t in exc_extra[j]:
                ehit_cell.append(c)
                ehit_exc.append(j)
    exc = np.array(exc_rows, dtype="float64")

    return {
        "exc_set": exc[:, 0].astype("int64"),
        "exc_rule": exc[:, 1].astype("int64"),
        "exc_h": exc[:, 2].astype("int64"),
        "exc_size": exc[:, 3],
        "exc_k": exc[:, 4],
        "exc_log_m": exc[:, 5],
        "ehit_cell": np.array(ehit_cell, dtype="int64"),
        "ehit_exc": np.array(ehit_exc, dtype="int64"),
        "rule_has_exc": rule_has_exc,
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "pair_set": np.array(pair_set, dtype="int64"),
        "pair_int": np.array(pair_int, dtype="int64"),
        "pair_k": np.array(pair_k, dtype="float64"),
        "pair_log_x": np.array(pair_log_x, dtype="float64"),
        "empty_size_count": np.stack(empty_size_count),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_empty_size_count": np.stack(cell_empty_size_count),
        "hit_pair": np.array(hit_pair, dtype="int64"),
        "hit_cell": np.array(hit_cell, dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
