"""Kind-level over-commitment.

People infer the hidden program as a Bayesian concept learner over number rules
(parity, special sets, multiples, final and contained digits, exception rules,
digit sums; a prior mass per kind shared equally by its members) and numeric
ranges (Erlang prior on width), with the size principle and 10% of examples
treated as uniform noise. The one distortion: inference runs in two stages.
People first judge what kind of program it is and weight each kind by its
posterior probability raised to a commitment exponent (above one: they
over-commit to the leading kind; below one: they hedge across kinds), and then
generalise within each kind exactly as the ideal learner does. An exponent of
one recovers the normative learner.

Speed: range likelihoods do not depend on parameters, so prepare_observed
precomputes per set (and per observed set-target cell) the sum over ranges of
each width of exp(log-likelihood); the graph works on (cells x 100) arrays.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
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
    rules.update({f"digit sum {t}": [n for n in NUMBERS if sum(map(int, str(n))) == t] for t in range(1, 19)})
    return rules


def family(name):
    if name in ("odd", "even"):
        return 0
    if name.startswith("multiples"):
        return 2
    if name.startswith("ends"):
        return 3
    if name.startswith("contains"):
        return 4
    if name.startswith("digit sum"):
        return 6
    if name.startswith("powers of 2,"):
        return 5  # gerrymandered exception rules
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
N_RULES = len(_RULES)
N_KINDS = 8  # seven rule kinds + ranges
RULE_FAMILY = np.array([family(n) for n in _RULES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in _RULES.values()], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
# Membership split by kind: (kinds-1, rules, numbers).
KIND_MEMBERSHIP = np.stack([RULE_MEMBERSHIP * (RULE_FAMILY == k)[:, None] for k in range(7)])
KIND_ONEHOT = np.eye(7)[RULE_FAMILY]  # (rules, 7)

LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
WIDTH = HI - LO + 1
WIDTHS = np.arange(1, 101, dtype="float64")
N_PER_WIDTH = 101.0 - WIDTHS
WIDTH_ONEHOT = np.eye(100)[WIDTH - 1]  # (ranges, widths)


def interval_log_lik(examples):
    ll = np.zeros(len(LO))
    for x in examples:
        inside = (LO <= x) & (x <= HI)
        ll += np.log(ALPHA * inside / WIDTH + (1 - ALPHA) / 100)
    return ll


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules), minus set max
    set_width_sum = pm.Data("set_width_sum", np.zeros((1, 100)))  # (sets, widths)
    cell_width_sum = pm.Data("cell_width_sum", np.zeros((1, 100)))  # (cells, widths)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the seven rule kinds and of ranges.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(N_KINDS))
    # Characteristic range width of the Erlang size prior.
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    # Commitment exponent on the posterior over kinds (1 = normative).
    kind_commitment = pm.LogNormal("kind_commitment", mu=0.0, sigma=0.5)

    rule_prior = family_weight[:7][RULE_FAMILY] / FAMILY_SIZE[RULE_FAMILY]
    rule_weight = rule_prior[None, :] * pt.exp(rule_log_lik)  # (sets, rules)
    kind_total_rules = pt.dot(rule_weight, KIND_ONEHOT)  # (sets, 7)
    kind_in_rules = pt.stack(
        [pt.dot(rule_weight, KIND_MEMBERSHIP[k])[cell_set, cell_target] for k in range(7)], axis=1
    )  # (cells, 7)

    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    log_per_range = erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH))
    width_weight = family_weight[7] * pt.exp(log_per_range)  # (widths,)
    int_total = pt.dot(set_width_sum, width_weight)  # (sets,)
    int_in = pt.dot(cell_width_sum, width_weight)  # (cells,)

    kind_total = pt.concatenate([kind_total_rules, int_total[:, None]], axis=1) + 1e-300
    kind_in = pt.concatenate([kind_in_rules, int_in[:, None]], axis=1)
    # Stage 1: distorted posterior over kinds.
    log_kind = kind_commitment * pt.log(kind_total)
    kind_post = pt.exp(log_kind - log_kind.max(axis=1, keepdims=True))
    kind_post = kind_post / kind_post.sum(axis=1, keepdims=True)  # (sets, kinds)
    # Stage 2: normative generalisation within each kind.
    p_cell = (kind_post[cell_set] * kind_in / kind_total[cell_set]).sum(axis=1)
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    rule_ll = np.zeros((len(unique), N_RULES))
    set_width_sum = np.zeros((len(unique), 100))
    range_weights = []
    for i, s in enumerate(unique):
        r_ll = RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1)
        i_ll = interval_log_lik(s)
        top = max(r_ll.max(), i_ll.max())
        rule_ll[i] = r_ll - top
        e = np.exp(i_ll - top)
        range_weights.append(e)
        set_width_sum[i] = e @ WIDTH_ONEHOT
    cell_width_sum = np.zeros((len(cells), 100))
    for j, (s, t) in enumerate(cells):
        e = range_weights[set_index[s]] * ((LO <= t) & (t <= HI))
        cell_width_sum[j] = e @ WIDTH_ONEHOT
    return {
        "rule_log_lik": rule_ll,
        "set_width_sum": set_width_sum,
        "cell_width_sum": cell_width_sum,
        "cell_set": np.array([set_index[s] for s, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
