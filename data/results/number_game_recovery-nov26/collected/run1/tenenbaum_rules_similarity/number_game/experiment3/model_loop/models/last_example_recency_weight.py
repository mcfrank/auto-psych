"""Last-example recency weight in Bayesian concept learning.

People do Bayesian concept learning over familiar number rules (all equally
plausible a priori, one shared mass), digit-sum rules ("digits add up to s",
prior share graded smoothly with s) and the 5,050 ranges of 1..100 (Erlang
width prior, the size principle, 10% of examples uniform noise). The one
change is that the examples are read in order and the most recently seen
example of a multi-example set stays most vivid: it counts as evidence with its
own inferred weight (above one: recency dominance; below one: it is
discounted), while every earlier example counts normally. Single-example sets
are unaffected. So which concepts survive depends on which example was shown
last, e.g. 16, 8, 64 and 64, 8, 16 are not read identically.
(Range outputs are uniform within the range: the incumbent's centre weighting
was inferred to be essentially zero.)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


def familiar_rules():
    rules = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "even": list(range(2, 101, 2)),
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    return list(rules.values())


DSUM_RULES = [[n for n in NUMBERS if sum(map(int, str(n))) == t] for t in range(1, 19)]
RULES = familiar_rules() + DSUM_RULES
N_FAMILIAR = len(RULES) - 18
N_RULES = len(RULES)
DSUM_VALUE = np.arange(1, 19, dtype="float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
WIDTH = HI - LO + 1
WIDTHS = np.arange(1, 101, dtype="float64")
N_PER_WIDTH = 101.0 - WIDTHS
# Ranges are grouped by (width, number of earlier examples inside 0..3, last
# example inside 0/1): every range in a group has the same likelihood.
N_K = 4
N_GROUPS = 100 * N_K * 2
G_WIDTH = np.repeat(WIDTHS, N_K * 2)
G_K = np.tile(np.repeat(np.arange(N_K, dtype="float64"), 2), 100)
G_L = np.tile(np.array([0.0, 1.0]), 100 * N_K)
LOG_IN = np.log(ALPHA / G_WIDTH + (1 - ALPHA) / 100)
LOG_OUT = np.log((1 - ALPHA) / 100)

with pm.Model() as model:
    rule_earlier = pm.Data("rule_earlier", np.zeros((1, N_RULES)))  # (sets, rules)
    rule_last = pm.Data("rule_last", np.zeros((1, N_RULES)))  # 0 for one-example sets
    n_earlier = pm.Data("n_earlier", np.zeros(1))  # (sets,)
    has_last = pm.Data("has_last", np.zeros(1))  # (sets,)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # log #ranges per group
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    # Sparse (cell, range group) entries: flat (set, group) index, #ranges containing the target.
    entry_cell = pm.Data("entry_cell", np.zeros(1, dtype="int64"))
    entry_flat_group = pm.Data("entry_flat_group", np.zeros(1, dtype="int64"))
    entry_count = pm.Data("entry_count", np.zeros(1))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    kind_weight = pm.Dirichlet("kind_weight", a=np.ones(3))  # familiar rules, digit sums, ranges
    log_w = pt.log(kind_weight)
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    dsum_decay = pm.Normal("dsum_decay", mu=0.0, sigma=1.0)
    # Evidential weight of the last-seen example (1 = ideal learner).
    last_weight = pm.LogNormal("last_weight", mu=0.0, sigma=0.5)

    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    log_per_width = erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH))
    dsum_log_share = -dsum_decay * DSUM_VALUE
    dsum_log_share = dsum_log_share - pt.logsumexp(dsum_log_share)
    log_prior_rules = pt.concatenate([pt.zeros(N_FAMILIAR) + log_w[0] - np.log(N_FAMILIAR), log_w[1] + dsum_log_share])

    log_rules = log_prior_rules[None, :] + rule_earlier + last_weight * rule_last
    last_ll = pt.as_tensor(G_L * LOG_IN + (1 - G_L) * LOG_OUT)
    log_range_unit = (
        log_w[2]
        + pt.repeat(log_per_width, N_K * 2)[None, :]
        + G_K[None, :] * LOG_IN[None, :]
        + (n_earlier[:, None] - G_K[None, :]) * LOG_OUT
        + has_last[:, None] * last_weight * last_ll[None, :]
    )  # (sets, groups): log joint of one range in the group
    log_z = pt.logsumexp(pt.concatenate([log_rules, log_range_unit + group_log_count], axis=1), axis=1)
    post_rules = pt.exp(log_rules - log_z[:, None])[cell_set]
    post_range = pt.exp(log_range_unit - log_z[:, None]).reshape((-1,))[entry_flat_group] * entry_count
    in_concept = pt.inc_subtensor(
        pt.zeros(cell_set.shape[0])[entry_cell], post_range
    ) + (post_rules * cell_rule_member).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _group_index(width, k, last_inside):
    return ((width - 1) * N_K + k) * 2 + last_inside


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys((s, int(r["target"])) for s, r in zip(sets, rows)))
    cell_index = {c: i for i, c in enumerate(cells)}
    rule_earlier, rule_last, n_earlier, has_last, group_ids = [], [], [], [], []
    for s in unique:
        idx = np.array(s) - 1
        earlier, last = (idx, None) if len(s) == 1 else (idx[:-1], idx[-1])
        rule_earlier.append(RULE_LOG_LIK[:, earlier].sum(axis=1))
        rule_last.append(np.zeros(N_RULES) if last is None else RULE_LOG_LIK[:, last])
        n_earlier.append(len(earlier))
        has_last.append(0.0 if last is None else 1.0)
        k = sum(((LO <= e + 1) & (e + 1 <= HI)).astype(int) for e in earlier)
        l = np.zeros(len(LO), dtype=int) if last is None else ((LO <= last + 1) & (last + 1 <= HI)).astype(int)
        group_ids.append(_group_index(WIDTH, k, l))
    counts = np.stack([np.bincount(g, minlength=N_GROUPS) for g in group_ids]).astype("float64")
    cell_group_count = np.zeros((len(cells), N_GROUPS))
    for c, (s, t) in enumerate(cells):
        g = group_ids[set_index[s]][(LO <= t) & (t <= HI)]
        cell_group_count[c] = np.bincount(g, minlength=N_GROUPS)
    entry_cell, entry_group = np.nonzero(cell_group_count)
    return {
        "rule_earlier": np.stack(rule_earlier),
        "rule_last": np.stack(rule_last),
        "n_earlier": np.array(n_earlier, dtype="float64"),
        "has_last": np.array(has_last),
        "group_log_count": np.where(counts > 0, np.log(np.maximum(counts, 1)), -1e4),
        "cell_set": np.array([set_index[s] for s, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "entry_cell": entry_cell.astype("int64"),
        "entry_flat_group": (np.array([set_index[s] for s, _ in cells])[entry_cell] * N_GROUPS + entry_group).astype("int64"),
        "entry_count": cell_group_count[entry_cell, entry_group],
        "row_cell": np.array([cell_index[(s, int(r["target"]))] for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
