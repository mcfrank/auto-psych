"""Digit-reversal pair concepts in Bayesian concept learning.

People infer the hidden program as a Bayesian concept learner over familiar
number rules (all equally plausible, one shared prior mass), digit-sum rules
("digits add up to s", prior share graded smoothly with s) and the 5,050 ranges
of 1..100 (Erlang width prior), with the size principle and 10% of examples
treated as uniform noise. The one change to the concept space is a new kind of
program, "a two-digit number and its mirror image" ({ab, ba}, e.g. {38, 83},
{47, 74}; the 36 pairs with distinct non-zero digits), with its own inferred
prior mass shared equally among the pairs. Because each such program has only
two members, a set containing both a number and its reversal is a strong
suspicious coincidence, so people endorse the pair and reject other numbers.
A single member of a pair gives its reversal little support.
(Range outputs are uniform within the range.)
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
REVERSAL_PAIRS = [[10 * a + b, 10 * b + a] for a in range(1, 10) for b in range(a + 1, 10)]
FAMILIAR = familiar_rules()
RULES = FAMILIAR + DSUM_RULES + REVERSAL_PAIRS
N_FAMILIAR, N_DSUM, N_PAIRS = len(FAMILIAR), len(DSUM_RULES), len(REVERSAL_PAIRS)
N_RULES = len(RULES)
DSUM_VALUE = np.arange(1, 19, dtype="float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
WIDTH = HI - LO + 1
WIDTHS = np.arange(1, 101, dtype="float64")
N_PER_WIDTH = 101.0 - WIDTHS
# Ranges grouped by (width, number of examples inside 0..4): same likelihood within a group.
N_K = 5
N_GROUPS = 100 * N_K
G_WIDTH = np.repeat(WIDTHS, N_K)
G_K = np.tile(np.arange(N_K, dtype="float64"), 100)
LOG_IN = np.log(ALPHA / G_WIDTH + (1 - ALPHA) / 100)
LOG_OUT = np.log((1 - ALPHA) / 100)

with pm.Model() as model:
    rule_ll = pm.Data("rule_ll", np.zeros((1, N_RULES)))  # (sets, rules) log-lik of the examples
    n_examples = pm.Data("n_examples", np.zeros(1))  # (sets,)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    entry_cell = pm.Data("entry_cell", np.zeros(1, dtype="int64"))
    entry_flat_group = pm.Data("entry_flat_group", np.zeros(1, dtype="int64"))
    entry_count = pm.Data("entry_count", np.zeros(1))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of each kind: familiar rules, digit sums, ranges, reversal pairs.
    kind_weight = pm.Dirichlet("kind_weight", a=np.ones(4))
    log_w = pt.log(kind_weight)
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    dsum_decay = pm.Normal("dsum_decay", mu=0.0, sigma=1.0)

    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    log_per_width = erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH))
    dsum_log_share = -dsum_decay * DSUM_VALUE
    dsum_log_share = dsum_log_share - pt.logsumexp(dsum_log_share)
    log_prior_rules = pt.concatenate([
        pt.zeros(N_FAMILIAR) + log_w[0] - np.log(N_FAMILIAR),
        log_w[1] + dsum_log_share,
        pt.zeros(N_PAIRS) + log_w[3] - np.log(N_PAIRS),
    ])

    log_rules = log_prior_rules[None, :] + rule_ll
    log_range_unit = (
        log_w[2]
        + pt.repeat(log_per_width, N_K)[None, :]
        + G_K[None, :] * LOG_IN[None, :]
        + (n_examples[:, None] - G_K[None, :]) * LOG_OUT
    )
    log_z = pt.logsumexp(pt.concatenate([log_rules, log_range_unit + group_log_count], axis=1), axis=1)
    post_rules = pt.exp(log_rules - log_z[:, None])[cell_set]
    post_range = pt.exp(log_range_unit - log_z[:, None]).reshape((-1,))[entry_flat_group] * entry_count
    in_concept = pt.inc_subtensor(
        pt.zeros(cell_set.shape[0])[entry_cell], post_range
    ) + (post_rules * cell_rule_member).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys((s, int(r["target"])) for s, r in zip(sets, rows)))
    cell_index = {c: i for i, c in enumerate(cells)}
    rule_ll, n_examples, group_ids = [], [], []
    for s in unique:
        idx = np.array(s) - 1
        rule_ll.append(RULE_LOG_LIK[:, idx].sum(axis=1))
        n_examples.append(len(s))
        k = sum(((LO <= e) & (e <= HI)).astype(int) for e in s)
        group_ids.append((WIDTH - 1) * N_K + k)
    counts = np.stack([np.bincount(g, minlength=N_GROUPS) for g in group_ids]).astype("float64")
    cell_group_count = np.zeros((len(cells), N_GROUPS))
    for c, (s, t) in enumerate(cells):
        g = group_ids[set_index[s]][(LO <= t) & (t <= HI)]
        cell_group_count[c] = np.bincount(g, minlength=N_GROUPS)
    entry_cell, entry_group = np.nonzero(cell_group_count)
    cell_sets = np.array([set_index[s] for s, _ in cells], dtype="int64")
    return {
        "rule_ll": np.stack(rule_ll),
        "n_examples": np.array(n_examples, dtype="float64"),
        "group_log_count": np.where(counts > 0, np.log(np.maximum(counts, 1)), -1e4),
        "cell_set": cell_sets,
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "entry_cell": entry_cell.astype("int64"),
        "entry_flat_group": (cell_sets[entry_cell] * N_GROUPS + entry_group).astype("int64"),
        "entry_count": cell_group_count[entry_cell, entry_group],
        "row_cell": np.array([cell_index[(s, int(r["target"]))] for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
