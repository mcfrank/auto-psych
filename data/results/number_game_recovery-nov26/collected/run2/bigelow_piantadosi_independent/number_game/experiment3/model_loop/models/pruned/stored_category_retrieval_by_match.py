"""Stored number categories retrieved by the examples' match.

People do not infer the concept behind the examples; they hold familiar number
categories in memory (odd, even, primes, squares and cubes, powers, multiples,
final or contained digit, multiples of 5, and stretches of the number line) and
the example set retrieves them as a memory probe does: each category comes to
mind in proportion to the share of the examples it contains (raised to a match
sharpness), small, specific categories more readily than broad ones (a fixed
size penalty, the same however many examples there are). The target is judged
"yes" to the extent that the retrieved categories contain it. Because
retrieval depends on the share of matching examples, not on how improbable the
examples would be under a category, more examples fitting a category do not
sharpen generalization onto it, and a category missing one example is still
retrieved fairly often.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


def rule_concepts():
    """(kind, members) for each stored rule category; kinds index KINDS."""
    primes = [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))]
    rules = [
        (0, [n for n in NUMBERS if n % 2 == 1]),
        (1, list(range(2, 101, 2))),
        (2, primes),
        (3, [n * n for n in range(1, 11)]),
        (3, [n**3 for n in range(1, 5)]),
        (4, [2, 4, 8, 16, 64]),
        (4, [2, 4, 8, 16, 32, 37, 64]),
    ]
    rules += [(8 if m == 5 else 5, list(range(m, 101, m))) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rules += [(4, [m**k for k in range(1, 8) if m**k <= 100]) for m in range(2, 11)]
    rules += [(6, [n for n in NUMBERS if n % 10 == d]) for d in range(10)]
    rules += [(7, [n for n in NUMBERS if str(d) in str(n)]) for d in range(10)]
    return rules


KINDS = ["odd", "even", "primes", "squares_cubes", "powers", "multiples", "final_digit", "contains_digit", "multiples_of_5"]
_RULES = rule_concepts()
RULE_KIND = np.array([k for k, _ in _RULES], dtype="int64")
N_RULES = len(_RULES)
KIND_LOG_COUNT = np.log(np.bincount(RULE_KIND, minlength=len(KINDS))).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for _, c in _RULES], dtype="float64")
INTERVALS = np.array([(m, n) for m in NUMBERS for n in range(m, 101)])
N_INTERVALS = len(INTERVALS)
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= INTERVALS[:, :1]) & (NUMBERS[None, :] <= INTERVALS[:, 1:])).astype("float64")
RULE_LOG_SIZE = np.log(RULE_MEMBERSHIP.sum(axis=1))
INTERVAL_WIDTH = INTERVALS[:, 1] - INTERVALS[:, 0] + 1
# Ranges are grouped by (how many examples they contain, k = 0..4; width w):
# all ranges in a group are retrieved equally strongly. Group index k*100 + w-1.
N_GROUPS = 500
GROUP_LOG_WIDTH = np.tile(np.log(np.arange(1, 101, dtype="float64")), 5)
GROUP_K = np.repeat(np.arange(5), 100).astype("float64")

with pm.Model() as model:
    rule_match = pm.Data("rule_match", np.zeros((1, N_RULES)))  # (sets, rules) share of examples contained
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups) log number of ranges
    set_n = pm.Data("set_n", np.ones(1))  # (sets,) number of examples
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,) distinct (set, target)
    cell_rule_in = pm.Data("cell_rule_in", np.zeros((1, N_RULES)))  # (cells, rules) target in rule
    cell_group_frac = pm.Data("cell_group_frac", np.zeros((1, N_GROUPS)))  # (cells, groups) share of ranges holding target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # How accessible each kind of stored category is, against the ranges.
    kind_logit = pm.Normal("kind_logit", mu=0.0, sigma=2.5, shape=len(KINDS))
    # Small, specific categories are retrieved more readily than broad ones.
    size_weight = pm.LogNormal("size_weight", mu=0.0, sigma=0.7)
    # How sharply retrieval favours categories holding more of the examples.
    match_weight = pm.LogNormal("match_weight", mu=1.0, sigma=1.0)

    def match_term(share):
        return match_weight * pt.log(0.98 * share + 0.01)

    log_rules = (kind_logit[RULE_KIND] - KIND_LOG_COUNT[RULE_KIND] - size_weight * RULE_LOG_SIZE)[None, :] + match_term(rule_match)
    group_share = GROUP_K[None, :] / set_n[:, None]
    log_groups = -np.log(N_INTERVALS) + group_log_count - size_weight * GROUP_LOG_WIDTH[None, :] + match_term(group_share)
    strength = pt.concatenate([log_rules, log_groups], axis=1)
    retrieved = pt.exp(strength - pt.logsumexp(strength, axis=1, keepdims=True))[cell_set]  # (cells, categories)
    endorse = (retrieved[:, :N_RULES] * cell_rule_in).sum(axis=1) + (retrieved[:, N_RULES:] * cell_group_frac).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(endorse, 1e-6, 1 - 1e-6)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Per distinct set: share of its examples each rule contains and the ranges grouped
    by (examples contained, width); per distinct (set, target) cell: which rules and
    what share of each range group hold the target. Empty groups get log count -1e4."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys((s, int(r["target"])) for s, r in zip(sets, rows)))
    cell_index = {c: k for k, c in enumerate(cells)}
    group_log_count = np.full((len(unique), N_GROUPS), -1e4)
    set_groups = []
    for i, st in enumerate(unique):
        k = INTERVAL_MEMBERSHIP[:, np.array(st) - 1].sum(axis=1).astype("int64")
        g = k * 100 + INTERVAL_WIDTH - 1
        counts = np.bincount(g, minlength=N_GROUPS)
        group_log_count[i, counts > 0] = np.log(counts[counts > 0])
        set_groups.append((g, np.maximum(counts, 1)))
    group_frac = np.zeros((len(cells), N_GROUPS))
    for c, (st, t) in enumerate(cells):
        g, counts = set_groups[set_index[st]]
        group_frac[c] = np.bincount(g, weights=INTERVAL_MEMBERSHIP[:, t - 1], minlength=N_GROUPS) / counts
    return {
        "rule_match": np.stack([RULE_MEMBERSHIP[:, np.array(st) - 1].mean(axis=1) for st in unique]),
        "group_log_count": group_log_count,
        "set_n": np.array([len(st) for st in unique], dtype="float64"),
        "cell_set": np.array([set_index[st] for st, _ in cells], dtype="int64"),
        "cell_rule_in": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_group_frac": group_frac,
        "row_cell": np.array([cell_index[(st, int(r["target"]))] for st, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(float(r.get("rating", 0))) for r in rows], dtype="int64"),
    }
