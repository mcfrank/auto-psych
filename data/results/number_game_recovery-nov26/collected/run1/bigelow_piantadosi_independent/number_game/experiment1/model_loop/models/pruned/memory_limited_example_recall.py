"""Memory-limited Bayesian concept learning (imperfect recall of the examples).

People do Bayesian concept learning over arithmetic rules and numerical
intervals, but they reason from an imperfect memory of the examples: each
example is independently retained with probability r, and they generalise
from only the examples they hold in mind (at least one). A target sharing a
property with just one example is then often endorsed, and generalisation
tightens less with more examples than an ideal learner predicts.

The learner given a recalled subset is the tempered-size-principle mixture
learner (45 rules with prior lambda_0, all intervals of 1..100 with prior
(1 - lambda_0) / 100^2, 90% strong / 10% uniform sampling, evidence exponent
beta). "Yes" is the posterior-predictive membership probability averaged over
recalled subsets, weighted r^k (1 - r)^(n - k), renormalised to exclude
recalling nothing.
"""

import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
MAX_EXAMPLES = 4
MAX_SUBSETS = 2**MAX_EXAMPLES - 1
N_GROUPS = 100 * (MAX_EXAMPLES + 1)


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
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)

with pm.Model() as model:
    # One row per distinct recalled subset of examples.
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, N_GROUPS)))
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))
    # One row per distinct (recalled subset, target) pair.
    pair_flat = pm.Data("pair_flat", np.zeros(1, dtype="int64"))  # subset * 100 + target - 1
    # Sparse entries sorted by pair: (subset * N_GROUPS + group, share of the group's
    # intervals holding the target); pair i owns entries pair_end[i-1]..pair_end[i].
    pair_end = pm.Data("pair_end", np.zeros(1, dtype="int64"))
    nz_group = pm.Data("nz_group", np.zeros(1, dtype="int64"))
    nz_share = pm.Data("nz_share", np.zeros(1, dtype="float64"))
    # (cells, MAX_SUBSETS) slots: each slot is one recalled subset of the cell's set.
    slot_pair = pm.Data("slot_pair", np.zeros((1, MAX_SUBSETS), dtype="int64"))
    slot_k = pm.Data("slot_k", np.zeros((1, MAX_SUBSETS)))  # examples recalled
    slot_miss = pm.Data("slot_miss", np.zeros((1, MAX_SUBSETS)))  # examples forgotten
    slot_mask = pm.Data("slot_mask", np.zeros((1, MAX_SUBSETS)))  # 1 = real slot
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)
    recall = pm.Beta("recall", 2, 2)  # probability each example is held in mind

    log_rules = pt.log(lambda_0) - np.log(N_RULES) + beta * rule_log_lik
    log_groups = pt.log1p(-lambda_0) - 2 * np.log(100) + beta * group_log_lik + group_log_count
    log_norm = pt.logsumexp(pt.concatenate([log_rules, log_groups], axis=1), axis=1, keepdims=True)
    rule_pred = pt.dot(pt.exp(log_rules - log_norm), RULE_MEMBERSHIP).flatten()  # (subsets * 100,)
    group_post = pt.exp(log_groups - log_norm).flatten()  # (subsets * groups,)
    cum = pt.concatenate([pt.zeros(1), pt.cumsum(group_post[nz_group] * nz_share)])
    group_pred = cum[pair_end] - cum[pt.concatenate([pt.zeros(1, dtype="int64"), pair_end[:-1]])]
    pair_pred = rule_pred[pair_flat] + group_pred

    w = slot_mask * pt.exp(slot_k * pt.log(recall) + slot_miss * pt.log1p(-recall))
    p_cell = (w * pair_pred[slot_pair]).sum(axis=1) / w.sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _interval_groups(examples):
    """Group intervals by (size s, examples inside k): log-lik, log count, membership share."""
    ex = np.array(examples)
    n = len(ex)
    log_lik = np.zeros(N_GROUPS)
    count = np.zeros(N_GROUPS)
    in_concept = np.zeros((N_GROUPS, 100))
    for s in range(1, 101):
        for k in range(MAX_EXAMPLES + 1):
            log_lik[(s - 1) * (MAX_EXAMPLES + 1) + k] = k * np.log(ALPHA / s + (1 - ALPHA) / 100) + (n - k) * np.log(
                (1 - ALPHA) / 100
            )
        starts = np.arange(1, 102 - s)
        ks = ((ex[None, :] >= starts[:, None]) & (ex[None, :] <= starts[:, None] + s - 1)).sum(axis=1)
        for k in np.unique(ks):
            g = (s - 1) * (MAX_EXAMPLES + 1) + k
            sel = starts[ks == k]
            count[g] = len(sel)
            cover = np.zeros(101)
            np.add.at(cover, sel - 1, 1)
            np.add.at(cover, sel - 1 + s, -1)
            in_concept[g] = np.cumsum(cover)[:100]
    nonzero = count > 0
    in_concept[nonzero] /= count[nonzero, None]
    log_count = np.where(nonzero, np.log(np.maximum(count, 1)), -1e4)
    return log_lik, log_count, in_concept


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    trial_cells = [(s, int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(trial_cells))
    cell_index = {c: i for i, c in enumerate(cells)}

    subsets = {}
    groups = []
    for s, _ in cells:
        for k in range(1, len(s) + 1):
            for sub in itertools.combinations(sorted(s), k):
                if sub not in subsets:
                    subsets[sub] = len(subsets)
                    groups.append(_interval_groups(sub))

    pairs = {}
    nc = len(cells)
    slot_pair = np.zeros((nc, MAX_SUBSETS), dtype="int64")
    slot_k = np.zeros((nc, MAX_SUBSETS))
    slot_miss = np.zeros((nc, MAX_SUBSETS))
    slot_mask = np.zeros((nc, MAX_SUBSETS))
    for i, (s, t) in enumerate(cells):
        j = 0
        for k in range(1, len(s) + 1):
            for sub in itertools.combinations(sorted(s), k):
                pair = (subsets[sub], t)
                if pair not in pairs:
                    pairs[pair] = len(pairs)
                slot_pair[i, j] = pairs[pair]
                slot_k[i, j] = k
                slot_miss[i, j] = len(s) - k
                slot_mask[i, j] = 1.0
                j += 1

    share = np.stack([groups[v][2][:, t - 1] for v, t in pairs])
    nz_pair, g = np.nonzero(share)
    nz_group = np.array([v for v, _ in pairs], dtype="int64")[nz_pair] * N_GROUPS + g
    nz_share = share[nz_pair, g]

    subs = list(subsets)
    return {
        "rule_log_lik": np.stack([RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in subs]),
        "group_log_lik": np.stack([g[0] for g in groups]),
        "group_log_count": np.stack([g[1] for g in groups]),
        "pair_flat": np.array([v * 100 + t - 1 for v, t in pairs], dtype="int64"),
        "pair_end": np.cumsum(np.bincount(nz_pair, minlength=len(pairs))).astype("int64"),
        "nz_group": nz_group,
        "nz_share": nz_share,
        "slot_pair": slot_pair,
        "slot_k": slot_k,
        "slot_miss": slot_miss,
        "slot_mask": slot_mask,
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
