"""Serial-position example forgetting: people do Bayesian concept learning over
42 rules and every interval of 1-100 (kind-level rule prior, rule share p_mix,
size principle with 10% uniform examples, Erlang width prior with a
decade-bounded range preference), but generalize only from the examples still
in working memory. The last example shown is always retained; an example
followed by L later examples is retained with probability retain**L, so older
examples are more often lost. "Yes" is the average, over remembered subsets, of
the posterior probability that the concept contains the target, weighted by
the probability that each subset is what remains. Order of examples matters.
"""

import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_LAG = 3


def concept_sets():
    rules = [
        [2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36],
        [7, 49], [8, 64], [9, 81], [10, 100],
    ]
    rules += [[i for i in NUMBERS if i % n == 0] for n in range(3, 13)]
    rules += [[i for i in NUMBERS if i % 10 == d] for d in range(1, 10)]
    rules += [[i for i in NUMBERS if str(d) in str(i)] for d in range(1, 10)]
    rules.append([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    rules.append([i for i in NUMBERS if i % 2 == 0])
    rules.append([i for i in NUMBERS if i % 2 == 1])
    rules.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    rules.append([1, 8, 27, 64])
    kinds = [0] * 9 + [1] * 10 + [2] * 9 + [3] * 9 + [4] * 5
    bounds = [(m, n) for m in range(1, 101) for n in range(m, 101)]
    return rules, kinds, bounds


RULES, RULE_KINDS, BOUNDS = concept_sets()
N_RULES = len(RULES)
N_KINDS = 5
KIND_MATRIX = np.eye(N_KINDS)[RULE_KINDS]
KIND_COUNTS = KIND_MATRIX.sum(axis=0)
INTERVALS = [list(range(m, n + 1)) for m, n in BOUNDS]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]

INTERVAL_WIDTH = np.array([n - m + 1 for m, n in BOUNDS])
DECADE_BOUNDED = np.array([(m - 1) % 10 == 0 and n % 10 == 0 for m, n in BOUNDS], dtype="int64")
GROUP_KEYS, GROUP_OF_INTERVAL, GROUP_SIZES = np.unique(
    np.stack([INTERVAL_WIDTH, DECADE_BOUNDED], axis=1), axis=0, return_inverse=True, return_counts=True
)
GROUP_OF_INTERVAL = GROUP_OF_INTERVAL.ravel()
GROUP_WIDTHS = GROUP_KEYS[:, 0].astype("float64")
GROUP_ALIGNED = GROUP_KEYS[:, 1].astype("float64")
N_GROUPS = len(GROUP_KEYS)
LAGS = np.arange(1, MAX_LAG + 1, dtype="float64")

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    group_log_evidence = pm.Data("group_log_evidence", np.zeros((1, N_GROUPS)))
    # One row per (cell, remembered subset) pair; the subset always contains
    # the last example. pair_kept_lag / pair_lost_lag mark, for lags 1..3,
    # whether the example shown that many positions before the last is kept / lost.
    pair_subset = pm.Data("pair_subset", np.zeros(1, dtype="int64"))
    pair_cell = pm.Data("pair_cell", np.zeros(1, dtype="int64"))
    pair_kept_lag = pm.Data("pair_kept_lag", np.zeros((1, MAX_LAG)))
    pair_lost_lag = pm.Data("pair_lost_lag", np.zeros((1, MAX_LAG)))
    pair_in_hypothesis = pm.Data("pair_in_hypothesis", np.zeros((1, N_RULES + N_GROUPS)))
    cell_count = pm.Data("cell_count", np.ones(1, dtype="float64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    rule_weights = pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS)
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    decade_boost = pm.Normal("decade_boost", mu=1.0, sigma=1.5)
    # Probability an example survives one later example in memory.
    retain = pm.Beta("retain", 4.0, 1.5)

    log_interval = np.log(GROUP_WIDTHS) - GROUP_WIDTHS / width_scale + decade_boost * GROUP_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval + np.log(GROUP_SIZES))

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_groups = pt.log1p(-p_mix) + log_interval_prior[None, :] + group_log_evidence
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    pair_yes = (posterior[pair_subset] * pair_in_hypothesis).sum(axis=1)

    # Retention at lag L is retain**L; subsets containing the last example sum to 1.
    log_keep = LAGS * pt.log(retain)
    log_lose = pt.log1p(-pt.exp(log_keep))
    log_w = pt.dot(pair_kept_lag, log_keep) + pt.dot(pair_lost_lag, log_lose)
    cell_yes = pt.zeros(cell_count.shape[0])
    cell_yes = pt.inc_subtensor(cell_yes[pair_cell], pt.exp(log_w) * pair_yes)

    p_yes = pm.Deterministic("p_yes", pt.clip(cell_yes[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}

    subsets = {}
    subset_list = []
    cell_pairs = []
    for s, _ in cells:
        pairs = []
        n = len(s)
        earlier = range(n - 1)
        for k in range(0, n):
            for idx in itertools.combinations(earlier, k):
                kept = np.zeros(MAX_LAG)
                lost = np.zeros(MAX_LAG)
                for i in earlier:
                    lag = n - 1 - i
                    if i in idx:
                        kept[lag - 1] += 1
                    else:
                        lost[lag - 1] += 1
                key = tuple(sorted([s[i] for i in idx] + [s[-1]]))
                if key not in subsets:
                    subsets[key] = len(subset_list)
                    subset_list.append(key)
                pairs.append((subsets[key], kept, lost))
        cell_pairs.append(pairs)

    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in subset_list])
    interval_log_lik = set_log_lik[:, N_RULES:]
    log_evidence = np.empty((len(subset_list), N_GROUPS))
    in_concept = np.empty((len(subset_list), N_GROUPS, 100))
    for g in range(N_GROUPS):
        members = GROUP_OF_INTERVAL == g
        log_evidence[:, g] = logsumexp(interval_log_lik[:, members], axis=1)
        weights = np.exp(interval_log_lik[:, members] - log_evidence[:, g : g + 1])
        in_concept[:, g] = weights @ MEMBERSHIP[N_RULES:][members]

    p_subset, p_cell, p_kept, p_lost, p_in = [], [], [], [], []
    for c, ((s, t), pairs) in enumerate(zip(cells, cell_pairs)):
        for sub, kept, lost in pairs:
            p_subset.append(sub)
            p_cell.append(c)
            p_kept.append(kept)
            p_lost.append(lost)
            p_in.append(np.concatenate([RULE_MEMBERSHIP[:, t - 1], in_concept[sub, :, t - 1]]))

    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "group_log_evidence": log_evidence,
        "pair_subset": np.array(p_subset, dtype="int64"),
        "pair_cell": np.array(p_cell, dtype="int64"),
        "pair_kept_lag": np.stack(p_kept),
        "pair_lost_lag": np.stack(p_lost),
        "pair_in_hypothesis": np.stack(p_in),
        "cell_count": np.ones(len(cells)),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
