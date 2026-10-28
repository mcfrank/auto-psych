"""Serial-position-weighted Bayesian concept learning.

People learn the hidden program's concept by Bayesian inference over 42
arithmetic rules and every interval of 1-100 (size principle, 10% of examples
uniform noise), but each example's evidence is weighted by its serial position
in the order shown: the weights are a softmax of rho * position, rescaled to
sum to the number of examples, so rho > 0 means later examples dominate
(recency), rho < 0 earlier ones (primacy), and the total evidence is unchanged.
Rules share a prior p_mix with Dirichlet weights; intervals share 1 - p_mix
equally (as in bigelow_piantadosi_independent). "Yes" is the posterior
probability that the concept contains the target.

For speed, each set's intervals are grouped by (which examples they contain,
their size): every interval in a group has the same likelihood whatever the
position weights, so a group is one hypothesis with log-count prior mass and
a precomputed share of its intervals containing each target. The posterior is
exactly the one over all 5,092 hypotheses.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_EX = 4
N_INTERVALS = 5050
PAD_LOG_COUNT = -1e4  # padding groups get no prior mass


def rule_sets():
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
    return rules


RULES = rule_sets()
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_EXAMPLE_LOG_LIK = np.log(
    RULE_MEMBERSHIP * (1 - EPSILON) / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100
)
OUT_LOG_LIK = np.log(EPSILON / 100)


def interval_groups(examples):
    """Group intervals [m, n] by (examples inside, size): counts and target memberships."""
    groups = {}
    for m in range(1, 101):
        for n in range(m, 101):
            key = (tuple(m <= x <= n for x in examples), n - m + 1)
            if key not in groups:
                groups[key] = np.zeros(101)
            groups[key][0] += 1.0
            groups[key][m : n + 1] += 1.0  # index t counts intervals containing t
    return groups


with pm.Model() as model:
    # (positions, sets, hypotheses): log P(x_pos | h), zero past the set's end.
    pos_log_lik = pm.Data("pos_log_lik", np.zeros((MAX_EX, 1, 1)))
    pos_mask = pm.Data("pos_mask", np.ones((MAX_EX, 1)))  # (positions, sets)
    group_log_count = pm.Data("group_log_count", np.zeros((1, 1)))  # (sets, groups)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_member = pm.Data("cell_member", np.zeros((1, 1)))  # (cells, hyp): share containing target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    rho = pm.Normal("rho", mu=0.0, sigma=1.0)  # serial-position gradient

    # Position weights: softmax over the shown positions, scaled to sum to n.
    positions = np.arange(MAX_EX, dtype="float64")[:, None]
    raw = pt.exp(rho * (positions - 1.5)) * pos_mask
    weights = raw / raw.sum(axis=0, keepdims=True) * pos_mask.sum(axis=0, keepdims=True)
    set_log_lik = (weights[:, :, None] * pos_log_lik).sum(axis=0)  # (sets, hyp)

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :]
    log_groups = pt.log1p(-p_mix) - np.log(N_INTERVALS) + group_log_count
    log_prior = pt.concatenate([pt.repeat(log_rules, group_log_count.shape[0], axis=0), log_groups], axis=1)
    log_joint = log_prior + set_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[cell_set] * cell_member).sum(axis=1)  # (cells,)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    per_set = [interval_groups(s) for s in unique]
    k_max = max(len(g) for g in per_set)
    n_hyp = N_RULES + k_max
    pos_log_lik = np.zeros((MAX_EX, len(unique), n_hyp))
    pos_mask = np.zeros((MAX_EX, len(unique)))
    log_count = np.full((len(unique), k_max), PAD_LOG_COUNT)
    share = np.zeros((len(unique), k_max, 101))
    for i, (s, groups) in enumerate(zip(unique, per_set)):
        for p, x in enumerate(s):
            pos_log_lik[p, i, :N_RULES] = RULE_EXAMPLE_LOG_LIK[:, x - 1]
            pos_mask[p, i] = 1.0
        for k, ((inside, size), counts) in enumerate(groups.items()):
            log_count[i, k] = np.log(counts[0])
            share[i, k] = counts / counts[0]
            in_log_lik = np.log((1 - EPSILON) / size + EPSILON / 100)
            for p, flag in enumerate(inside):
                pos_log_lik[p, i, N_RULES + k] = in_log_lik if flag else OUT_LOG_LIK
    cell_set = np.array([index[s] for s, _ in cells], dtype="int64")
    cell_member = np.concatenate(
        [RULE_MEMBERSHIP[:, [t - 1 for _, t in cells]].T,
         np.stack([share[index[s], :, t] for s, t in cells])],
        axis=1,
    )
    return {
        "pos_log_lik": pos_log_lik,
        "pos_mask": pos_mask,
        "group_log_count": log_count,
        "cell_set": cell_set,
        "cell_member": cell_member,
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
