"""Normative Bayesian concept learning (bigelow_piantadosi_independent: 42
rules with inferred prior weights sharing p_mix, all 5,050 intervals of 1..100
equally likely, size principle) with one distortion: the learner's belief about
how often an example is a uniformly random number rather than a draw from the
concept is not fixed at 10% but is an inferred parameter `noise_belief`.
P(x | h) = (1 - e) [x in h] / |h| + e / 100, and "yes" is the posterior
probability that the concept contains the target. A high believed noise rate
lets rules that fit only some examples survive and broadens interval
generalization.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4


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
    return rules


RULES = concept_sets()
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
INTERVALS = [(m, n) for m in range(1, 101) for n in range(m, 101)]
N_INTERVALS = len(INTERVALS)
# Intervals grouped by (examples inside k, size s): group g = k * 100 + s - 1.
N_GROUPS = (MAX_N + 1) * 100
GROUP_K = np.repeat(np.arange(MAX_N + 1), 100).astype("float64")
GROUP_S = np.tile(np.arange(1, 101), MAX_N + 1).astype("float64")
TINY = 1e-300

with pm.Model() as model:
    set_n = pm.Data("set_n", np.ones(1))  # (sets,) number of examples
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules) examples inside rule
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # (cells,) target - 1
    cell_group_count = pm.Data("cell_group_count", np.zeros((1, N_GROUPS)))  # intervals containing target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    noise_belief = pm.Beta("noise_belief", 2, 8)

    log_noise = pt.log(noise_belief / 100.0)
    n_col = set_n[:, None]
    rule_ll = rule_k * pt.log((1 - noise_belief) / RULE_SIZE[None, :] + noise_belief / 100.0) + (n_col - rule_k) * log_noise
    group_ll = GROUP_K[None, :] * pt.log((1 - noise_belief) / GROUP_S[None, :] + noise_belief / 100.0) \
        + (n_col - GROUP_K[None, :]) * log_noise
    group_term = group_log_count + group_ll  # (sets, groups)
    interval_log_evidence = pt.logsumexp(group_term, axis=1)  # (sets,)

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_ll
    log_intervals = pt.log1p(-p_mix) - np.log(N_INTERVALS) + interval_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, rules + 1)

    # Per (set, target) cell: rule part + interval part.
    rule_part = (posterior[cell_set, :N_RULES] * pt.as_tensor(RULE_MEMBERSHIP.T)[cell_target]).sum(axis=1)
    group_w = pt.exp(group_ll[cell_set] - interval_log_evidence[cell_set][:, None])
    interval_part = posterior[cell_set, N_RULES] * (cell_group_count * group_w).sum(axis=1)
    concept_p = (rule_part + interval_part)[row_cell]
    p_yes = pm.Deterministic("p_yes", pt.clip(concept_p, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    lo = np.array([m for m, _ in INTERVALS])
    hi = np.array([n for _, n in INTERVALS])
    size = hi - lo + 1
    set_n = np.array([len(s) for s in unique], dtype="float64")
    rule_k = np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_k = []
    counts = np.zeros((len(unique), N_GROUPS))
    for i, s in enumerate(unique):
        k = sum(((lo <= x) & (x <= hi)).astype(int) for x in s)
        interval_k.append(k)
        np.add.at(counts[i], k * 100 + size - 1, 1.0)
    cells = list(dict.fromkeys((set_index[s], int(r["target"])) for s, r in zip(sets, rows)))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_counts = np.zeros((len(cells), N_GROUPS))
    for j, (si, t) in enumerate(cells):
        inside = (lo <= t) & (t <= hi)
        np.add.at(cell_counts[j], interval_k[si][inside] * 100 + size[inside] - 1, 1.0)
    return {
        "set_n": set_n,
        "rule_k": rule_k,
        "group_log_count": np.log(counts + TINY),
        "cell_set": np.array([c[0] for c in cells], dtype="int64"),
        "cell_target": np.array([c[1] - 1 for c in cells], dtype="int64"),
        "cell_group_count": cell_counts,
        "row_cell": np.array([cell_index[(set_index[s], int(r["target"]))] for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
