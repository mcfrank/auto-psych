"""Refinement of split_rule_range_tempering: strong/weak sampling mixture instead of power-law tempering.

People do Bayesian concept learning over 42 weighted rules and all 5,050
intervals of 1..100 (10% noisy examples) and answer with
p_yes = sigmoid(gain * logit(P(target in concept | D)) + bias), as in the
incumbent, with the evidence treated separately for rules and for ranges. The
single change: instead of raising the size-principle likelihood to a power,
people believe each example was either drawn from the concept itself (strong
sampling, probability 1/|h|) or drawn at random from 1..100 and merely checked
to fit the concept (weak sampling, 1/100), with an inferred strong-sampling
share for rules and another for ranges. The size penalty therefore saturates:
it separates tiny concepts sharply from each other but barely penalises large
concepts relative to medium ones (after 8, 64, "multiples of 8" stays
plausible against "powers of 8").
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)


def concept_sets():
    """42 rules, then 5,050 intervals."""
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
    intervals = [list(range(m, n + 1)) for m in range(1, 101) for n in range(m, 101)]
    return rules, intervals


RULES, INTERVALS = concept_sets()
N_RULES = len(RULES)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_WIDTH = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
LOG_N_INTERVALS = np.log(len(INTERVALS))
MAX_SET = 4
N_GROUPS = 100 * (MAX_SET + 1)
# Group g holds intervals of width g // 5 + 1 containing g % 5 of the examples.
GROUP_WIDTH = (np.arange(N_GROUPS) // (MAX_SET + 1) + 1).astype("float64")
GROUP_INSIDE = (np.arange(N_GROUPS) % (MAX_SET + 1)).astype("float64")
LOG_NOISE = np.log(EPSILON / 100)


def member_log_lik(strong_share, size):
    """Log-probability of an example that is a member of a concept of this size."""
    return pt.log((1 - EPSILON) * (strong_share / size + (1 - strong_share) / 100) + EPSILON / 100)


with pm.Model() as model:
    rule_inside = pm.Data("rule_inside", np.zeros((1, N_RULES)))  # (sets, rules) examples in rule
    set_size = pm.Data("set_size", np.ones(1))  # (sets,)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, N_GROUPS)))  # (trials, groups)
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    # The single change: strong-sampling share (vs weak sampling), for rules and for ranges.
    rule_strong_share = pm.Beta("rule_strong_share", 2, 2)
    range_strong_share = pm.Beta("range_strong_share", 2, 2)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)

    n_ex = set_size[:, None]
    rule_ll = rule_inside * member_log_lik(rule_strong_share, RULE_SIZE)[None, :] + (n_ex - rule_inside) * LOG_NOISE
    group_ll = (
        GROUP_INSIDE[None, :] * member_log_lik(range_strong_share, GROUP_WIDTH)[None, :]
        + (n_ex - GROUP_INSIDE[None, :]) * LOG_NOISE
    )
    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_ll
    log_groups = pt.log1p(-p_mix) - LOG_N_INTERVALS + group_log_count + group_ll
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    rule_part = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)).reshape((-1,))[row_cell]
    interval_part = (posterior[:, N_RULES:][row_set] * group_in_concept).sum(axis=1)
    concept_p = pt.clip(rule_part + interval_part, 1e-6, 1 - 1e-6)

    belief_logit = pt.log(concept_p) - pt.log1p(-concept_p)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias), 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    log_count = np.full((n_sets, N_GROUPS), -700.0)  # empty groups get no mass
    share = np.zeros((n_sets, N_GROUPS, 100))  # (sets, groups, targets)
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        inside = INTERVAL_MEMBERSHIP[:, ex].sum(axis=1).astype("int64")
        group = (INTERVAL_WIDTH - 1) * (MAX_SET + 1) + inside
        for g in np.unique(group):
            members = group == g
            log_count[i, g] = np.log(members.sum())
            share[i, g] = INTERVAL_MEMBERSHIP[members].mean(axis=0)
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    return {
        "rule_inside": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_size": np.array([float(len(s)) for s in unique]),
        "group_log_count": log_count,
        "group_in_concept": share[set_idx, :, tgt],
        "row_set": set_idx,
        "row_cell": set_idx * 100 + tgt,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
