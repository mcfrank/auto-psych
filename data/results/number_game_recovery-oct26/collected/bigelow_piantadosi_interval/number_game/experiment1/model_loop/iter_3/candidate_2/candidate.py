"""Predictive representativeness judgment.

People do not ask whether the target belongs to the concept they have inferred;
they ask how representative the target would be as the program's next output:
how much more likely the program is to produce exactly this number than a
random number, given the concepts the examples suggest (42 weighted rules and
all intervals of 1..100, size principle, 10% noisy examples). "Yes" rises
smoothly with the log of this posterior-predictive generation probability
relative to chance:
p_yes = sigmoid(criterion + slope * log(100 * P(next number = target | D))).
Members of small concepts are thus endorsed more readily than equally certain
members of large ones.
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
GEN_PROB = MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100
EXAMPLE_LOG_LIK = np.log(GEN_PROB)
RULE_GEN = GEN_PROB[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_GEN = GEN_PROB[N_RULES:]
INTERVAL_WIDTH = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
LOG_N_INTERVALS = np.log(len(INTERVALS))
MAX_SET = 4
N_GROUPS = 100 * (MAX_SET + 1)

# Intervals of the same width containing the same number of examples share a
# prior and a likelihood, so prepare_observed collapses each set's intervals
# into <= 500 groups (width x examples inside); the predictive is exact.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_gen = pm.Data("group_gen", np.zeros((1, N_GROUPS)))  # (trials, groups) mean P(target | interval)
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    criterion = pm.Normal("criterion", mu=0.0, sigma=2.0)
    slope = pm.HalfNormal("slope", sigma=2.0)

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_groups = pt.log1p(-p_mix) - LOG_N_INTERVALS + group_log_count + group_log_lik
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    # Posterior-predictive probability that the program's next output is the target.
    rule_part = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_GEN)).reshape((-1,))[row_cell]
    interval_part = (posterior[:, N_RULES:][row_set] * group_gen).sum(axis=1)
    predictive = rule_part + interval_part

    representativeness = pt.log(100.0 * predictive)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(criterion + slope * representativeness), 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    log_count = np.full((n_sets, N_GROUPS), -700.0)  # empty groups get no mass
    log_lik = np.zeros((n_sets, N_GROUPS))
    gen = np.zeros((n_sets, N_GROUPS, 100))  # (sets, groups, targets)
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        inside = INTERVAL_MEMBERSHIP[:, ex].sum(axis=1).astype("int64")
        ll = EXAMPLE_LOG_LIK[N_RULES:, ex].sum(axis=1)
        group = (INTERVAL_WIDTH - 1) * (MAX_SET + 1) + inside
        for g in np.unique(group):
            members = group == g
            log_count[i, g] = np.log(members.sum())
            log_lik[i, g] = ll[members][0]
            gen[i, g] = INTERVAL_GEN[members].mean(axis=0)
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    return {
        "rule_log_lik": np.stack([EXAMPLE_LOG_LIK[:N_RULES, np.array(s) - 1].sum(axis=1) for s in unique]),
        "group_log_count": log_count,
        "group_log_lik": log_lik,
        "group_gen": gen[set_idx, :, tgt],
        "row_set": set_idx,
        "row_cell": set_idx * 100 + tgt,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
