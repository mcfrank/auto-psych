"""Refinement of tempered_evidence_concept_guessing: Bayesian concept learning
over 42 rules (each with an inferred prior weight, sharing p_mix) and all 5,050
intervals of 1..100 (equally likely), examples sampled from the concept (size
principle), tempered evidence (log P(D | h) multiplied by an inferred
`evidence_weight`) and occasional guessing, as in the incumbent. The single
change, taken from example_noise_concept_learning: the share of examples
believed to be random numbers from 1..100 rather than from the concept is an
inferred `epsilon` instead of a fixed 10%. Tempering sets how strongly small
consistent concepts are favoured; epsilon sets how readily a concept survives
an example it does not contain.
p_yes = (1 - guess_rate) * P_w(target in concept | D) + guess_rate * guess_bias.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

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
N_RULES, N_INTERVALS = len(RULES), len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
SIZES = MEMBERSHIP.sum(axis=1)

RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
RULE_SIZES = SIZES[:N_RULES]
MAX_K = 4  # sets have at most four examples
N_GROUPS = 100 * (MAX_K + 1)
GROUP_SIZE = np.repeat(np.arange(1, 101), MAX_K + 1).astype("float64")  # group g = (width - 1) * 5 + k
GROUP_K = np.tile(np.arange(MAX_K + 1), 100).astype("float64")

# With noise rate e, an example x has P(x | h) = (1 - e)/|h| + e/100 if x is in h,
# else e/100. So log P(D | h) = k_h log((1-e)/|h| + e/100) + (n - k_h) log(e/100),
# where k_h counts the examples in h. Intervals with the same width and k_h share
# a likelihood, so per set they are pooled into 500 (width, k) groups:
# prepare_observed gives each group's log count and, per target, the share of
# its intervals containing that target. The posterior is exact.

with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples in rule
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, N_GROUPS, 100)))
    n_examples = pm.Data("n_examples", np.zeros(1))  # (sets,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    # Tempered evidence, as in the incumbent.
    evidence_weight = pm.LogNormal("evidence_weight", mu=0.0, sigma=0.5)
    # The single change: the example-noise rate is inferred, not fixed at 10%.
    epsilon = pm.Beta("epsilon", 2, 8)

    log_out = pt.log(epsilon / 100)
    rule_ll = rule_k * pt.log((1 - epsilon) / RULE_SIZES + epsilon / 100)[None, :] \
        + (n_examples[:, None] - rule_k) * log_out
    group_ll = GROUP_K[None, :] * pt.log((1 - epsilon) / GROUP_SIZE + epsilon / 100)[None, :] \
        + (n_examples[:, None] - GROUP_K[None, :]) * log_out
    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + evidence_weight * rule_ll
    log_groups = pt.log1p(-p_mix) - np.log(N_INTERVALS) + group_log_count + evidence_weight * group_ll
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) \
        + (posterior[:, N_RULES:, None] * group_in_concept).sum(axis=1)  # (sets, 100)
    concept_p = pt.clip(in_concept.reshape((-1,))[row_cell], 0.0, 1.0)

    guess_rate = pm.Beta("guess_rate", 2, 8)
    guess_bias = pm.Beta("guess_bias", 2, 2)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip((1 - guess_rate) * concept_p + guess_rate * guess_bias, 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    k_all = np.stack([MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique])  # (sets, hyps)
    iv_member = MEMBERSHIP[N_RULES:]
    iv_size = SIZES[N_RULES:].astype(int)
    log_count = np.full((len(unique), N_GROUPS), -1e4)
    in_concept = np.zeros((len(unique), N_GROUPS, 100))
    for i in range(len(unique)):
        g = (iv_size - 1) * (MAX_K + 1) + k_all[i, N_RULES:].astype(int)
        counts = np.bincount(g, minlength=N_GROUPS).astype("float64")
        sums = np.zeros((N_GROUPS, 100))
        np.add.at(sums, g, iv_member)
        nz = counts > 0
        log_count[i, nz] = np.log(counts[nz])
        in_concept[i, nz] = sums[nz] / counts[nz, None]
    return {
        "rule_k": k_all[:, :N_RULES],
        "group_log_count": log_count,
        "group_in_concept": in_concept,
        "n_examples": np.array([float(len(s)) for s in unique]),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
