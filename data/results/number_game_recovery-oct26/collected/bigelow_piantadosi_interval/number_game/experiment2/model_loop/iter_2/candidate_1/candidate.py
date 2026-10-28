"""Pedagogical (helpful-teacher) sampling of examples.

People assume the examples were chosen by a helpful teacher: for each example
the teacher picks a member of the concept with probability rising, by an
inferred power (pedagogy strength), with how strongly that single example
would point a naive learner to the concept. The learner inverts this teacher
model over 42 weighted rules and all intervals of 1..100 (10% noisy examples,
evidence tempered separately for rules and ranges) and answers through a
decision gain and shifted criterion. Pedagogy strength zero recovers ordinary
strong sampling (the size principle).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_SET = 4


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
    intervals = [list(range(m, n + 1)) for m in range(1, 101) for n in range(m, 101)]
    return rules, intervals


RULES, INTERVALS = concept_sets()
N_RULES = len(RULES)
N_INT = len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")  # (H, 100)
SIZE = MEMBERSHIP.sum(axis=1, keepdims=True)
# Naive single-example likelihood (strong sampling with noise), log scale.
NAIVE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / SIZE + EPSILON / 100)
NONMEMBER_MASK = (1.0 - MEMBERSHIP) * -1e4  # excludes non-members from the teacher's choice
IS_RULE = np.r_[np.ones(N_RULES), np.zeros(N_INT)]


with pm.Model() as model:
    set_examples = pm.Data("set_examples", np.zeros((1, MAX_SET), dtype="int64"))  # 0-based numbers
    set_mask = pm.Data("set_mask", np.zeros((1, MAX_SET)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    rule_evidence_weight = pm.LogNormal("rule_evidence_weight", mu=0.0, sigma=0.5)
    range_evidence_weight = pm.LogNormal("range_evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    # The mechanism: how strongly the teacher favours diagnostic examples.
    pedagogy = pm.Normal("pedagogy", mu=0.0, sigma=1.0)

    log_prior = pt.concatenate(
        [pt.log(p_mix) + pt.log(rule_weights), pt.ones(N_INT) * (pt.log1p(-p_mix) - np.log(N_INT))]
    )
    # Naive learner's posterior over concepts after one example x: (H, 100).
    naive = log_prior[:, None] + NAIVE_LOG_LIK
    log_naive_post = naive - pt.logsumexp(naive, axis=0, keepdims=True)
    # Teacher picks members of h in proportion to naive_post(h|x)^pedagogy.
    teach_logits = pedagogy * log_naive_post + NONMEMBER_MASK
    teach = pt.exp(teach_logits - pt.logsumexp(teach_logits, axis=1, keepdims=True)) * MEMBERSHIP
    log_teach = pt.log((1 - EPSILON) * teach + EPSILON / 100)  # (H, 100)

    ex_lik = log_teach.T[set_examples]  # (sets, MAX_SET, H)
    set_lik = (ex_lik * set_mask[:, :, None]).sum(axis=1)  # (sets, H)
    evidence_weight = rule_evidence_weight * IS_RULE + range_evidence_weight * (1 - IS_RULE)
    log_joint = log_prior[None, :] + evidence_weight[None, :] * set_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    concept_p = (posterior @ pt.as_tensor(MEMBERSHIP)).reshape((-1,))[row_cell]
    concept_p = pt.clip(concept_p, 1e-6, 1 - 1e-6)

    belief_logit = pt.log(concept_p) - pt.log1p(-concept_p)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias), 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    ex = np.zeros((len(unique), MAX_SET), dtype="int64")
    mask = np.zeros((len(unique), MAX_SET))
    for i, s in enumerate(unique):
        ex[i, : len(s)] = np.array(s) - 1
        mask[i, : len(s)] = 1.0
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    return {
        "set_examples": ex,
        "set_mask": mask,
        "row_cell": set_idx * 100 + tgt,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
