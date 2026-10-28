"""Graded exemplar matching to stored number categories.

People store each familiar number category (42 mathematical rules and all
ranges of 1..100) as the collection of its member numbers, and judge both the
examples and the target by exemplar similarity to those stored members rather
than by all-or-none membership: a number's fit to a category is
exp(-d / w), where d is its distance on the number line to the category's
nearest stored member and w an inferred similarity width. A category is
retrieved in proportion to its prior times the product over examples of
(fit / number of stored members) (10% of examples treated as random), with
the match evidence weighted separately for rules and ranges; the target's
endorsement is its expected fit to the retrieved categories, turned into a
yes/no answer by a decision gain and criterion on the log-odds. As w -> 0 this
recovers crisp membership.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)


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
CONCEPTS = RULES + INTERVALS
SIZE = np.array([len(c) for c in CONCEPTS], dtype="float64")
# Distance from every number 1..100 to each category's nearest stored member.
NEAREST = np.array(
    [np.abs(NUMBERS[None, :] - np.array(c)[:, None]).min(axis=0) for c in CONCEPTS], dtype="float64"
)  # (concepts, 100)
LOG_N_INTERVALS = np.log(len(INTERVALS))

with pm.Model() as model:
    set_counts = pm.Data("set_counts", np.zeros((1, 100)))  # (sets, numbers): examples shown
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,): set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    rule_evidence_weight = pm.LogNormal("rule_evidence_weight", mu=0.0, sigma=0.5)
    range_evidence_weight = pm.LogNormal("range_evidence_weight", mu=0.0, sigma=0.5)
    # The single mechanism: width of exemplar similarity on the number line.
    similarity_width = pm.LogNormal("similarity_width", mu=np.log(0.5), sigma=0.7)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)

    fit = pt.exp(-pt.as_tensor(NEAREST) / similarity_width)  # (concepts, 100)
    log_match = pt.log((1 - EPSILON) * fit / SIZE[:, None] + EPSILON / 100)
    set_log_lik = set_counts @ log_match.T  # (sets, concepts)
    log_prior = pt.concatenate(
        [pt.log(p_mix) + pt.log(rule_weights), pt.ones(len(INTERVALS)) * (pt.log1p(-p_mix) - LOG_N_INTERVALS)]
    )
    evidence_weight = pt.concatenate(
        [pt.ones(N_RULES) * rule_evidence_weight, pt.ones(len(INTERVALS)) * range_evidence_weight]
    )
    log_joint = log_prior[None, :] + evidence_weight[None, :] * set_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    # Expected fit of every target to the retrieved categories, per set.
    set_target_p = (posterior @ fit).reshape((-1,))
    concept_p = pt.clip(set_target_p[row_cell], 1e-6, 1 - 1e-6)

    belief_logit = pt.log(concept_p) - pt.log1p(-concept_p)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias), 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    counts = np.zeros((len(unique), 100))
    for i, s in enumerate(unique):
        for x in s:
            counts[i, x - 1] += 1
    return {
        "set_counts": counts,
        "row_cell": np.array(
            [set_index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"
        ),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
