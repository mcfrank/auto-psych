"""Recency-decayed Bayesian concept learning.

People infer the hidden program by Bayesian concept learning over 42
mathematical rules and the 5,050 intervals of 1..100, but their memory of the
examples decays with recency: the last-shown example contributes its full
log-likelihood while an example shown k positions earlier is weighted by
decay**k (decay inferred, in (0, 1)). Concepts consistent with the later
examples therefore dominate the posterior, producing a recency effect; "yes"
is the posterior probability that the concept contains the target. Rule prior:
p_mix on the rules shared by Dirichlet weights, 1 - p_mix uniformly on the
intervals; 10% of examples are noise drawn uniformly from 1..100.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_LAG = 3  # sets hold up to four examples
SAMPLER_SETTINGS = {"target_accept": 0.9}


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
N_RULES, N_INTERVALS = len(RULES), len(INTERVALS)
N_HYP = N_RULES + N_INTERVALS
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
# log P(x | h) of one example x (column x - 1) under each hypothesis.
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)


def prepare_observed(rows):
    """Per distinct set: the last example's log-likelihood under every concept
    (weight 1), and that of the example k = 1..3 positions earlier (weight
    decay**k; zero where the set is shorter)."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    last = np.zeros((len(unique), N_HYP))
    earlier = np.zeros((MAX_LAG, len(unique), N_HYP))
    for i, s in enumerate(unique):
        for j, x in enumerate(s):
            lag = len(s) - 1 - j
            if lag == 0:
                last[i] = EXAMPLE_LOG_LIK[:, x - 1]
            else:
                earlier[lag - 1, i] = EXAMPLE_LOG_LIK[:, x - 1]
    return {
        "last_log_lik": last,
        "earlier_log_lik": earlier.reshape(MAX_LAG, -1),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    last_log_lik = pm.Data("last_log_lik", np.zeros((1, N_HYP)))  # (sets, hypotheses)
    earlier_log_lik = pm.Data("earlier_log_lik", np.zeros((MAX_LAG, N_HYP)))  # (lags, sets * hypotheses)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    decay = pm.Beta("decay", 2, 2)

    # Evidence: the last example at full weight, earlier ones at decay**lag.
    decay_pow = pt.exp(pt.arange(1, MAX_LAG + 1) * pt.log(decay))
    set_log_lik = last_log_lik + pt.dot(decay_pow, earlier_log_lik).reshape(last_log_lik.shape)

    log_prior = pt.concatenate([
        pt.log(p_mix) + pt.log(rule_weights),
        pt.log1p(-p_mix) - np.log(N_INTERVALS) + pt.zeros(N_INTERVALS),
    ])
    posterior = pt.special.softmax(log_prior[None, :] + set_log_lik, axis=1)
    in_concept = pt.dot(posterior, pt.as_tensor(MEMBERSHIP))  # (sets, 100)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
