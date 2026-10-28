"""One-sided magnitude threshold concepts: people do Bayesian concept learning
over 42 arithmetic rules (primes excluding 1, kind-level prior) with the size
principle (10% random examples, "yes" = posterior probability the concept
contains the target), but their only magnitude concepts are one-sided
thresholds - "numbers up to N" or "numbers N and above" - never two-ended
ranges. Round decade thresholds get an inferred prior boost, and an inferred
share splits belief between upper bounds and lower bounds. So size-based "yes"
extends without limit on one side of the examples.
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
    rules.append([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    rules.append([i for i in NUMBERS if i % 2 == 0])
    rules.append([i for i in NUMBERS if i % 2 == 1])
    rules.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    rules.append([1, 8, 27, 64])
    kinds = [0] * 9 + [1] * 10 + [2] * 9 + [3] * 9 + [4] * 5
    return rules, kinds


RULES, RULE_KINDS = concept_sets()
N_RULES = len(RULES)
N_KINDS = 5
KIND_MATRIX = np.eye(N_KINDS)[RULE_KINDS]
KIND_COUNTS = KIND_MATRIX.sum(axis=0)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
# Thresholds N = 1..99: "up to N" (1..N) and "N+1 and above" (N+1..100); the
# whole scale is left to the rules (even + odd etc.) and excluded here.
THRESH = np.arange(1, 100)
UP_TO = (NUMBERS[None, :] <= THRESH[:, None]).astype("float64")
AT_LEAST = (NUMBERS[None, :] > THRESH[:, None]).astype("float64")
ROUND = (THRESH % 10 == 0).astype("float64")
N_T = len(THRESH)


def log_lik(member, examples):
    size = member.sum(1)
    per = np.log(member[:, examples - 1] * (1 - EPSILON) / np.maximum(size, 1.0)[:, None] + EPSILON / 100)
    return per.sum(1)


with pm.Model() as model:
    rule_ll = pm.Data("rule_ll", np.zeros((1, N_RULES)))
    upto_ll = pm.Data("upto_ll", np.zeros((1, N_T)))
    atleast_ll = pm.Data("atleast_ll", np.zeros((1, N_T)))
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))
    pair_in_hypothesis = pm.Data("pair_in_hypothesis", np.zeros((1, N_RULES + 2 * N_T)))
    row_pair = pm.Data("row_pair", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    rule_share = pm.Beta("rule_share", alpha=2.0, beta=2.0)
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    rule_weights = pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS)
    upper_share = pm.Beta("upper_share", alpha=2.0, beta=2.0)
    round_boost = pm.Normal("round_boost", mu=1.0, sigma=1.5)

    log_t = round_boost * ROUND
    log_t = log_t - pt.logsumexp(log_t)
    log_rules = pt.log(rule_share) + pt.log(rule_weights)[None, :] + rule_ll
    log_upto = pt.log1p(-rule_share) + pt.log(upper_share) + log_t[None, :] + upto_ll
    log_atleast = pt.log1p(-rule_share) + pt.log1p(-upper_share) + log_t[None, :] + atleast_ll
    log_joint = pt.concatenate([log_rules, log_upto, log_atleast], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[pair_set] * pair_in_hypothesis).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_pair], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    pairs = list(dict.fromkeys(zip(sets, targets)))
    pair_index = {c: i for i, c in enumerate(pairs)}
    ex = {s: np.array(s) for s in unique}
    return {
        "rule_ll": np.stack([log_lik(RULE_MEMBERSHIP, ex[s]) for s in unique]),
        "upto_ll": np.stack([log_lik(UP_TO, ex[s]) for s in unique]),
        "atleast_ll": np.stack([log_lik(AT_LEAST, ex[s]) for s in unique]),
        "pair_set": np.array([index[s] for s, _ in pairs], dtype="int64"),
        "pair_in_hypothesis": np.stack(
            [np.concatenate([RULE_MEMBERSHIP[:, t - 1], UP_TO[:, t - 1], AT_LEAST[:, t - 1]]) for _, t in pairs]),
        "row_pair": np.array([pair_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
