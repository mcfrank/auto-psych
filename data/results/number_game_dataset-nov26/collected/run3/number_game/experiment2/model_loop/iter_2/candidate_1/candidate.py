"""Pedagogical sampling concept learning.

Hypothesis: people assume the examples were chosen by a helpful teacher, not
sampled at random: the program picked numbers that would best lead a learner to
its concept. So an example counts as evidence for a concept (one of 45
mathematical/digit rules with its own prior weight, or a range of 1-100 whose
prior grows as a power of its size) in proportion to how strongly that example
alone would make a naive (random-sampling) learner believe the concept, relative
to the other numbers 1-100 the teacher could have shown: P_teach(x | h) is
proportional to P_naive(h | x) ** teach. People answer "yes" with their
posterior belief that the concept contains the target.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
ALPHA_NOISE = 0.9


def _rules():
    n = NUMBERS
    r = [n % 2 == 1, n % 2 == 0,
         np.array([k > 1 and all(k % a for a in range(2, int(k ** 0.5) + 1)) for k in n]),
         np.isin(n, [k * k for k in range(1, 11)]), np.isin(n, [1, 8, 27, 64]),
         np.isin(n, [2, 4, 8, 16, 64]), np.isin(n, [2, 4, 8, 16, 32, 37, 64])]
    r += [n % m == 0 for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    r += [np.isin(n, [m ** k for k in range(1, 8)]) for m in range(2, 11)]
    r += [n % 10 == d for d in range(10)]
    r += [np.array([str(d) in str(k) for k in n]) for d in range(10)]
    return np.array(r, dtype="float64")


RULES = _rules()
N_RULES = RULES.shape[0]
INTERVALS = np.array([(NUMBERS >= a) & (NUMBERS <= b) for a in NUMBERS for b in range(a, 101)], dtype="float64")
MEMBERSHIP = np.concatenate([RULES, INTERVALS])
SIZE = MEMBERSHIP.sum(axis=1)
LOG_INT_SIZE = np.log(SIZE[N_RULES:])
# Literal (random-sampling) likelihood of one example x under concept h, with 10% strays.
LIT_LL = np.log(ALPHA_NOISE * MEMBERSHIP / SIZE[:, None] + (1 - ALPHA_NOISE) / 100)


def prepare_observed(rows):
    sets = [str(r["number_set"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    counts = np.zeros((len(unique), 100))
    for s, i in index.items():
        for x in s.split(","):
            counts[i, int(x) - 1] += 1
    return {
        "example_counts": counts,
        "row_set": np.array([index[s] for s in sets], dtype="int64"),
        "row_target": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


def log_softmax(x, axis):
    return x - pt.logsumexp(x, axis=axis, keepdims=True)


with pm.Model() as model:
    example_counts = pm.Data("example_counts", np.zeros((1, 100), dtype="float64"))
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))

    # Prior: share of rules, each rule's weight, and the range-size power.
    lambda_0 = pm.Beta("lambda_0", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    size_power = pm.Normal("size_power", 0.0, 1.0)
    # How sharply the teacher prefers the most informative example.
    log_teach = pm.Normal("log_teach", 0.0, 1.0)
    teach = pt.exp(log_teach)

    log_int = size_power * LOG_INT_SIZE
    log_int = log_int - pt.logsumexp(log_int) + pt.log(1 - lambda_0)
    log_prior = pt.concatenate([pt.log(lambda_0) + pt.log(rule_weights), log_int])
    # Naive learner's belief in each concept after one example x: (concepts, 100).
    lit_post = log_softmax(log_prior[:, None] + LIT_LL, axis=0)
    # Teacher's choice of example given the concept, normalised over the 100 numbers.
    ex_ll = log_softmax(teach * lit_post, axis=1)
    log_post = log_softmax(log_prior[None, :] + pt.dot(example_counts, ex_ll.T), axis=1)
    in_concept = pt.dot(pt.exp(log_post), MEMBERSHIP)
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_set, row_target], 1e-6, 1 - 1e-6))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
