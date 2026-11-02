"""Bayesian concept learning in which the program may also be a "digit
vocabulary" concept: the numbers written using only a particular set of
digits (e.g. only the digits 2, 3, 6 and 7, giving 2, 3, 6, 7, 22, 23, ...,
76, 77).

The learner weighs the 45 natural rules (family-weighted prior, exception
rules in their own family), all intervals of 1..100, and one concept for every
non-empty set of digits, each family with its own inferred prior mass shared
equally by its members. Examples are sampled from the concept (size principle,
10% uniformly from 1..100) and "yes" is the posterior probability that the
concept contains the target. Because a small digit vocabulary that covers all
the examples is a suspicious coincidence, people should endorse targets that
recombine the examples' digits (76 after 27, 37, 67) more than neighbouring
targets that introduce a new digit.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


def rule_concepts():
    rules = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
        "powers of 2, not 32": [2, 4, 8, 16, 64],
        "powers of 2, or 37": [2, 4, 8, 16, 32, 37, 64],
        "even": list(range(2, 101, 2)),
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(10)})
    return rules


def family(name):
    if name in ("odd", "even"):
        return 0
    if name.startswith("multiples"):
        return 2
    if name.startswith("ends"):
        return 3
    if name.startswith("contains"):
        return 4
    if name.startswith("powers of 2,"):
        return 5  # gerrymandered exception rules
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in RULE_NAMES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
# Digit-vocabulary concepts: numbers whose every digit lies in a digit set D.
_DIGITS = [set(str(n)) for n in NUMBERS]
VOCAB = [[n for n, ds in zip(NUMBERS, _DIGITS) if ds <= {str(d) for d in range(10) if mask >> d & 1}] for mask in range(1, 1024)]
VOCAB = [c for c in VOCAB if c]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS + VOCAB], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
N_INTERVALS = len(INTERVALS)
N_VOCAB = len(VOCAB)
VOCAB_START = N_RULES + N_INTERVALS

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    interval_log_evidence = pm.Data("interval_log_evidence", np.zeros(1))  # log sum_j P(D | interval j)
    interval_in_concept = pm.Data("interval_in_concept", np.zeros((1, 100)))
    vocab_log_evidence = pm.Data("vocab_log_evidence", np.zeros(1))  # log sum_j P(D | digit set j)
    vocab_in_concept = pm.Data("vocab_in_concept", np.zeros((1, 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the six rule families (incl. exception rules), of intervals
    # and of digit-vocabulary concepts.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(8))
    log_w = pt.log(family_weight)

    log_rules = (log_w[:6] - np.log(FAMILY_SIZE))[RULE_FAMILY][None, :] + rule_log_lik
    log_intervals = log_w[6] - np.log(N_INTERVALS) + interval_log_evidence
    log_vocab = log_w[7] - np.log(N_VOCAB) + vocab_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None], log_vocab[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (
        posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)
        + posterior[:, N_RULES][:, None] * interval_in_concept
        + posterior[:, N_RULES + 1][:, None] * vocab_in_concept
    )

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_log_lik = set_log_lik[:, N_RULES:VOCAB_START]
    log_evidence = logsumexp(interval_log_lik, axis=1)
    vocab_log_lik = set_log_lik[:, VOCAB_START:]
    vocab_evidence = logsumexp(vocab_log_lik, axis=1)
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "interval_log_evidence": log_evidence,
        "interval_in_concept": np.exp(interval_log_lik - log_evidence[:, None]) @ MEMBERSHIP[N_RULES:VOCAB_START],
        "vocab_log_evidence": vocab_evidence,
        "vocab_in_concept": np.exp(vocab_log_lik - vocab_evidence[:, None]) @ MEMBERSHIP[VOCAB_START:],
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
