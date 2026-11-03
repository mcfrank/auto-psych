"""People do Bayesian concept learning over 45 arithmetic and digit rules and
every interval of 1..100. They use strong sampling without replacement
(each example is a distinct member, with 10% of examples random) and a
response lapse (on a fraction `lapse` of trials they guess "yes" with
probability `guess_yes`). Considering a rule costs effort that grows with
the length of its description, so a rule's prior is proportional to
exp(-cost * description_length). Single-primitive rules (odd, even,
primes, squares, cubes) have length 1. Parameterised families (multiples
of m, powers of m, ends in d, contains d) have length 2. Exception rules
(powers of 2 except 32, powers of 2 plus 37) have length 4.
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
    return list(rules.values())


RULES = rule_concepts()
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
CONCEPT_SIZE = MEMBERSHIP.sum(axis=1)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
# Description length of each rule, in the order rule_concepts() builds them.
DESCRIPTION_LENGTH = np.array([1, 1, 1, 1, 4, 4, 1] + [2] * 38, dtype="float64")
# Centre so the cost does not trade off against lambda_0's normalisation.
DESCRIPTION_LENGTH_C = DESCRIPTION_LENGTH - DESCRIPTION_LENGTH.mean()


def set_log_lik(examples):
    """log P(examples | h) for every hypothesis, sampling distinct members without replacement."""
    total = np.zeros(len(MEMBERSHIP))
    shown = np.zeros(len(MEMBERSHIP))
    seen = set()
    for x in examples:
        if x in seen:  # a repeated number can only be a stray draw
            total += np.log((1 - ALPHA) / 100)
            continue
        seen.add(x)
        in_h = MEMBERSHIP[:, x - 1]
        remaining = np.maximum(CONCEPT_SIZE - shown, 1.0)
        total += np.log(ALPHA * in_h / remaining + (1 - ALPHA) / 100)
        shown += in_h
    return total


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules): log P(D | r)
    interval_log_evidence = pm.Data("interval_log_evidence", np.zeros(1))  # (sets,)
    interval_in_concept = pm.Data("interval_in_concept", np.zeros((1, 100)))  # (sets, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    cost = pm.Normal("cost", 0.0, 1.0)  # effort per unit of description length
    log_rule_prior = -cost * DESCRIPTION_LENGTH_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)
    rule_joint = rule_log_lik + log_rule_prior[None, :]  # (sets, rules)
    rule_log_evidence = pt.logsumexp(rule_joint, axis=1)
    rule_post = pt.exp(rule_joint - rule_log_evidence[:, None])
    rule_in_concept = pt.dot(rule_post, RULE_MEMBERSHIP)  # (sets, targets)

    log_rules = pt.log(lambda_0) + rule_log_evidence
    log_intervals = pt.log1p(-lambda_0) - np.log(len(INTERVALS)) + interval_log_evidence
    p_rule = pm.math.sigmoid(log_rules - log_intervals)[:, None]
    in_concept = p_rule * rule_in_concept + (1 - p_rule) * interval_in_concept

    p_concept = pt.clip(in_concept.reshape((-1,))[row_cell], 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    all_log_lik = np.stack([set_log_lik(s) for s in unique])
    interval_log_lik = all_log_lik[:, N_RULES:]
    interval_log_evidence = logsumexp(interval_log_lik, axis=1)
    return {
        "rule_log_lik": all_log_lik[:, :N_RULES],
        "interval_log_evidence": interval_log_evidence,
        "interval_in_concept": np.exp(interval_log_lik - interval_log_evidence[:, None]) @ MEMBERSHIP[N_RULES:],
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
