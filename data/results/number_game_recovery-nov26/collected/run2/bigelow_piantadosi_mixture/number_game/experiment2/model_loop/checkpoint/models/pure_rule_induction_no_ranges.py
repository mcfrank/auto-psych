"""Pure rule induction, no numeric ranges: people weigh only the 45 arithmetic
and digit rules (equal prior) by the size principle with sampling without
replacement (10% of examples random), plus one catch-all concept "any number
1..100"; on a fraction `lapse` of trials they guess "yes" at rate `guess_yes`.
No interval concepts, so no magnitude-graded generalization.
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
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
SIZE = MEMBERSHIP.sum(axis=1)


def set_log_lik(examples):
    """log P(examples in order | rule), members sampled without replacement."""
    total = np.zeros(N_RULES)
    shown = np.zeros(N_RULES)
    for x in examples:
        member = MEMBERSHIP[:, x - 1]
        remaining = np.maximum(SIZE - shown, 1.0)
        total += np.log(ALPHA * member / remaining + (1 - ALPHA) / 100)
        shown += member
    return total


def catch_all_log_lik(examples):
    """log P(examples | any number 1..100), distinct draws."""
    return float(sum(-np.log(100 - i) for i in range(len(examples))))


with pm.Model() as model:
    rule_log_evidence = pm.Data("rule_log_evidence", np.zeros(1))  # (sets,): log mean_r P(D | r)
    rule_in_concept = pm.Data("rule_in_concept", np.zeros((1, 100)))  # (sets, targets)
    any_log_evidence = pm.Data("any_log_evidence", np.zeros(1))  # (sets,): log P(D | any number)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)  # prior share of the rules vs the catch-all
    log_odds = pt.log(lambda_0) - pt.log1p(-lambda_0) + rule_log_evidence - any_log_evidence
    p_rule = pm.math.sigmoid(log_odds)[:, None]
    in_concept = p_rule * rule_in_concept + (1 - p_rule)

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
    log_lik = np.stack([set_log_lik(s) for s in unique])
    log_evidence = logsumexp(log_lik, axis=1)
    return {
        "rule_log_evidence": log_evidence - np.log(N_RULES),
        "rule_in_concept": np.exp(log_lik - log_evidence[:, None]) @ MEMBERSHIP,
        "any_log_evidence": np.array([catch_all_log_lik(s) for s in unique]),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
