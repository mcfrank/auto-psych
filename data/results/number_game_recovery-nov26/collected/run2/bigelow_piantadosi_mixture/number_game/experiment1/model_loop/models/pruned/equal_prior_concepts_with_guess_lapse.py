"""Refinement of concept_learning_with_guess_lapse: Bayesian concept learning
over the same 45 rules and every interval of 1..100 (size principle, 10% of
examples drawn uniformly), plus a response lapse -- on a fraction `lapse` of
trials the participant ignores the concept and answers "yes" with probability
`guess_yes`. The one change: no learned preference among rules. All 45 rules
share the rule prior lambda_0 equally, instead of 45 separately inferred
weights; each interval keeps prior (1 - lambda_0) / 5050.
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
# log P(x | h) of one example x (column x - 1) under each hypothesis.
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]

# With equal priors the rules' and intervals' likelihoods are parameter-free,
# so prepare_observed collapses each class per example set: total evidence
# and, per target, the evidence-weighted share of hypotheses containing it.
# The posterior is then a two-way split between "a rule" and "an interval".

with pm.Model() as model:
    rule_log_evidence = pm.Data("rule_log_evidence", np.zeros(1))  # (sets,): log sum_r P(D | r)
    rule_in_concept = pm.Data("rule_in_concept", np.zeros((1, 100)))  # (sets, targets)
    interval_log_evidence = pm.Data("interval_log_evidence", np.zeros(1))  # (sets,)
    interval_in_concept = pm.Data("interval_in_concept", np.zeros((1, 100)))  # (sets, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_rules = pt.log(lambda_0) - np.log(N_RULES) + rule_log_evidence
    log_intervals = pt.log1p(-lambda_0) - np.log(len(INTERVALS)) + interval_log_evidence
    p_rule = pm.math.sigmoid(log_rules - log_intervals)[:, None]  # posterior that the concept is a rule
    in_concept = p_rule * rule_in_concept + (1 - p_rule) * interval_in_concept

    # Rounding can put a sure membership a few ulps above 1.
    p_concept = pt.clip(in_concept.reshape((-1,))[row_cell], 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)  # share of trials answered by guessing
    guess_yes = pm.Beta("guess_yes", 2, 2)  # P(yes) when guessing
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    out = {}
    for name, part in [("rule", slice(0, N_RULES)), ("interval", slice(N_RULES, None))]:
        log_lik = set_log_lik[:, part]
        log_evidence = logsumexp(log_lik, axis=1)
        out[f"{name}_log_evidence"] = log_evidence
        out[f"{name}_in_concept"] = np.exp(log_lik - log_evidence[:, None]) @ MEMBERSHIP[part]
    out["row_cell"] = np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64")
    out["rating"] = np.array([int(r["rating"]) for r in rows], dtype="int64")
    return out
