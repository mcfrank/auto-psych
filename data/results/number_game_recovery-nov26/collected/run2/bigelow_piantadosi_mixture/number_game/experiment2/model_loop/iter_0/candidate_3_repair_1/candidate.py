"""Refinement of without_replacement_size_principle: Bayesian concept learning
over the same 45 rules and every interval of 1..100, with examples drawn without
replacement from the concept's not-yet-shown members (10% uniform strays) and a
guessing lapse. The one change: the ten final-digit rules ("ends in d") carry a
separately inferred prior weight relative to the other rules (log-weight
`ends_in_weight`; negative = a shared last digit is discounted as coincidence),
while every other rule keeps an equal prior and the rules as a group keep their
share lambda_0 against the ranges.
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
    return list(rules.items())


RULE_ITEMS = rule_concepts()
N_RULES = len(RULE_ITEMS)
IS_ENDS_IN = np.array([name.startswith("ends in") for name, _ in RULE_ITEMS], dtype="float64")
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for _, c in RULE_ITEMS] + [np.isin(NUMBERS, c) for c in INTERVALS], dtype="float64")
SIZE = MEMBERSHIP.sum(axis=1)


def set_log_lik(examples):
    """log P(examples in order | h) for every hypothesis, sampling members without replacement."""
    total = np.zeros(len(MEMBERSHIP))
    shown = np.zeros(len(MEMBERSHIP))
    for x in examples:
        member = MEMBERSHIP[:, x - 1]
        remaining = np.maximum(SIZE - shown, 1.0)
        total += np.log(ALPHA * member / remaining + (1 - ALPHA) / 100)
        shown += member
    return total


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    interval_log_evidence = pm.Data("interval_log_evidence", np.zeros(1))  # (sets,): log mean over intervals
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    rule_member = pm.Data("rule_member", np.zeros((1, N_RULES)))  # (cells, rules): target in rule
    interval_in_concept = pm.Data("interval_in_concept", np.zeros(1))  # (cells,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (rows,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    ends_in_weight = pm.Normal("ends_in_weight", 0.0, 1.0)
    log_rule_prior = ends_in_weight * IS_ENDS_IN
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)
    weighted = rule_log_lik + log_rule_prior[None, :]
    rule_log_evidence = pt.logsumexp(weighted, axis=1)
    rule_post = pt.special.softmax(weighted, axis=1)
    p_rule = pm.math.sigmoid(pt.log(lambda_0) + rule_log_evidence - pt.log1p(-lambda_0) - interval_log_evidence)

    rule_in_concept = (rule_post[cell_set] * rule_member).sum(axis=1)
    p_cell = p_rule[cell_set]
    p_concept = pt.clip(p_cell * rule_in_concept + (1 - p_cell) * interval_in_concept, 0.0, 1.0)[row_cell]

    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n = len(unique)
    rule_ll = np.zeros((n, N_RULES))
    int_ev = np.zeros(n)
    int_in = np.zeros((n, 100))
    for i, s in enumerate(unique):
        ll = set_log_lik(s)
        rule_ll[i] = ll[:N_RULES]
        li = ll[N_RULES:]
        int_ev[i] = logsumexp(li) - np.log(len(li))
        int_in[i] = np.exp(li - logsumexp(li)) @ MEMBERSHIP[N_RULES:]
    keys = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {k: i for i, k in enumerate(cells)}
    cs = np.array([k[0] for k in cells], dtype="int64")
    ct = np.array([k[1] for k in cells], dtype="int64")
    return {
        "rule_log_lik": rule_ll,
        "interval_log_evidence": int_ev,
        "cell_set": cs,
        "rule_member": MEMBERSHIP[:N_RULES, ct].T.copy(),
        "interval_in_concept": int_in[cs, ct],
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
