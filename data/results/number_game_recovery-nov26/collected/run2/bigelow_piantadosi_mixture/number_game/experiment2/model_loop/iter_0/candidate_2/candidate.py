"""Position-graded trust in the examples: people do Bayesian concept learning
over the incumbent's 45 arithmetic and digit rules and every interval of
1..100 (equal rule prior, examples drawn without replacement from the
concept, occasional guessing at their own yes rate), but how likely they
think each example is a stray random number changes steadily with its place
in the sequence -- later examples are trusted more (recency) or less
(primacy) than earlier ones. The one new parameter `order_slope` shifts the
log-odds that an example is a stray by a fixed step per position, centred on
the sequence so the average stray rate stays near one in ten; at 0 the model
is the incumbent. The order effect is tabulated at Chebyshev nodes in
`order_slope` per example set, so the fit stays fast.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

BASE_LOGIT = np.log(0.1 / 0.9)  # log-odds of a stray at the middle position
SLOPE_RANGE = 3.0
N_NODES = 21
NODES = np.cos(np.pi * (np.arange(N_NODES) + 0.5) / N_NODES)  # in [-1, 1]
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
SIZE = MEMBERSHIP.sum(axis=1)


def set_log_lik(examples, slope):
    """log P(examples in order | h), without replacement, stray rate graded by position."""
    n = len(examples)
    total = np.zeros(len(MEMBERSHIP))
    shown = np.zeros(len(MEMBERSHIP))
    for i, x in enumerate(examples):
        stray = 1.0 / (1.0 + np.exp(-(BASE_LOGIT + slope * (i - (n - 1) / 2.0))))
        member = MEMBERSHIP[:, x - 1]
        remaining = np.maximum(SIZE - shown, 1.0)
        total += np.log((1 - stray) * member / remaining + stray / 100)
        shown += member
    return total


def chebyshev_basis(x, degree):
    terms = [pt.ones_like(x), x]
    for _ in range(degree - 1):
        terms.append(2 * x * terms[-1] - terms[-2])
    return pt.stack(terms[: degree + 1])


with pm.Model() as model:
    # Chebyshev coefficients (in order_slope) of each set's evidence and membership.
    rule_log_evidence = pm.Data("rule_log_evidence", np.zeros((N_NODES, 1)))  # (coef, sets)
    interval_log_evidence = pm.Data("interval_log_evidence", np.zeros((N_NODES, 1)))
    rule_in_concept = pm.Data("rule_in_concept", np.zeros((N_NODES, 1)))  # (coef, sets*100)
    interval_in_concept = pm.Data("interval_in_concept", np.zeros((N_NODES, 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    order_slope = pm.Normal("order_slope", 0.0, 1.0)  # >0: later examples more often strays
    basis = chebyshev_basis(pt.clip(order_slope / SLOPE_RANGE, -1.0, 1.0), N_NODES - 1)
    rule_ev = pt.dot(basis, rule_log_evidence)
    interval_ev = pt.dot(basis, interval_log_evidence)
    rule_mem = pt.dot(basis, rule_in_concept[:, row_cell])
    interval_mem = pt.dot(basis, interval_in_concept[:, row_cell])

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_rules = pt.log(lambda_0) - np.log(N_RULES) + rule_ev
    log_intervals = pt.log1p(-lambda_0) - np.log(len(INTERVALS)) + interval_ev
    p_rule = pm.math.sigmoid(log_rules - log_intervals)[row_set]
    p_concept = pt.clip(p_rule * rule_mem + (1 - p_rule) * interval_mem, 0.0, 1.0)

    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    tables = {k: [] for k in ["rule_log_evidence", "interval_log_evidence", "rule_in_concept", "interval_in_concept"]}
    for node in NODES:
        log_lik_all = np.stack([set_log_lik(s, SLOPE_RANGE * node) for s in unique])
        for name, part in [("rule", slice(0, N_RULES)), ("interval", slice(N_RULES, None))]:
            log_lik = log_lik_all[:, part]
            log_evidence = logsumexp(log_lik, axis=1)
            tables[f"{name}_log_evidence"].append(log_evidence)
            tables[f"{name}_in_concept"].append((np.exp(log_lik - log_evidence[:, None]) @ MEMBERSHIP[part]).ravel())
    out = {}
    for name, values in tables.items():
        values = np.stack(values)  # (nodes, ...)
        coef = np.polynomial.chebyshev.chebfit(NODES, values, N_NODES - 1)
        out[name] = np.ascontiguousarray(coef)
    out["row_cell"] = np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64")
    out["row_set"] = np.array([index[s] for s in sets], dtype="int64")
    out["rating"] = np.array([int(r["rating"]) for r in rows], dtype="int64")
    return out
