"""Concept-mediated exemplar similarity (summed similarity to stored examples).

People do not infer one concept that covers the whole set. They store each
shown example and judge the target by its summed (averaged) similarity to those
stored examples, as in an exemplar model. The similarity between an example and
the target is the probability that they share a number concept -- a rule
(arithmetic or digit, e.g. "powers of 2", "contains a 3") or a numeric range --
judged from that example alone with the size principle, so sharing a narrow
property counts for more than sharing a broad one. The averaged similarity is
mapped to a yes/no answer by a Luce-style response rule (gain and criterion).
Because similarity is summed across examples rather than combined by a single
concept, a target resembling only one example still draws "yes".
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
# log P(x | h) of a single stored example x (column x - 1) under each concept.
EX_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)


def _single_example_tables():
    """Per stored example x: rule-route and range-route similarity to every target."""
    out = []
    for lo, hi in [(0, N_RULES), (N_RULES, len(MEMBERSHIP))]:
        ll = EX_LOG_LIK[lo:hi].T  # (examples, concepts), equal prior within route
        ev = logsumexp(ll, axis=1) - np.log(hi - lo)
        post = np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
        out.append((post @ MEMBERSHIP[lo:hi], ev))  # (examples, targets), (examples,)
    return out


(RULE_SIM, RULE_EV), (RANGE_SIM, RANGE_EV) = _single_example_tables()
LOG_EV_RATIO = RULE_EV - RANGE_EV  # how much better rules than ranges explain each example

with pm.Model() as model:
    example_weights = pm.Data("example_weights", np.zeros((1, 100)))  # (sets, examples): 1/|set| per shown example
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Share of similarity carried by rules (vs numeric ranges) a priori.
    lambda_0 = pm.Beta("lambda_0", 2, 2)
    p_rule = pm.math.sigmoid(pt.log(lambda_0) - pt.log1p(-lambda_0) + LOG_EV_RATIO)[:, None]
    similarity = p_rule * RULE_SIM + (1 - p_rule) * RANGE_SIM  # (examples, targets)

    # Summed (averaged) similarity of each target to the stored examples.
    set_similarity = pt.dot(example_weights, similarity)  # (sets, targets)
    s = pt.clip(set_similarity[row_set, row_target], 1e-6, 1.0)

    # Luce response rule: yes odds = (s / criterion) ** gain.
    gain = pm.LogNormal("gain", 0.0, 0.5)
    log_criterion = pm.Normal("log_criterion", -1.0, 1.0)
    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(gain * (pt.log(s) - log_criterion)), 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    weights = np.zeros((len(unique), 100))
    for i, s in enumerate(unique):
        for x in s:
            weights[i, x - 1] += 1.0 / len(s)
    return {
        "example_weights": weights,
        "row_set": np.array([index[s] for s in sets], dtype="int64"),
        "row_target": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
