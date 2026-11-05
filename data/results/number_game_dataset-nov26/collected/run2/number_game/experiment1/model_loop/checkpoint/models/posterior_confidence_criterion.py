"""Posterior-confidence criterion on the simplest Bayesian concept learner.

People infer the program's concept exactly as the simplest Bayesian
rule-and-interval learner does (45 equally likely rules plus all intervals of
1-100, size principle, 10% noisy examples), but they do not answer "yes" with
their posterior probability that the target belongs (probability matching).
Instead they compare that confidence to an internal criterion and say "yes"
only when it is high enough, via a soft threshold of inferred steepness on the
log-odds of membership: p_yes = sigmoid(gain * (logit(belief) - criterion)).
gain = 1, criterion = 0 recovers probability matching; gain > 1 is a sharper
threshold; criterion shifts how sure one must be.
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


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
EPS = 1e-4

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    interval_log_evidence = pm.Data("interval_log_evidence", np.zeros(1))
    interval_in_concept = pm.Data("interval_in_concept", np.zeros((1, 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.75)
    criterion = pm.Normal("criterion", mu=0.0, sigma=1.5)
    gain = pt.exp(log_gain)

    log_rules = pt.log(lambda_0) - np.log(N_RULES) + rule_log_lik
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + interval_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + posterior[:, N_RULES][:, None] * interval_in_concept

    belief = pt.clip(in_concept.reshape((-1,))[row_cell], EPS, 1 - EPS)
    log_odds = pt.log(belief) - pt.log1p(-belief)
    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(gain * (log_odds - criterion)), 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_log_lik = set_log_lik[:, N_RULES:]
    log_evidence = logsumexp(interval_log_lik, axis=1)
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "interval_log_evidence": log_evidence,
        "interval_in_concept": np.exp(interval_log_lik - log_evidence[:, None]) @ MEMBERSHIP[N_RULES:],
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
