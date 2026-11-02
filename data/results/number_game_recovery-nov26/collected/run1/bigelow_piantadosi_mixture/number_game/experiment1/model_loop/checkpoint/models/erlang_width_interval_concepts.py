"""Refinement of bayesian_concepts_logodds_readout: Bayesian concept learning
over 45 rules and every interval of 1..100 (size principle, 10% of examples
drawn uniformly), rule weights inferred, rules as a group lambda_0, and a
scaled/shifted log-odds decision readout -- with one change: the interval prior
is no longer flat but follows an Erlang density on interval width,
w(s) proportional to s * exp(-s / sigma), with the typical width sigma fitted.
People thus expect the program's numeric range to have a characteristic
breadth, governing how far they generalise beyond the examples' span.

The total prior mass on intervals is kept at the original's
(1 - lambda_0) * 5050 / 100^2; only its distribution over widths changes.
Intervals of equal width share a prior, so per example set they are grouped by
width (evidence and in-concept share per width precomputed in prepare_observed).
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


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
INTERVAL_WIDTH = np.array([len(c) for c in INTERVALS])  # 1..100
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
WIDTHS = np.arange(1, 101, dtype="float64")
WIDTH_COUNT = 101.0 - WIDTHS  # number of intervals of each width
INTERVAL_MASS = np.log(5050.0 / 100**2)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    width_log_evidence = pm.Data("width_log_evidence", np.zeros((1, 100)))  # (sets, widths)
    width_in_concept = pm.Data("width_in_concept", np.zeros((1, 100, 100)))  # (sets, widths, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    # Erlang(shape 2) prior on interval width with fitted typical width sigma.
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    log_w = np.log(WIDTHS) - WIDTHS / sigma
    log_per_interval = log_w - pt.logsumexp(log_w + np.log(WIDTH_COUNT))  # prior share of one interval

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_widths = pt.log1p(-lambda_0) + INTERVAL_MASS + log_per_interval[None, :] + width_log_evidence
    log_joint = pt.concatenate([log_rules, log_widths], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, rules + 100)
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + pt.sum(
        posterior[:, N_RULES:, None] * width_in_concept, axis=1)

    q = pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6)
    gain = pm.LogNormal("gain", mu=0.0, sigma=0.5)
    criterion = pm.Normal("criterion", mu=0.0, sigma=1.5)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(gain * (pt.log(q) - pt.log1p(-q)) - criterion))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n = len(unique)
    rule_ll = np.zeros((n, N_RULES))
    width_ev = np.zeros((n, 100))
    width_in = np.zeros((n, 100, 100))
    interval_member = MEMBERSHIP[N_RULES:]
    for i, s in enumerate(unique):
        ll = EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1)
        rule_ll[i] = ll[:N_RULES]
        ill = ll[N_RULES:]
        for w in range(1, 101):
            sel = INTERVAL_WIDTH == w
            ev = logsumexp(ill[sel])
            width_ev[i, w - 1] = ev
            width_in[i, w - 1] = np.exp(ill[sel] - ev) @ interval_member[sel]
    return {
        "rule_log_lik": rule_ll,
        "width_log_evidence": width_ev,
        "width_in_concept": width_in,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
