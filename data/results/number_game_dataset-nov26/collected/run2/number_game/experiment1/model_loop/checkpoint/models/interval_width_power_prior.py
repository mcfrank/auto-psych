"""Refinement of bigelow_piantadosi_independent: Bayesian concept learning over
the same 45 rules (each with its own prior weight, lambda_0 for the rules as a
group) and every interval of 1..100, with the size principle and 10% uniform
example noise; "yes" is the posterior probability that the concept contains
the target. The single change: an interval's prior is proportional to
width**beta (beta inferred, beta = 0 recovers the incumbent), normalised so the
intervals together still have prior mass 1 - lambda_0. This sets how readily
people extend a range far beyond the examples.
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
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
# log P(x | h) of one example x (column x - 1) under each hypothesis.
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]

# "ends in 0" and "contains digit 0" are one set, {10, 20, ..., 100}, so the
# data fix only the sum of their weights, and sampling both mixes slowly. The
# weights are sampled instead as one per distinct set, the shared set's
# concentration 4, and an independent Beta(2, 2) split of it: by Dirichlet
# aggregation, exactly the Dirichlet(2, ..., 2) prior on the 45 rule weights.
END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

# Intervals of equal width share prior and are grouped: for each set and width
# w, the log evidence summed over intervals of that width, and per target the
# share of that evidence from intervals containing it. Exact, 100 groups.
WIDTHS = np.array([len(c) for c in INTERVALS])
N_W = 100
LOG_W = np.log(np.arange(1, N_W + 1, dtype="float64"))
COUNT_W = np.array([(WIDTHS == w).sum() for w in range(1, N_W + 1)], dtype="float64")

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    width_log_evidence = pm.Data("width_log_evidence", np.zeros((1, N_W)))  # (sets, widths)
    width_in_concept = pm.Data("width_in_concept", np.zeros((1, N_W, 100)))  # (sets, widths, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    beta = pm.Normal("beta", 0.0, 1.5)  # width exponent of the interval prior
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    # log prior of ONE interval of width w; total interval mass is 1 - lambda_0.
    log_unnorm = beta * pt.as_tensor(LOG_W)
    log_z = pt.logsumexp(log_unnorm + np.log(COUNT_W))
    log_interval_prior = pt.log1p(-lambda_0) + log_unnorm - log_z  # (widths,)

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_widths = log_interval_prior[None, :] + width_log_evidence
    log_joint = pt.concatenate([log_rules, log_widths], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + (
        posterior[:, N_RULES:, None] * width_in_concept
    ).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_log_lik = set_log_lik[:, N_RULES:]
    interval_mem = MEMBERSHIP[N_RULES:]
    w_ev = np.zeros((len(unique), N_W))
    w_in = np.zeros((len(unique), N_W, 100))
    for w in range(1, N_W + 1):
        sel = WIDTHS == w
        ll = interval_log_lik[:, sel]
        ev = logsumexp(ll, axis=1)
        w_ev[:, w - 1] = ev
        w_in[:, w - 1, :] = np.exp(ll - ev[:, None]) @ interval_mem[sel]
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "width_log_evidence": w_ev,
        "width_in_concept": w_in,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
