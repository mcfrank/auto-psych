"""Refinement of bigelow_piantadosi_independent with an Erlang prior on interval size.

People do Bayesian concept learning over 45 rules (odd, even, primes, squares,
cubes, powers of 2-10, powers of 2 without 32 or with 37, multiples of 3-9, 11
and 12, ending in 0-9, containing the digit 0-9) and every interval [m, n] of
1..100, with a prior weight per rule and share lambda_0 for the rules. Each
example comes from the concept (uniformly: the size principle) with probability
0.9 and from 1..100 otherwise. The one change from the incumbent: instead of
(1 - lambda_0) / 100^2 per interval (a flat prior over endpoint pairs), the
interval share is spread over widths L by an Erlang prior L / sigma^2 exp(-L / sigma)
(normalised over L = 1..100), with sigma inferred, and evenly over the 101 - L
intervals of each width. "Yes" is the posterior probability that the concept
contains the target.

Because the prior depends on sigma, intervals are pooled per set by width:
the summed likelihood of all intervals of each width and, per target, the
share of that likelihood from intervals containing it.
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
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)

M_ARR = np.array([m for m in NUMBERS for n in range(m, 101)])
N_ARR = np.array([n for m in NUMBERS for n in range(m, 101)])
WIDTH = N_ARR - M_ARR + 1
INTERVAL_MEMBERSHIP = ((M_ARR[:, None] <= NUMBERS[None, :]) & (NUMBERS[None, :] <= N_ARR[:, None])).astype("float64")
INTERVAL_LOG_LIK = np.log(ALPHA * INTERVAL_MEMBERSHIP / WIDTH[:, None] + (1 - ALPHA) / 100)
WIDTHS = np.arange(1, 101, dtype="float64")
LOG_N_OF_WIDTH = np.log(101 - WIDTHS)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    width_log_evidence = pm.Data("width_log_evidence", np.zeros((1, 100)))  # (sets, widths): log sum P(D | interval)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # distinct (set, target) cells
    cell_in_concept = pm.Data("cell_in_concept", np.zeros((1, N_RULES + 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    sigma = pm.LogNormal("sigma", mu=np.log(15.0), sigma=1.0)  # typical range width scale
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    log_erlang = np.log(WIDTHS) - WIDTHS / sigma
    log_width_prior = log_erlang - pt.logsumexp(log_erlang)
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_widths = pt.log1p(-lambda_0) + (log_width_prior - LOG_N_OF_WIDTH)[None, :] + width_log_evidence
    log_joint = pt.concatenate([log_rules, log_widths], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[cell_set] * cell_in_concept).sum(axis=1)[row_cell]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    rule_ll, width_ev, width_in = [], [], []
    for s in unique:
        x = np.array(s) - 1
        rule_ll.append(RULE_LOG_LIK[:, x].sum(axis=1))
        ll = INTERVAL_LOG_LIK[:, x].sum(axis=1)
        ev = np.array([logsumexp(ll[WIDTH == L]) for L in range(1, 101)])
        w = np.exp(ll - ev[WIDTH - 1])
        member = np.zeros((100, 100))
        np.add.at(member, WIDTH - 1, w[:, None] * INTERVAL_MEMBERSHIP)
        width_ev.append(ev)
        width_in.append(member)
    cells = list(dict.fromkeys((index[s], int(r["target"])) for s, r in zip(sets, rows)))
    cell_index = {c: i for i, c in enumerate(cells)}
    return {
        "rule_log_lik": np.array(rule_ll),
        "width_log_evidence": np.array(width_ev),
        "cell_set": np.array([c[0] for c in cells], dtype="int64"),
        "cell_in_concept": np.array([
            np.concatenate([RULE_MEMBERSHIP[:, t - 1], width_in[i][:, t - 1]]) for i, t in cells
        ]),
        "row_cell": np.array([cell_index[(index[s], int(r["target"]))] for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
