"""Refinement of bigelow_piantadosi_independent: Bayesian concept learning
over the same 45 rules (a prior weight inferred for each, a share lambda_0 for
the rules) and the 5,050 intervals of 1..100, with examples sampled from the
concept (size principle, 10% uniform noise). The one change: an interval's
prior depends on its width L, proportional to L^beta. beta = 0 is the
incumbent's uniform prior (1 - lambda_0) / 100^2 per interval; beta < 0 makes
narrow magnitude ranges a priori more plausible. "Yes" is the posterior
probability that the concept contains the target.
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
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_SIZE = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")  # 1..100
_IV_ROWS, _IV_COLS = np.nonzero(INTERVAL_MEMBERSHIP)
WIDTHS = np.arange(1, 101, dtype="float64")
LOG_WIDTHS = np.log(WIDTHS)
LOG_N_OF_WIDTH = np.log(101.0 - WIDTHS)  # number of intervals of each width

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

# Intervals of one width share a prior, so per example set they are grouped
# exactly by width: log of their summed likelihood, and for each asked target
# the share of that likelihood from intervals containing it.

with pm.Model() as model:
    set_log_lik = pm.Data("set_log_lik", np.zeros((1, N_RULES + 100)))  # (sets, rules + widths)
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_member = pm.Data("row_member", np.zeros((1, N_RULES + 100)))  # (rows, rules + widths)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    # Width exponent of the interval prior: 0 = the incumbent's uniform prior.
    beta = pm.Normal("beta", 0.0, 1.0)

    # Per-interval log prior, scaled so that beta = 0 gives (1 - lambda_0) / 100^2.
    log_unnorm = beta * LOG_WIDTHS
    log_z = pt.logsumexp(LOG_N_OF_WIDTH + log_unnorm)
    log_interval_prior = pt.log1p(-lambda_0) + np.log(0.505) + log_unnorm - log_z
    log_prior = pt.concatenate([pt.log(lambda_0) + pt.log(rule_weights), log_interval_prior])
    log_joint = log_prior[None, :] + set_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, rules + widths)
    in_concept = (posterior[row_set] * row_member).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_ll, width_share = [], []
    for s in unique:
        ll = EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1)
        iv = ll[N_RULES:]
        width_ll = np.array([logsumexp(iv[INTERVAL_SIZE == w]) for w in range(1, 101)])
        weight = np.exp(iv - width_ll[INTERVAL_SIZE - 1])
        share = np.bincount((INTERVAL_SIZE[_IV_ROWS] - 1) * 100 + _IV_COLS, weights=weight[_IV_ROWS],
                            minlength=100 * 100).reshape(100, 100)  # (widths, targets)
        set_ll.append(np.concatenate([ll[:N_RULES], width_ll]))
        width_share.append(share)
    row_set = np.array([index[s] for s in sets], dtype="int64")
    targets = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    width_share = np.stack(width_share)
    row_member = np.concatenate([RULE_MEMBERSHIP[:, targets].T, width_share[row_set, :, targets]], axis=1)
    return {
        "set_log_lik": np.stack(set_ll),
        "row_set": row_set,
        "row_member": np.clip(row_member, 0.0, 1.0),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
