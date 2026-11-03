"""Graded final digits on the equal-prior familiar rules.

Refinement of `equal_prior_familiar_rules_digit_sums`, adding one component from
`graded_final_digit_digit_sum_concepts`: people do the same Bayesian concept
learning over equally plausible familiar number rules, digit-sum rules (prior
share proportional to exp(-dsum_decay * s)) and the 5,050 ranges of 1..100
(Erlang width prior, outputs clustered toward the range's middle, the size
principle, 10% of examples uniform noise). The one change is that the ten
"ends in d" rules are not equally natural to each other. Together they keep
their equal share of the familiar-rule mass, but within it a rule's prior is
proportional to exp(-digit_decay * d), so how readily people generalise along a
shared final digit depends on which digit it is.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from numpy.polynomial import chebyshev
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


def familiar_rules():
    rules = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "even": list(range(2, 101, 2)),
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    return list(rules.values())


DSUM_RULES = [[n for n in NUMBERS if sum(map(int, str(n))) == t] for t in range(1, 19)]
RULES = familiar_rules() + DSUM_RULES
N_FAMILIAR = len(RULES) - 18
N_RULES = len(RULES)
DSUM_VALUE = np.arange(1, 19, dtype="float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
WIDTH = HI - LO + 1
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
WIDTHS = np.arange(1, 101, dtype="float64")
N_PER_WIDTH = 101.0 - WIDTHS
CHEB_DEGREE = 6
# log(u * (1 - u)) at the k-th of W positions of a width-W range, u = (k - 0.5) / W.
_K = np.arange(1, 101)[None, :]
_U = np.clip((_K - 0.5) / WIDTHS[:, None], 1e-9, 1 - 1e-9)
SHAPE_TABLE = np.where(_K <= WIDTHS[:, None], np.log(_U * (1 - _U)), 0.0)
SHAPE_MASK = np.where(_K <= WIDTHS[:, None], 0.0, -1e4)


def example_log_lik_table(kappa):
    """Log P(example at position k of a width-W range), (W, k), for centre weight kappa."""
    log_density = kappa * SHAPE_TABLE - logsumexp(kappa * SHAPE_TABLE + SHAPE_MASK, axis=1, keepdims=True)
    return np.log(ALPHA * np.exp(log_density) + (1 - ALPHA) / 100)


# Interval log likelihoods are smooth in kappa: Chebyshev expansion in
# 2 * kappa - 1 on [0, 1]; the last row is "outside the range".
_NODES = np.cos(np.pi * (np.arange(30) + 0.5) / 30)
_TABLES = np.stack([example_log_lik_table((x + 1) / 2).reshape(-1) for x in _NODES])
CHEB_TABLE = np.concatenate(
    [
        chebyshev.chebfit(_NODES, _TABLES, CHEB_DEGREE).T,
        np.eye(1, CHEB_DEGREE + 1) * np.log((1 - ALPHA) / 100),
    ]
)
OUTSIDE = 100 * 100

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    interval_cheb = pm.Data("interval_cheb", np.zeros((1, len(LO), CHEB_DEGREE + 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of familiar rules (all equally plausible), digit sums, ranges.
    kind_weight = pm.Dirichlet("kind_weight", a=np.ones(3))
    log_w = pt.log(kind_weight)
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    centre_weight = pm.Beta("centre_weight", alpha=1.5, beta=6.0)
    dsum_decay = pm.Normal("dsum_decay", mu=0.0, sigma=1.0)

    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    log_per_interval = (erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH)))[WIDTH - 1]

    x = 2.0 * centre_weight - 1.0
    cheb = [pt.ones(()), x]
    for _ in range(CHEB_DEGREE - 1):
        cheb.append(2.0 * x * cheb[-1] - cheb[-2])
    interval_log_lik = pt.tensordot(interval_cheb, pt.stack(cheb), axes=[[2], [0]])  # (sets, intervals)

    dsum_log_share = -dsum_decay * DSUM_VALUE
    dsum_log_share = dsum_log_share - pt.logsumexp(dsum_log_share)
    digit_decay = pm.Normal("digit_decay", mu=0.0, sigma=0.5)
    ends_log_share = -digit_decay * np.arange(10.0)
    ends_log_share = ends_log_share - pt.logsumexp(ends_log_share) + np.log(10.0)
    familiar_log_share = pt.concatenate([pt.zeros(N_FAMILIAR - 10), ends_log_share]) - np.log(N_FAMILIAR)
    log_prior_rules = pt.concatenate(
        [familiar_log_share + log_w[0], log_w[1] + dsum_log_share]
    )
    log_rules = log_prior_rules[None, :] + rule_log_lik
    log_intervals = log_w[2] + log_per_interval[None, :] + interval_log_lik
    log_joint = pt.concatenate([log_rules, log_intervals], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + posterior[:, N_RULES:] @ pt.as_tensor(
        INTERVAL_MEMBERSHIP
    )

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    interval_cheb = np.zeros((len(unique), len(LO), CHEB_DEGREE + 1))
    for i, s in enumerate(unique):
        for x in s:
            hit = (LO <= x) & (x <= HI)
            interval_cheb[i] += CHEB_TABLE[np.where(hit, (WIDTH - 1) * 100 + (x - LO), OUTSIDE)]
    return {
        "rule_log_lik": np.stack([RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "interval_cheb": interval_cheb,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
