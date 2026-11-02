"""Target-as-evidence (positive-test) concept learning.

People infer the hidden program as a Bayesian concept learner (number rules
incl. digit-sum rules with a graded prior over sums, ranges with an Erlang
width prior and centre-weighted outputs, a prior mass per kind of program, the
size principle, 10% noise examples), but when they judge a target they test it
positively: they imagine the target as one more output of the program and let
it count, with a partial weight inferred from the data, as evidence for the
programs that would produce it. Programs containing the target, the more so
the smaller they are, gain belief at the moment of judgement, so targets that
fit a tight concept are endorsed more than the examples alone justify, while
targets fitting only broad concepts gain little (weight 0 is the normative
learner).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from numpy.polynomial import chebyshev
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
    rules.update({f"digit sum {t}": [n for n in NUMBERS if sum(map(int, str(n))) == t] for t in range(1, 19)})
    return rules


def family(name):
    if name in ("odd", "even"):
        return 0
    if name.startswith("multiples"):
        return 2
    if name.startswith("ends"):
        return 3
    if name.startswith("contains"):
        return 4
    if name.startswith("digit sum"):
        return 6
    if name.startswith("powers of 2,"):
        return 5  # gerrymandered exception rules
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
RULES = list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in _RULES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
IS_DSUM = RULE_FAMILY == 6
DSUM_INDEX = np.where(IS_DSUM)[0]
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


# An interval's log likelihood is a smooth function of kappa alone, so each
# table entry is expanded in Chebyshev polynomials of 2 * kappa - 1 on [0, 1]
# (max error ~5e-5 nats); the last row is "outside the range".
_NODES = np.cos(np.pi * (np.arange(30) + 0.5) / 30)
_TABLES = np.stack([example_log_lik_table((x + 1) / 2).reshape(-1) for x in _NODES])
CHEB_TABLE = np.concatenate(
    [
        chebyshev.chebfit(_NODES, _TABLES, CHEB_DEGREE).T,
        np.eye(1, CHEB_DEGREE + 1) * np.log((1 - ALPHA) / 100),
    ]
)  # (100 * 100 + 1, degree + 1)
OUTSIDE = 100 * 100

# Per (interval, target): index into the Chebyshev table of the log probability
# that the interval produces the target (OUTSIDE if it does not contain it).
_T = NUMBERS[None, :]
TARGET_CELL = np.where(
    (_T >= LO[:, None]) & (_T <= HI[:, None]), (WIDTH[:, None] - 1) * 100 + (_T - LO[:, None]), OUTSIDE
)
MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, INTERVAL_MEMBERSHIP])  # (hypotheses, 100)
LOG_NOISE = np.log((1 - ALPHA) / 100)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    interval_cheb = pm.Data("interval_cheb", np.zeros((1, len(LO), CHEB_DEGREE + 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_weight = pm.Dirichlet("family_weight", a=np.ones(8))
    log_w = pt.log(family_weight)
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    centre_weight = pm.Beta("centre_weight", alpha=1.5, beta=6.0)
    dsum_decay = pm.Normal("dsum_decay", mu=0.0, sigma=1.0)
    # Evidential weight the judged target itself receives (0 = normative).
    target_weight = pm.HalfNormal("target_weight", sigma=0.5)

    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    log_per_interval = (erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH)))[WIDTH - 1]

    x = 2.0 * centre_weight - 1.0
    cheb = [pt.ones(()), x]
    for _ in range(CHEB_DEGREE - 1):
        cheb.append(2.0 * x * cheb[-1] - cheb[-2])
    cheb = pt.stack(cheb)
    interval_log_lik = pt.tensordot(interval_cheb, cheb, axes=[[2], [0]])  # (sets, intervals)

    dsum_log_share = -dsum_decay * DSUM_VALUE
    dsum_log_share = dsum_log_share - pt.logsumexp(dsum_log_share)
    within = pt.set_subtensor(pt.as_tensor(-np.log(FAMILY_SIZE)[RULE_FAMILY])[DSUM_INDEX], dsum_log_share)
    log_rules = (log_w[:7][RULE_FAMILY] + within)[None, :] + rule_log_lik
    log_intervals = log_w[7] + log_per_interval[None, :] + interval_log_lik
    log_joint = pt.concatenate([log_rules, log_intervals], axis=1)
    weight = pt.exp(log_joint - pt.max(log_joint, axis=1, keepdims=True))  # (sets, hypotheses)

    # Log probability that each hypothesis would produce each target.
    table = pt.dot(pt.as_tensor(CHEB_TABLE), cheb)
    target_log_lik = pt.concatenate([pt.as_tensor(RULE_LOG_LIK), table[TARGET_CELL]], axis=0)
    # Programs containing the target gain (P(t|h) / P(t|noise))^weight; others are unchanged.
    boost = pt.as_tensor(MEMBERSHIP) * pt.exp(target_weight * (target_log_lik - LOG_NOISE))
    yes_mass = pt.dot(weight, boost)
    in_mass = pt.dot(weight, pt.as_tensor(MEMBERSHIP))
    total = pt.sum(weight, axis=1, keepdims=True)
    in_concept = yes_mass / (yes_mass + total - in_mass)

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
