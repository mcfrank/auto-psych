"""People do Bayesian concept learning over the 45 arithmetic and digit rules
(prior proportional to |rule| ** gamma) and every interval of 1..100, with a
response lapse (guessing "yes" at rate `guess_yes`). They treat the examples as
distinct draws from the concept (strong sampling without replacement), 10% of
them stray. The one new claim: they believe the program does not pick numbers
uniformly. Round numbers (multiples of 5) carry a production weight exp(b)
relative to other numbers, and b (its sign and size) is inferred. So the k-th
example x is drawn with probability
    0.9 * [x in h] w(x) / (W(h) - weight of h's members already shown)
    + 0.1 * w(x) / W(1..100).
This shifts belief toward or away from concepts made of round numbers, and it
makes a round example more or less informative than a non-round one.

The examples' likelihoods depend on b nonlinearly. So each per-set quantity is
tabulated in prepare_observed as a Chebyshev series in b over [-B_MAX, B_MAX].
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from numpy.polynomial import chebyshev as cheb
from scipy.special import logsumexp

ALPHA = 0.9
B_MAX = 2.0
DEGREE = 16
NUMBERS = np.arange(1, 101)
ROUND = (NUMBERS % 5 == 0).astype("float64")


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
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
LOG_RULE_SIZE = np.log(MEMBERSHIP[:N_RULES].sum(axis=1))
LOG_RULE_SIZE_C = LOG_RULE_SIZE - LOG_RULE_SIZE.mean()
NODES = cheb.chebpts1(DEGREE + 1)  # in [-1, 1]; b = B_MAX * t


def set_log_lik(examples, b):
    """log P(examples | h) for every hypothesis under production weights exp(b * round)."""
    w = np.exp(b * ROUND)
    stray = (1 - ALPHA) * w / w.sum()
    weight_left = MEMBERSHIP @ w
    total = np.zeros(len(MEMBERSHIP))
    seen = set()
    for x in examples:
        if x in seen:  # a repeated number can only be a stray draw
            total += np.log(stray[x - 1])
            continue
        seen.add(x)
        in_h = MEMBERSHIP[:, x - 1]
        total += np.log(ALPHA * in_h * w[x - 1] / np.maximum(weight_left, 1e-12) + stray[x - 1])
        weight_left = weight_left - in_h * w[x - 1]
    return total


def set_summaries(examples, b):
    ll = set_log_lik(examples, b)
    interval_ll = ll[N_RULES:]
    ev = logsumexp(interval_ll)
    return ll[:N_RULES], ev, np.exp(interval_ll - ev) @ MEMBERSHIP[N_RULES:]


with pm.Model() as model:
    rule_coef = pm.Data("rule_coef", np.zeros((1, 1, N_RULES)))  # (degree+1, sets, rules)
    iev_coef = pm.Data("iev_coef", np.zeros((1, 1)))  # (degree+1, sets)
    iic_coef = pm.Data("iic_coef", np.zeros((1, 1, 100)))  # (degree+1, sets, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    b = pm.TruncatedNormal("round_bias", mu=0.0, sigma=0.75, lower=-B_MAX, upper=B_MAX)
    t = b / B_MAX
    basis = [pt.ones_like(t), t]
    for _ in range(DEGREE - 1):
        basis.append(2 * t * basis[-1] - basis[-2])
    T = pt.stack(basis)  # (degree+1,)
    rule_log_lik = pt.tensordot(T, rule_coef, axes=[[0], [0]])  # (sets, rules)
    interval_log_evidence = pt.tensordot(T, iev_coef, axes=[[0], [0]])  # (sets,)
    interval_in_concept = pt.clip(pt.tensordot(T, iic_coef, axes=[[0], [0]]), 0.0, 1.0)

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 1.0)
    log_rule_prior = gamma * LOG_RULE_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)
    rule_joint = rule_log_lik + log_rule_prior[None, :]
    rule_log_evidence = pt.logsumexp(rule_joint, axis=1)
    rule_post = pt.exp(rule_joint - rule_log_evidence[:, None])
    rule_in_concept = pt.dot(rule_post, RULE_MEMBERSHIP)

    log_rules = pt.log(lambda_0) + rule_log_evidence
    log_intervals = pt.log1p(-lambda_0) - np.log(len(INTERVALS)) + interval_log_evidence
    p_rule = pm.math.sigmoid(log_rules - log_intervals)[:, None]
    in_concept = p_rule * rule_in_concept + (1 - p_rule) * interval_in_concept

    p_concept = pt.clip(in_concept.reshape((-1,))[row_cell], 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)

SAMPLER_SETTINGS = {"target_accept": 0.9}


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    rl, iev, iic = [], [], []
    for t in NODES:
        out = [set_summaries(s, B_MAX * t) for s in unique]
        rl.append(np.stack([o[0] for o in out]))
        iev.append(np.array([o[1] for o in out]))
        iic.append(np.stack([o[2] for o in out]))

    def fit(values):  # values: (nodes, ...) -> Chebyshev coefficients (degree+1, ...)
        v = np.stack(values)
        flat = v.reshape(len(NODES), -1)
        return cheb.chebfit(NODES, flat, DEGREE).reshape((DEGREE + 1,) + v.shape[1:])

    return {
        "rule_coef": fit(rl),
        "iev_coef": fit(iev),
        "iic_coef": fit(iic),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
