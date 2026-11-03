"""People do Bayesian concept learning over the same 45 arithmetic and digit
rules and every interval of 1..100 as `concepts_sampled_without_replacement`
(rule prior proportional to |rule| ** gamma, distinct examples drawn without
replacement, 10% of examples uniform strays, a guessing lapse), but they believe
the program favours numbers of a particular magnitude: a member x of concept h
is produced with probability proportional to x ** (-beta), so beta > 0 is a
small-number bias and beta < 0 a large-number bias. The likelihood of the k-th
example is 0.9 * [x in h] * x**-beta / (magnitude-weighted mass of h's members
not yet shown) + 0.1 / 100. This one magnitude-biased sampling assumption makes
large examples stronger (beta > 0) evidence for concepts made of large numbers.

beta is the only parameter that enters the hypothesis likelihoods, so each
set's rule log-likelihoods, interval log-evidence and interval predictions are
tabulated as Chebyshev series in beta in `prepare_observed`.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
BETA_MAX = 0.6  # beta lives in [-BETA_MAX, BETA_MAX]
DEGREE = 20


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
RULE_SIZE = MEMBERSHIP[:N_RULES].sum(axis=1)
LOG_RULE_SIZE_C = np.log(RULE_SIZE) - np.log(RULE_SIZE).mean()
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
NODES = np.cos(np.pi * (np.arange(DEGREE + 1) + 0.5) / (DEGREE + 1))  # Chebyshev nodes in u = beta / BETA_MAX


def set_log_lik(examples, beta):
    """log P(examples | h) for every hypothesis under magnitude-biased sampling without replacement."""
    w = NUMBERS.astype("float64") ** (-beta)
    mass = MEMBERSHIP @ w
    total = np.zeros(len(MEMBERSHIP))
    shown = np.zeros(len(MEMBERSHIP))
    seen = set()
    for x in examples:
        if x in seen:  # a repeated number can only be a stray draw
            total += np.log((1 - ALPHA) / 100)
            continue
        seen.add(x)
        in_h = MEMBERSHIP[:, x - 1]
        remaining = np.maximum(mass - shown, 1e-300)
        total += np.log(ALPHA * in_h * w[x - 1] / remaining + (1 - ALPHA) / 100)
        shown += in_h * w[x - 1]
    return total


def chebyshev_coefficients(values):
    """values: (..., DEGREE + 1) at NODES -> coefficients (..., DEGREE + 1)."""
    return np.polynomial.chebyshev.chebfit(NODES, values.reshape(-1, DEGREE + 1).T, DEGREE).T.reshape(values.shape)


with pm.Model() as model:
    rule_log_lik_c = pm.Data("rule_log_lik_c", np.zeros((1, N_RULES, DEGREE + 1)))  # (sets, rules, coef)
    interval_log_evidence_c = pm.Data("interval_log_evidence_c", np.zeros((1, DEGREE + 1)))  # (sets, coef)
    interval_in_concept_c = pm.Data("interval_in_concept_c", np.zeros((1, 100, DEGREE + 1)))  # (sets, targets, coef)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    beta = pm.TruncatedNormal("beta", mu=0.0, sigma=0.2, lower=-BETA_MAX, upper=BETA_MAX)
    u = beta / BETA_MAX
    basis = [pt.ones_like(u), u]
    for _ in range(DEGREE - 1):
        basis.append(2 * u * basis[-1] - basis[-2])
    basis = pt.stack(basis)  # (coef,)

    rule_log_lik = pt.tensordot(rule_log_lik_c, basis, axes=[[2], [0]])  # (sets, rules)
    interval_log_evidence = pt.dot(interval_log_evidence_c, basis)  # (sets,)
    interval_in_concept = pt.tensordot(interval_in_concept_c, basis, axes=[[2], [0]])  # (sets, targets)

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


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    rule_ll = np.zeros((len(unique), N_RULES, DEGREE + 1))
    int_ev = np.zeros((len(unique), DEGREE + 1))
    int_pred = np.zeros((len(unique), 100, DEGREE + 1))
    for j, node in enumerate(NODES):
        for i, s in enumerate(unique):
            ll = set_log_lik(s, node * BETA_MAX)
            interval_ll = ll[N_RULES:]
            ev = logsumexp(interval_ll)
            rule_ll[i, :, j] = ll[:N_RULES]
            int_ev[i, j] = ev
            int_pred[i, :, j] = np.exp(interval_ll - ev) @ INTERVAL_MEMBERSHIP
    return {
        "rule_log_lik_c": chebyshev_coefficients(rule_ll),
        "interval_log_evidence_c": chebyshev_coefficients(int_ev),
        "interval_in_concept_c": chebyshev_coefficients(int_pred),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
