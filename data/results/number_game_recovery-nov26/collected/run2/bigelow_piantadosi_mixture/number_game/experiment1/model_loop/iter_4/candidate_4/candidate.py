"""Refinement of concepts_sampled_without_replacement: Bayesian concept learning
over the same 45 rules and every interval of 1..100, with a rule prior
proportional to |rule| ** gamma, a response lapse (guessing "yes" at rate
`guess_yes` on a fraction `lapse` of trials) and strong sampling without
replacement. The one change (taken from breadth_prior_concepts_free_outlier_rate):
the rate `stray` at which people believe an example is a random number from
1..100 is inferred rather than fixed at 0.1. The likelihood of the k-th example
under concept h is (1 - stray) * [x in h] / (|h| - members of h already shown)
+ stray / 100, so a misfitting example can be discounted as a stray and tight
concepts survive it.

The rule likelihoods are computed exactly in the model. The interval terms
(log evidence per set, log interval mass on each asked-about target) depend on
`stray` only, so they are tabulated once at Chebyshev nodes in log(stray) and
evaluated by the Chebyshev series inside the model.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

NUMBERS = np.arange(1, 101)
MAX_EX = 4
LOG_EPS_LO, LOG_EPS_HI = np.log(1e-3), np.log(0.6)  # range of the stray rate
CHEB_DEG = 24


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
CONCEPT_SIZE = MEMBERSHIP.sum(axis=1)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
LOG_RULE_SIZE = np.log(CONCEPT_SIZE[:N_RULES])
LOG_RULE_SIZE_C = LOG_RULE_SIZE - LOG_RULE_SIZE.mean()


def example_terms(examples, membership, sizes):
    """Per example position: kind (0 pad, 1 first showing, 2 repeat), in_h and members left."""
    k = len(examples)
    kind = np.zeros(MAX_EX)
    in_h = np.zeros((MAX_EX, len(membership)))
    remaining = np.ones((MAX_EX, len(membership)))
    shown = np.zeros(len(membership))
    seen = set()
    for i, x in enumerate(examples[:MAX_EX]):
        if x in seen:  # a repeated number can only be a stray draw
            kind[i] = 2
            continue
        seen.add(x)
        kind[i] = 1
        in_h[i] = membership[:, x - 1]
        remaining[i] = np.maximum(sizes - shown, 1.0)
        shown += in_h[i]
    return kind, in_h, remaining


def set_log_lik(kind, in_h, remaining, eps):
    total = np.zeros(in_h.shape[1])
    for i in range(MAX_EX):
        if kind[i] == 1:
            total += np.log((1 - eps) * in_h[i] / remaining[i] + eps / 100)
        elif kind[i] == 2:
            total += np.log(eps / 100)
    return total


def cheb_series(coefs, x):
    """Evaluate sum_k coefs[..., k] T_k(x) for scalar x (Chebyshev recurrence)."""
    t_prev, t_cur = pt.ones_like(x), x
    out = coefs[..., 0] * t_prev + coefs[..., 1] * t_cur
    for k in range(2, CHEB_DEG + 1):
        t_prev, t_cur = t_cur, 2 * x * t_cur - t_prev
        out = out + coefs[..., k] * t_cur
    return out


with pm.Model() as model:
    rule_in_h = pm.Data("rule_in_h", np.zeros((1, MAX_EX, N_RULES)))  # (sets, pos, rules)
    rule_remaining = pm.Data("rule_remaining", np.ones((1, MAX_EX, N_RULES)))
    is_first = pm.Data("is_first", np.zeros((1, MAX_EX)))  # first showing of a number
    is_repeat = pm.Data("is_repeat", np.zeros((1, MAX_EX)))
    iev_coef = pm.Data("iev_coef", np.zeros((1, CHEB_DEG + 1)))  # (sets, deg): interval log evidence
    imass_coef = pm.Data("imass_coef", np.zeros((1, CHEB_DEG + 1)))  # (rows, deg): log interval mass on target
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))  # target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    stray = pm.Beta("stray", 2, 10)
    eps = pt.clip(stray, 1.001e-3, 0.599)
    x_cheb = (2 * pt.log(eps) - (LOG_EPS_LO + LOG_EPS_HI)) / (LOG_EPS_HI - LOG_EPS_LO)

    # Exact rule likelihoods under sampling without replacement with strays.
    pos_ll = pt.log((1 - eps) * rule_in_h / rule_remaining + eps / 100)
    rule_log_lik = (pos_ll * is_first[:, :, None]).sum(axis=1) + (is_repeat.sum(axis=1) * pt.log(eps / 100))[:, None]

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 1.0)
    log_rule_prior = gamma * LOG_RULE_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)
    rule_joint = rule_log_lik + log_rule_prior[None, :]
    rule_log_evidence = pt.logsumexp(rule_joint, axis=1)
    rule_post = pt.exp(rule_joint - rule_log_evidence[:, None])
    rule_in_concept = pt.dot(rule_post, RULE_MEMBERSHIP)  # (sets, targets)

    interval_log_evidence = cheb_series(iev_coef, x_cheb)  # (sets,)
    interval_in_concept = pt.exp(cheb_series(imass_coef, x_cheb) - interval_log_evidence[row_set])  # (rows,)

    log_rules = pt.log(lambda_0) + rule_log_evidence
    log_intervals = pt.log1p(-lambda_0) - np.log(len(INTERVALS)) + interval_log_evidence
    p_rule = pm.math.sigmoid(log_rules - log_intervals)[row_set]
    in_concept = p_rule * rule_in_concept[row_set, row_target] + (1 - p_rule) * interval_in_concept

    p_concept = pt.clip(in_concept, 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    row_set = np.array([index[s] for s in sets], dtype="int64")

    rule_terms = [example_terms(s, RULE_MEMBERSHIP, CONCEPT_SIZE[:N_RULES]) for s in unique]
    interval_terms = [example_terms(s, MEMBERSHIP[N_RULES:], CONCEPT_SIZE[N_RULES:]) for s in unique]
    kinds = np.stack([t[0] for t in rule_terms])

    # Interval terms tabulated at Chebyshev nodes in log(stray).
    nodes = np.cos(np.pi * (np.arange(CHEB_DEG + 1) + 0.5) / (CHEB_DEG + 1))
    log_eps = 0.5 * (nodes * (LOG_EPS_HI - LOG_EPS_LO) + LOG_EPS_LO + LOG_EPS_HI)
    iev = np.zeros((len(unique), len(nodes)))
    imass = np.zeros((len(rows), len(nodes)))
    interval_membership = MEMBERSHIP[N_RULES:]
    for j, le in enumerate(log_eps):
        for i, (kind, in_h, remaining) in enumerate(interval_terms):
            ll = set_log_lik(kind, in_h, remaining, np.exp(le))
            top = ll.max()
            w = np.exp(ll - top)
            iev[i, j] = top + np.log(w.sum())
            mass = w @ interval_membership  # (targets,)
            sel = row_set == i
            imass[sel, j] = top + np.log(np.maximum(mass[targets[sel]], 1e-300))
    iev_coef = np.polynomial.chebyshev.chebfit(nodes, iev.T, CHEB_DEG).T
    imass_coef = np.polynomial.chebyshev.chebfit(nodes, imass.T, CHEB_DEG).T
    return {
        "rule_in_h": np.stack([t[1] for t in rule_terms]),
        "rule_remaining": np.stack([t[2] for t in rule_terms]),
        "is_first": (kinds == 1).astype("float64"),
        "is_repeat": (kinds == 2).astype("float64"),
        "iev_coef": iev_coef,
        "imass_coef": imass_coef,
        "row_set": row_set,
        "row_target": targets,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
