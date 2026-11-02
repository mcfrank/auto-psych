"""Refinement of digit_coincidence_noticing_guess: every rule, not only a
surface-digit rule, must be noticed as a coincidence before it is a salient guess.

People hold a single working guess about the concept (one of 45 rules or an
interval of 1..100, power-law-endpoint and Erlang-width interval prior, digit
rules discounted by one fitted weight, fitted trust `alpha` in the examples),
formed from the first example by Bayesian concept learning, kept with
probability `stay` while later examples fit it and redrawn from the posterior
given all examples so far when contradicted. The incumbent made only digit
rules wait to be noticed (shared by at least two examples seen so far). The one
change: any rule -- odd, even, multiples, powers, squares, primes as well as
"ends in d" -- is seen in the examples only once at least two of them share it;
until then its prior is scaled by one fitted availability factor. So a lone
example is read mainly as a point on the number line (a range guess), and
far-away numbers sharing a property with a single example (its parity, its
last digit, a divisor) are endorsed less than when two examples share it.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4
G_MAX = 480  # interval groups per set (padded)


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


RULE_NAMES = list(rule_concepts().keys())
RULES = list(rule_concepts().values())
IS_DIGIT_RULE = np.array([n.startswith("ends in") or n.startswith("contains digit") for n in RULE_NAMES], dtype="float64")
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
LO = np.array([m for m in NUMBERS for n in range(m, 101)])
HI = np.array([n for m in NUMBERS for n in range(m, 101)])
N_INT = len(LO)
INTERVAL_SIZE = (HI - LO + 1).astype("float64")
LOG_ENDPOINTS = np.log(LO) + np.log(HI)
LOG_ENDPOINTS_C = LOG_ENDPOINTS - LOG_ENDPOINTS.mean()
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
N_HYP = N_RULES + G_MAX

RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)

with pm.Model() as model:
    contains = pm.Data("contains", np.ones((1, MAX_N, N_HYP)))  # example k lies in hypothesis
    step_mask = pm.Data("step_mask", np.zeros((1, MAX_N)))  # example k exists
    group_size = pm.Data("group_size", np.ones((1, G_MAX)))
    group_present = pm.Data("group_present", np.zeros((1, G_MAX)))  # 1 for a real (non-padding) group
    interval_group = pm.Data("interval_group", np.zeros((1, N_INT), dtype="int64"))  # flat group id of each interval
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))  # target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pt.exp(log_sigma)
    # Log prior weight of surface-digit rules relative to arithmetic rules (0 = incumbent).
    log_digit_w = pm.Normal("log_digit_w", mu=0.0, sigma=1.5)
    rule_lw = log_digit_w * pt.as_tensor(IS_DIGIT_RULE)
    rule_log_prior = rule_lw - pt.logsumexp(rule_lw)
    # Power-law endpoint exponent (0 recovers the incumbent's location-free prior).
    gamma = pm.Normal("gamma", mu=0.0, sigma=1.0)

    size = pt.as_tensor(INTERVAL_SIZE)
    endpoint_lw = -gamma * LOG_ENDPOINTS_C  # (intervals,)
    log_norm = pt.logsumexp(pt.log(size) - size / sigma + endpoint_lw)
    n_sets = interval_group.shape[0]
    flat_gid = interval_group.flatten()
    w = pt.tile(pt.exp(endpoint_lw), n_sets)  # endpoint weight of each (set, interval)
    group_w = pt.inc_subtensor(pt.zeros(n_sets * G_MAX)[flat_gid], w).reshape((n_sets, G_MAX))
    group_w_safe = group_w + (1 - group_present)
    group_log_prior = (pt.log(group_size) - group_size / sigma + pt.log(group_w_safe) - log_norm
                       - 1e4 * (1 - group_present))
    log_prior = pt.concatenate([
        pt.zeros((n_sets, N_RULES)) + pt.log(lambda_0) + rule_log_prior[None, :],
        pt.log1p(-lambda_0) + group_log_prior,
    ], axis=1)  # (sets, hyp)

    # Probability of keeping a guess that the new example fits.
    stay = pm.Beta("stay", 2, 2)

    # Fitted trust that examples come from the concept (incumbent: fixed 0.9).
    alpha = pm.Beta("alpha", 9, 1)
    hyp_size = pt.concatenate([pt.zeros((n_sets, N_RULES)) + pt.as_tensor(RULE_SIZE)[None, :], group_size], axis=1)
    log_in = pt.log(alpha / hyp_size + (1 - alpha) / 100.0)[:, None, :]  # (sets, 1, hyp)
    log_out = pt.log1p(-alpha) - np.log(100.0)
    step_ll = step_mask[:, :, None] * (contains * log_in + (1 - contains) * log_out)
    prefix_log_lik = pt.cumsum(step_ll, axis=1)  # log lik of examples 1..k

    # Coincidence noticing: any rule is available at step k only if at least two
    # of examples 1..k belong to it; otherwise its prior is scaled by
    # exp(log_unnoticed).
    log_unnoticed = pm.Normal("log_unnoticed", mu=0.0, sigma=1.5)
    n_fit = pt.cumsum(step_mask[:, :, None] * contains[:, :, :N_RULES], axis=1)  # (sets, steps, rules)
    unnoticed = pt.lt(n_fit, 1.5)  # every rule, not only digit rules
    notice_lp = pt.concatenate([log_unnoticed * unnoticed, pt.zeros((n_sets, MAX_N, G_MAX))], axis=2)

    post = pt.special.softmax(log_prior[:, None, :] + notice_lp + prefix_log_lik, axis=2)  # (sets, steps, hyp)

    q = post[:, 0, :]
    for k in range(1, MAX_N):
        kept = stay * q * contains[:, k, :]
        updated = kept + (1 - kept.sum(axis=1, keepdims=True)) * post[:, k, :]
        m = step_mask[:, k][:, None]
        q = m * updated + (1 - m) * q

    # Spread each group's final mass over its members in proportion to their prior.
    share = (q[:, N_RULES:] / group_w_safe).flatten()
    q_int = (share[flat_gid] * w).reshape((n_sets, N_INT))
    gen = pt.dot(q[:, :N_RULES], pt.as_tensor(RULE_MEMBERSHIP)) + pt.dot(q_int, pt.as_tensor(INTERVAL_MEMBERSHIP))
    p_yes = pm.Deterministic("p_yes", pt.clip(gen[row_set, row_target], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    contains = np.ones((n_sets, MAX_N, N_HYP))
    mask = np.zeros((n_sets, MAX_N))
    gsize = np.ones((n_sets, G_MAX))
    present = np.zeros((n_sets, G_MAX))
    igroup = np.zeros((n_sets, N_INT), dtype="int64")
    for i, s in enumerate(unique):
        inside = np.stack([(LO <= x) & (HI >= x) for x in s], axis=1)  # (intervals, examples)
        code = (inside * (1 << np.arange(len(s)))).sum(axis=1)
        keys, group = np.unique(np.stack([INTERVAL_SIZE.astype(int), code], axis=1), axis=0, return_inverse=True)
        group = group.reshape(-1)
        n_g = len(keys)
        assert n_g <= G_MAX
        igroup[i] = i * G_MAX + group
        gsize[i, :n_g] = keys[:, 0]
        present[i, :n_g] = 1.0
        g_in = np.zeros((n_g, len(s)))
        for j in range(len(s)):
            g_in[:, j] = (keys[:, 1] >> j) & 1
        for k in range(MAX_N):
            if k < len(s):
                contains[i, k, :N_RULES] = RULE_MEMBERSHIP[:, s[k] - 1]
                contains[i, k, N_RULES:] = 0.0
                contains[i, k, N_RULES:N_RULES + n_g] = g_in[:, k]
                mask[i, k] = 1.0
    return {
        "contains": contains,
        "step_mask": mask,
        "group_size": gsize,
        "group_present": present,
        "interval_group": igroup,
        "row_set": np.array([index[s] for s in sets], dtype="int64"),
        "row_target": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
