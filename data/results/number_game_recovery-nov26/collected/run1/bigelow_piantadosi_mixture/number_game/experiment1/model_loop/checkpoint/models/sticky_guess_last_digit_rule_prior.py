"""Refinement of sticky_guess_power_law_ranges: win-stay, lose-shift concept
tracking with a power-law prior on interval endpoints, plus one change: the ten
"ends in d" (last-digit) rules share a single fitted log prior offset relative
to the other rules, so people may find last-digit rules less natural guesses
than familiar arithmetic rules and generalise to far-away same-last-digit
numbers less than the equal-rule prior predicts.

Original description:

People hold a single working guess about the concept (one of 45 rules or an
interval [m, n] of 1..100). After the first example the guess is drawn from the
Bayesian posterior (size principle, 10% noisy examples, rules sharing prior mass
lambda_0 equally). As each further example arrives, if it lies in the current
guess it is kept with probability `stay`; otherwise the guess is redrawn from
the posterior given all examples so far. "Yes" is the probability that the final
guess contains the target. The one change (from power_law_interval_endpoints):
an interval's prior is the incumbent's Erlang density on its size times a
power law (m n)^-gamma over its endpoints, with gamma fitted, so where a range
lies on the number line shapes how readily it is guessed and kept.

For speed, intervals are grouped per example set by (size, which examples they
contain): members of a group share likelihood and containment of every example,
so the guess dynamics are exact on the groups, each group carrying the summed
prior of its members; within a group, membership of the target is weighted by
each member's power-law prior.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
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


RULES = list(rule_concepts().values())
RULE_NAMES = list(rule_concepts().keys())
N_RULES = len(RULES)
IS_LAST_DIGIT = np.array([name.startswith("ends in") for name in RULE_NAMES], dtype="float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_EX_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
LO = np.array([m for m in NUMBERS for n in range(m, 101)])
HI = np.array([n for m in NUMBERS for n in range(m, 101)])
N_INT = len(LO)
INTERVAL_SIZE = (HI - LO + 1).astype("float64")
LOG_ENDPOINTS = np.log(LO) + np.log(HI)
LOG_ENDPOINTS_C = LOG_ENDPOINTS - LOG_ENDPOINTS.mean()
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
N_HYP = N_RULES + G_MAX

with pm.Model() as model:
    prefix_log_lik = pm.Data("prefix_log_lik", np.zeros((1, MAX_N, N_HYP)))  # log lik of examples 1..k
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
    # Power-law endpoint exponent (0 recovers the incumbent's location-free prior).
    gamma = pm.Normal("gamma", mu=0.0, sigma=1.0)

    # Log prior offset of the last-digit rules relative to the other rules
    # (0 recovers the incumbent's equal rule prior).
    last_digit_offset = pm.Normal("last_digit_offset", mu=0.0, sigma=1.5)
    rule_log_w = last_digit_offset * pt.as_tensor(IS_LAST_DIGIT)
    rule_log_prior = rule_log_w - pt.logsumexp(rule_log_w)

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
        pt.zeros((n_sets, N_RULES)) + (pt.log(lambda_0) + rule_log_prior)[None, :],
        pt.log1p(-lambda_0) + group_log_prior,
    ], axis=1)  # (sets, hyp)

    # Probability of keeping a guess that the new example fits.
    stay = pm.Beta("stay", 2, 2)

    post = pt.special.softmax(log_prior[:, None, :] + prefix_log_lik, axis=2)  # (sets, steps, hyp)

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
    prefix = np.zeros((n_sets, MAX_N, N_HYP))
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
        g_ex_ll = np.where(g_in > 0, np.log(ALPHA / keys[:, :1] + (1 - ALPHA) / 100), np.log((1 - ALPHA) / 100))
        running = np.zeros(N_HYP)
        for k in range(MAX_N):
            if k < len(s):
                step_ll = np.zeros(N_HYP)
                step_ll[:N_RULES] = RULE_EX_LOG_LIK[:, s[k] - 1]
                step_ll[N_RULES:N_RULES + n_g] = g_ex_ll[:, k]
                running = running + step_ll
                contains[i, k, :N_RULES] = RULE_MEMBERSHIP[:, s[k] - 1]
                contains[i, k, N_RULES:] = 0.0
                contains[i, k, N_RULES:N_RULES + n_g] = g_in[:, k]
                mask[i, k] = 1.0
            prefix[i, k] = running
    return {
        "prefix_log_lik": prefix,
        "contains": contains,
        "step_mask": mask,
        "group_size": gsize,
        "group_present": present,
        "interval_group": igroup,
        "row_set": np.array([index[s] for s in sets], dtype="int64"),
        "row_target": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
