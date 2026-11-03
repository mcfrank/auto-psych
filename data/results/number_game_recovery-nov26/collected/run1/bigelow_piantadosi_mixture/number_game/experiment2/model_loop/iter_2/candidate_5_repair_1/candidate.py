"""Refinement of sticky_guess_fitted_sampling_trust: the working guess may be a
rule confined to a decade-bounded region of the number line.

People hold a single working guess about the concept (one of 45 rules or an
interval of 1..100, power-law-endpoint and Erlang-width interval prior, digit
rules discounted by one fitted weight, fitted trust `alpha` that examples come
from the concept), formed from the first example by Bayesian concept learning,
kept with probability `stay` while later examples fit it and redrawn from the
posterior given all examples so far when contradicted. The one change (the
component taken from decade_restricted_rule_concepts): each rule can also be
held in a version confined to a decade-bounded region (from 1, 11, ..., 91 up
to 10, 20, ..., 100, e.g. "ends in 6 between 41 and 50"); a fitted share of
each rule's prior is spread equally over its restricted versions. Restricted
rules are small, so the size principle favours them when the examples sit in
one region, curbing extension of a rule (e.g. a lone 46's last digit) to
far-away targets such as 96.

For speed, intervals are grouped per example set by (size, which examples they
contain), and restricted rules by (size, which examples they contain, digit or
arithmetic family); both groupings are exact.
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
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)

# Decade-restricted versions of each rule (proper decade-bounded sub-ranges of
# 1..100); each rule's restricted share is split equally over its versions.
RANGES = [(lo, hi) for lo in range(1, 92, 10) for hi in range(lo + 9, 101, 10) if (lo, hi) != (1, 100)]
_base, _mem = [], []
for _r in range(N_RULES):
    for _lo, _hi in RANGES:
        _m = RULE_MEMBERSHIP[_r] * ((NUMBERS >= _lo) & (NUMBERS <= _hi))
        if _m.sum() > 0:
            _base.append(_r)
            _mem.append(_m)
RC_BASE = np.array(_base, dtype="int64")
RC_MEMBERSHIP = np.array(_mem)
RC_SIZE = RC_MEMBERSHIP.sum(axis=1).astype(int)
RC_DIGIT = IS_DIGIT_RULE[RC_BASE].astype(int)
RC_SHARE = 1.0 / np.bincount(RC_BASE, minlength=N_RULES)[RC_BASE]  # within-rule share of each version

LO = np.array([m for m in NUMBERS for n in range(m, 101)])
HI = np.array([n for m in NUMBERS for n in range(m, 101)])
N_INT = len(LO)
INTERVAL_SIZE = (HI - LO + 1).astype("float64")
LOG_ENDPOINTS = np.log(LO) + np.log(HI)
LOG_ENDPOINTS_C = LOG_ENDPOINTS - LOG_ENDPOINTS.mean()
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
N_HYP = N_RULES + G_MAX

with pm.Model() as model:
    contains = pm.Data("contains", np.ones((1, MAX_N, N_HYP)))  # example k lies in hypothesis
    step_mask = pm.Data("step_mask", np.zeros((1, MAX_N)))  # example k exists
    group_size = pm.Data("group_size", np.ones((1, G_MAX)))
    group_present = pm.Data("group_present", np.zeros((1, G_MAX)))  # 1 for a real (non-padding) group
    interval_group = pm.Data("interval_group", np.zeros((1, N_INT), dtype="int64"))  # flat group id of each interval
    # Restricted-rule groups per set (padded to the data's maximum).
    rc_contains = pm.Data("rc_contains", np.zeros((1, MAX_N, 1)))  # example k lies in the group's rules
    rc_size = pm.Data("rc_size", np.ones((1, 1)))
    rc_log_w = pm.Data("rc_log_w", np.zeros((1, 1)))  # log summed within-rule share (-1e4 for padding)
    rc_digit = pm.Data("rc_digit", np.zeros((1, 1)))  # 1 for restricted digit rules
    rc_gen = pm.Data("rc_gen", np.zeros((1, 1, 100)))  # share-weighted fraction of the group containing each target
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))  # target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pt.exp(log_sigma)
    # Log prior weight of surface-digit rules relative to arithmetic rules.
    log_digit_w = pm.Normal("log_digit_w", mu=0.0, sigma=1.5)
    rule_lw = log_digit_w * pt.as_tensor(IS_DIGIT_RULE)
    log_rule_norm = pt.logsumexp(rule_lw)
    # Share of each rule's prior given to its decade-restricted versions.
    restrict = pm.Beta("restrict_share", 1.5, 3.0)
    full_log_prior = pt.log(lambda_0) + pt.log1p(-restrict) + rule_lw - log_rule_norm
    rc_log_prior = pt.log(lambda_0) + pt.log(restrict) + log_digit_w * rc_digit - log_rule_norm + rc_log_w
    # Power-law endpoint exponent.
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
        pt.zeros((n_sets, N_RULES)) + full_log_prior[None, :],
        rc_log_prior,
        pt.log1p(-lambda_0) + group_log_prior,
    ], axis=1)  # (sets, hyp): whole rules, restricted-rule groups, interval groups
    all_contains = pt.concatenate([contains[:, :, :N_RULES], rc_contains, contains[:, :, N_RULES:]], axis=2)
    n_rc = rc_size.shape[1]

    # Probability of keeping a guess that the new example fits.
    stay = pm.Beta("stay", 2, 2)

    # Fitted trust that examples come from the concept.
    alpha = pm.Beta("alpha", 9, 1)
    hyp_size = pt.concatenate([pt.zeros((n_sets, N_RULES)) + pt.as_tensor(RULE_SIZE)[None, :], rc_size, group_size], axis=1)
    log_in = pt.log(alpha / hyp_size + (1 - alpha) / 100.0)[:, None, :]  # (sets, 1, hyp)
    log_out = pt.log1p(-alpha) - np.log(100.0)
    step_ll = step_mask[:, :, None] * (all_contains * log_in + (1 - all_contains) * log_out)
    prefix_log_lik = pt.cumsum(step_ll, axis=1)  # log lik of examples 1..k

    post = pt.special.softmax(log_prior[:, None, :] + prefix_log_lik, axis=2)  # (sets, steps, hyp)

    q = post[:, 0, :]
    for k in range(1, MAX_N):
        kept = stay * q * all_contains[:, k, :]
        updated = kept + (1 - kept.sum(axis=1, keepdims=True)) * post[:, k, :]
        m = step_mask[:, k][:, None]
        q = m * updated + (1 - m) * q

    # Spread each group's final mass over its members in proportion to their prior.
    q_rc = q[:, N_RULES:N_RULES + n_rc]
    share = (q[:, N_RULES + n_rc:] / group_w_safe).flatten()
    q_int = (share[flat_gid] * w).reshape((n_sets, N_INT))
    gen = (pt.dot(q[:, :N_RULES], pt.as_tensor(RULE_MEMBERSHIP)) + (q_rc[:, :, None] * rc_gen).sum(axis=1)
           + pt.dot(q_int, pt.as_tensor(INTERVAL_MEMBERSHIP)))
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
    rc_groups = []
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
        rc_code = sum(RC_MEMBERSHIP[:, x - 1].astype(int) << j for j, x in enumerate(s))
        rkeys, rgroup = np.unique(np.stack([RC_SIZE, rc_code, RC_DIGIT], axis=1), axis=0, return_inverse=True)
        rc_groups.append((rkeys, rgroup.reshape(-1)))
    n_rc = max(len(k) for k, _ in rc_groups)
    rc_contains = np.zeros((n_sets, MAX_N, n_rc))
    rc_size = np.ones((n_sets, n_rc))
    rc_log_w = np.full((n_sets, n_rc), -1e4)
    rc_digit = np.zeros((n_sets, n_rc))
    rc_gen = np.zeros((n_sets, n_rc, 100))
    for i, (s, (rkeys, rgroup)) in enumerate(zip(unique, rc_groups)):
        n_r = len(rkeys)
        wsum = np.bincount(rgroup, weights=RC_SHARE, minlength=n_r)
        members = np.zeros((n_r, 100))
        np.add.at(members, rgroup, RC_SHARE[:, None] * RC_MEMBERSHIP)
        rc_size[i, :n_r] = rkeys[:, 0]
        rc_log_w[i, :n_r] = np.log(wsum)
        rc_digit[i, :n_r] = rkeys[:, 2]
        rc_gen[i, :n_r] = members / wsum[:, None]
        for k in range(len(s)):
            rc_contains[i, k, :n_r] = (rkeys[:, 1] >> k) & 1
    return {
        "contains": contains,
        "step_mask": mask,
        "group_size": gsize,
        "group_present": present,
        "interval_group": igroup,
        "rc_contains": rc_contains,
        "rc_size": rc_size,
        "rc_log_w": rc_log_w,
        "rc_digit": rc_digit,
        "rc_gen": rc_gen,
        "row_set": np.array([index[s] for s in sets], dtype="int64"),
        "row_target": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
