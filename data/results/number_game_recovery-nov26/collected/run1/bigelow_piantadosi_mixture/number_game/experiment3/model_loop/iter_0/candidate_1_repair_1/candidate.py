"""Two-digit-ending union concepts.

People's space of candidate concepts includes, besides single rules and
numerical ranges, "either-or" last-digit concepts -- numbers ending in one of
two digits, such as "ends in 2 or 7" (82, 67, 72) -- entertained as written-form
rules with their own fitted prior plausibility (`log_union_w`), shared equally
among the 45 such pairs. Otherwise people learn as the current best
single-sticky-guess learner does: one working guess (a rule or an interval with
an Erlang width prior; digit rules -- the unions included -- discounted and
salient only once two examples fit them; fitted trust `alpha`) formed from the
first example, kept with probability `stay` while later examples fit it,
dropped with probability `disrupt` when a new example reveals a shared digit,
and kept despite a contradicting example with probability `tolerate`.

For speed, intervals of equal width that contain the same examples are pooled
(their prior depends only on width), and every quantity that does not depend
on the parameters is precomputed once per distinct example set.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4


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


def union_concepts():
    return {f"ends in {a} or {b}": [n for n in NUMBERS if n % 10 in (a, b)]
            for a in range(10) for b in range(a + 1, 10)}


RULE_NAMES = list(rule_concepts().keys()) + list(union_concepts().keys())
RULES = list(rule_concepts().values()) + list(union_concepts().values())
IS_DIGIT_RULE = np.array([n.startswith("ends in") or n.startswith("contains digit") for n in RULE_NAMES], dtype="float64")
IS_UNION = np.array([" or " in n for n in RULE_NAMES], dtype="float64")
IS_SINGLE_DIGIT_RULE = IS_DIGIT_RULE * (1 - IS_UNION)
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
LO = np.array([m for m in NUMBERS for n in range(m, 101)])
HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INTERVAL_SIZE = (HI - LO + 1).astype("float64")
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")

with pm.Model() as model:
    contains = pm.Data("contains", np.ones((1, MAX_N, N_RULES + 1)))  # example k lies in hypothesis
    step_mask = pm.Data("step_mask", np.zeros((1, MAX_N)))  # example k exists
    group_size = pm.Data("group_size", np.ones((1, 1)))  # width of the intervals in a group
    group_logcount = pm.Data("group_logcount", np.zeros((1, 1)))  # log #intervals (-1e4 for padding)
    new_coinc = pm.Data("new_coinc", np.zeros((1, MAX_N)))  # example k newly reveals a shared digit
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_rule_in = pm.Data("row_rule_in", np.zeros((1, N_RULES)))  # target lies in rule
    row_group_frac = pm.Data("row_group_frac", np.zeros((1, 1)))  # share of group's intervals containing target
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    n_sets = step_mask.shape[0]
    n_groups = group_size.shape[1]

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pt.exp(log_sigma)
    # Log prior weight of written-form rules relative to arithmetic rules.
    log_digit_w = pm.Normal("log_digit_w", mu=0.0, sigma=1.5)
    # Log prior weight of the whole two-digit-ending union family relative to
    # one ordinary rule, split equally among its 45 members.
    log_union_w = pm.Normal("log_union_w", mu=-1.0, sigma=1.5)
    rule_lw = (log_digit_w * pt.as_tensor(IS_DIGIT_RULE)
               + (log_union_w - np.log(IS_UNION.sum())) * pt.as_tensor(IS_UNION))
    rule_log_prior = rule_lw - pt.logsumexp(rule_lw)

    size_all = pt.as_tensor(INTERVAL_SIZE)
    log_norm = pt.logsumexp(pt.log(size_all) - size_all / sigma)
    group_log_prior = group_logcount + pt.log(group_size) - group_size / sigma - log_norm
    log_prior = pt.concatenate([
        pt.zeros((n_sets, N_RULES)) + pt.log(lambda_0) + rule_log_prior[None, :],
        pt.log1p(-lambda_0) + group_log_prior,
    ], axis=1)  # (sets, hyp)

    stay = pm.Beta("stay", 2, 2)  # keep a guess the new example fits
    disrupt = pm.Beta("disrupt", 2, 2)  # a newly noticed coincidence prompts a rethink
    tolerate = pm.Beta("tolerate", 1, 4)  # keep a guess despite a contradicting example
    alpha = pm.Beta("alpha", 9, 1)  # trust that examples come from the concept

    hyp_size = pt.concatenate([pt.zeros((n_sets, N_RULES)) + pt.as_tensor(RULE_SIZE)[None, :], group_size], axis=1)
    log_in = pt.log(alpha / hyp_size + (1 - alpha) / 100.0)[:, None, :]
    log_out = pt.log1p(-alpha) - np.log(100.0)
    step_ll = step_mask[:, :, None] * (contains * log_in + (1 - contains) * log_out)
    prefix_log_lik = pt.cumsum(step_ll, axis=1)

    # A written-form rule is salient at step k only once two of examples 1..k fit it.
    log_unnoticed = pm.Normal("log_unnoticed", mu=0.0, sigma=1.5)
    n_fit = pt.cumsum(step_mask[:, :, None] * contains[:, :, :N_RULES], axis=1)
    unnoticed = pt.as_tensor(IS_DIGIT_RULE)[None, None, :] * pt.lt(n_fit, 1.5)
    notice_lp = pt.concatenate([log_unnoticed * unnoticed, pt.zeros((n_sets, MAX_N, n_groups))], axis=2)

    post = pt.special.softmax(log_prior[:, None, :] + notice_lp + prefix_log_lik, axis=2)

    q = post[:, 0, :]
    for k in range(1, MAX_N):
        stay_k = stay * (1 - disrupt * new_coinc[:, k])[:, None]
        kept = stay_k * q * (contains[:, k, :] + tolerate * (1 - contains[:, k, :]))
        updated = kept + (1 - kept.sum(axis=1, keepdims=True)) * post[:, k, :]
        m = step_mask[:, k][:, None]
        q = m * updated + (1 - m) * q

    q_row = q[row_set]
    gen = (q_row[:, :N_RULES] * row_rule_in).sum(axis=1) + (q_row[:, N_RULES:] * row_group_frac).sum(axis=1)
    p_yes = pm.Deterministic("p_yes", pt.clip(gen, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    per_set = []
    for s in unique:
        inside = np.stack([(LO <= x) & (HI >= x) for x in s], axis=1)
        code = (inside * (1 << np.arange(len(s)))).sum(axis=1)
        keys, group = np.unique(np.stack([INTERVAL_SIZE.astype(int), code], axis=1), axis=0, return_inverse=True)
        group = group.reshape(-1)
        count = np.bincount(group, minlength=len(keys)).astype(float)
        # share of each group's intervals that contain each number
        frac = np.zeros((len(keys), 100))
        np.add.at(frac, group, INTERVAL_MEMBERSHIP)
        frac /= count[:, None]
        per_set.append((s, keys, count, frac))
    n_sets = len(unique)
    g_max = max(len(p[1]) for p in per_set)
    contains = np.zeros((n_sets, MAX_N, N_RULES + g_max))
    mask = np.zeros((n_sets, MAX_N))
    gsize = np.ones((n_sets, g_max))
    glogcount = np.full((n_sets, g_max), -1e4)
    new_coinc = np.zeros((n_sets, MAX_N))
    fracs = np.zeros((n_sets, g_max, 100))
    digit_rows = RULE_MEMBERSHIP[IS_SINGLE_DIGIT_RULE > 0]
    for i, (s, keys, count, frac) in enumerate(per_set):
        n_g = len(keys)
        gsize[i, :n_g] = keys[:, 0]
        glogcount[i, :n_g] = np.log(count)
        fracs[i, :n_g] = frac
        for k, x in enumerate(s):
            contains[i, k, :N_RULES] = RULE_MEMBERSHIP[:, x - 1]
            contains[i, k, N_RULES:N_RULES + n_g] = (keys[:, 1] >> k) & 1
            mask[i, k] = 1.0
        for k in range(1, len(s)):
            before = digit_rows[:, [x - 1 for x in s[:k]]].sum(axis=1)
            if np.any((digit_rows[:, s[k] - 1] > 0) & (before == 1)):
                new_coinc[i, k] = 1.0
    row_set = np.array([index[s] for s in sets], dtype="int64")
    row_target = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    return {
        "contains": contains,
        "step_mask": mask,
        "group_size": gsize,
        "group_logcount": glogcount,
        "new_coinc": new_coinc,
        "row_set": row_set,
        "row_rule_in": RULE_MEMBERSHIP[:, row_target].T.copy(),
        "row_group_frac": fracs[row_set, :, row_target],
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
