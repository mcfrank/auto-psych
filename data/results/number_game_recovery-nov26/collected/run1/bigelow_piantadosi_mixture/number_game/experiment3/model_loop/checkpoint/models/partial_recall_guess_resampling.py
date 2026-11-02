"""Partial recall when rethinking a working guess.

People learn as the incumbent gamma_width_range_prior_guess does (a single
working guess — one of 45 rules or an interval of 1..100, gamma width prior,
digit rules discounted and salient only once two examples share the digit,
fitted trust `alpha` — formed from the first example, kept with probability
`stay` while later examples fit it, dropped with probability `disrupt` when a
new example reveals a shared digit, and kept despite a contradicting example
with probability 1 - (1 - tolerate) ** (earlier examples it explains)). The one
change: memory of earlier examples is limited. When people drop their guess and
rethink at example k, they always use example k but recall each earlier example
independently with fitted probability `recall`, and draw the new guess from the
Bayesian posterior given only the recalled examples (averaged over which ones
are recalled). recall = 1 recovers the incumbent; recall = 0 is the classic
no-memory hypothesis-resampling learner.
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
LO = np.array([m for m in NUMBERS for n in range(m, 101)])
HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INTERVAL_SIZE = (HI - LO + 1).astype("float64")
WIDTHS = np.arange(1, 101, dtype="float64")
LOG_WIDTH_COUNT = np.log(101 - WIDTHS)  # number of intervals of each width
N_HYP = N_RULES + G_MAX

with pm.Model() as model:
    contains = pm.Data("contains", np.ones((1, MAX_N, N_HYP)))  # example k lies in hypothesis
    step_mask = pm.Data("step_mask", np.zeros((1, MAX_N)))  # example k exists
    group_size = pm.Data("group_size", np.ones((1, G_MAX)))  # width of the group's intervals
    group_count = pm.Data("group_count", np.ones((1, G_MAX)))  # intervals in the group (1 for padding)
    group_present = pm.Data("group_present", np.zeros((1, G_MAX)))  # 1 for a real (non-padding) group
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))  # set of each distinct (set, target)
    pair_target = pm.Data("pair_target", np.zeros(1, dtype="int64"))  # target - 1
    pair_in_group = pm.Data("pair_in_group", np.zeros((1, G_MAX)))  # share of group's intervals containing target
    row_pair = pm.Data("row_pair", np.zeros(1, dtype="int64"))
    new_coinc = pm.Data("new_coinc", np.zeros((1, MAX_N)))  # example k newly reveals a shared digit
    n_explained = pm.Data("n_explained", np.zeros((1, MAX_N, N_HYP)))  # examples 1..k in hypothesis
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pt.exp(log_sigma)
    # Log prior weight of surface-digit rules relative to arithmetic rules.
    log_digit_w = pm.Normal("log_digit_w", mu=0.0, sigma=1.5)
    rule_lw = log_digit_w * pt.as_tensor(IS_DIGIT_RULE)
    rule_log_prior = rule_lw - pt.logsumexp(rule_lw)

    # Gamma-width interval prior (fitted shape; the incumbent fixes shape = 2),
    # summed over the intervals in each group.
    log_shape = pm.Normal("log_shape", mu=np.log(2.0), sigma=0.7)
    shape_m1 = pt.exp(log_shape) - 1.0
    log_norm = pt.logsumexp(LOG_WIDTH_COUNT + shape_m1 * np.log(WIDTHS) - WIDTHS / sigma)
    n_sets = group_size.shape[0]
    group_log_prior = (shape_m1 * pt.log(group_size) - group_size / sigma + pt.log(group_count) - log_norm
                       - 1e4 * (1 - group_present))
    log_prior = pt.concatenate([
        pt.zeros((n_sets, N_RULES)) + pt.log(lambda_0) + rule_log_prior[None, :],
        pt.log1p(-lambda_0) + group_log_prior,
    ], axis=1)  # (sets, hyp)

    # Probability of keeping a guess that the new example fits.
    stay = pm.Beta("stay", 2, 2)
    # Probability that a newly noticed coincidence makes people abandon a fitting guess.
    disrupt = pm.Beta("disrupt", 2, 2)
    # Per-confirming-example probability of writing off a contradicting example.
    tolerate = pm.Beta("tolerate", 2, 6)

    # Fitted trust that examples come from the concept.
    alpha = pm.Beta("alpha", 9, 1)
    hyp_size = pt.concatenate([pt.zeros((n_sets, N_RULES)) + pt.as_tensor(RULE_SIZE)[None, :], group_size], axis=1)
    log_in = pt.log(alpha / hyp_size + (1 - alpha) / 100.0)[:, None, :]  # (sets, 1, hyp)
    log_out = pt.log1p(-alpha) - np.log(100.0)
    step_ll = step_mask[:, :, None] * (contains * log_in + (1 - contains) * log_out)

    # Coincidence noticing: a digit rule is available at step k only if at least
    # two of examples 1..k have that digit; otherwise its prior is scaled by
    # exp(log_unnoticed).
    log_unnoticed = pm.Normal("log_unnoticed", mu=0.0, sigma=1.5)
    n_fit = n_explained[:, :, :N_RULES]
    unnoticed = pt.as_tensor(IS_DIGIT_RULE)[None, None, :] * pt.lt(n_fit, 1.5)
    notice_lp = pt.concatenate([log_unnoticed * unnoticed, pt.zeros((n_sets, MAX_N, G_MAX))], axis=2)

    # Rethinking at step k: example k always used, each earlier example recalled
    # with probability `recall`; posterior averaged over the recalled subsets.
    recall = pm.Beta("recall", 3, 2)
    base = log_prior[:, None, :] + notice_lp  # (sets, steps, hyp)
    posts = [pt.special.softmax(base[:, 0, :] + step_ll[:, 0, :], axis=1)]
    for k in range(1, MAX_N):
        acc = 0.0
        for mask_bits in range(1 << k):
            members = [j for j in range(k) if (mask_bits >> j) & 1]
            w = recall ** len(members) * (1 - recall) ** (k - len(members))
            ll = base[:, k, :] + step_ll[:, k, :]
            for j in members:
                ll = ll + step_ll[:, j, :]
            acc = acc + w * pt.special.softmax(ll, axis=1)
        posts.append(acc)
    post = pt.stack(posts, axis=1)  # (sets, steps, hyp)

    q = post[:, 0, :]
    for k in range(1, MAX_N):
        stay_k = stay * (1 - disrupt * new_coinc[:, k])[:, None]
        tol_k = 1 - pt.pow(1 - tolerate, n_explained[:, k - 1, :])
        kept = stay_k * q * (contains[:, k, :] + tol_k * (1 - contains[:, k, :]))
        updated = kept + (1 - kept.sum(axis=1, keepdims=True)) * post[:, k, :]
        m = step_mask[:, k][:, None]
        q = m * updated + (1 - m) * q

    # Within a group all intervals are equally plausible, so a group's mass
    # covers the target in proportion to the share of its intervals containing it.
    gen_rule = pt.dot(q[:, :N_RULES], pt.as_tensor(RULE_MEMBERSHIP))  # (sets, 100)
    gen_pair = gen_rule[pair_set, pair_target] + (pt.take(q[:, N_RULES:], pair_set, axis=0) * pair_in_group).sum(axis=1)
    p_yes = pm.Deterministic("p_yes", pt.clip(gen_pair[row_pair], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    contains = np.ones((n_sets, MAX_N, N_HYP))
    mask = np.zeros((n_sets, MAX_N))
    gsize = np.ones((n_sets, G_MAX))
    gcount = np.ones((n_sets, G_MAX))
    present = np.zeros((n_sets, G_MAX))
    new_coinc = np.zeros((n_sets, MAX_N))
    groups = []
    digit_rows = RULE_MEMBERSHIP[IS_DIGIT_RULE > 0]
    for i, s in enumerate(unique):
        inside = np.stack([(LO <= x) & (HI >= x) for x in s], axis=1)  # (intervals, examples)
        code = (inside * (1 << np.arange(len(s)))).sum(axis=1)
        keys, group = np.unique(np.stack([INTERVAL_SIZE.astype(int), code], axis=1), axis=0, return_inverse=True)
        group = group.reshape(-1)
        n_g = len(keys)
        assert n_g <= G_MAX
        groups.append(group)
        gsize[i, :n_g] = keys[:, 0]
        gcount[i, :n_g] = np.bincount(group, minlength=n_g)
        present[i, :n_g] = 1.0
        for k in range(len(s)):
            contains[i, k, :N_RULES] = RULE_MEMBERSHIP[:, s[k] - 1]
            contains[i, k, N_RULES:] = 0.0
            contains[i, k, N_RULES:N_RULES + n_g] = (keys[:, 1] >> k) & 1
            mask[i, k] = 1.0
        for k in range(1, len(s)):
            # a digit rule that example k has and exactly one earlier example has
            before = digit_rows[:, [x - 1 for x in s[:k]]].sum(axis=1)
            if np.any((digit_rows[:, s[k] - 1] > 0) & (before == 1)):
                new_coinc[i, k] = 1.0
    pairs = list(dict.fromkeys(zip([index[s] for s in sets], targets)))
    pair_index = {p: j for j, p in enumerate(pairs)}
    in_group = np.zeros((len(pairs), G_MAX))
    for j, (i, t) in enumerate(pairs):
        hit = (LO <= t) & (HI >= t)
        in_group[j] = np.bincount(groups[i][hit], minlength=G_MAX)[:G_MAX] / gcount[i]
    return {
        "contains": contains,
        "step_mask": mask,
        "group_size": gsize,
        "group_count": gcount,
        "group_present": present,
        "pair_set": np.array([p[0] for p in pairs], dtype="int64"),
        "pair_target": np.array([p[1] - 1 for p in pairs], dtype="int64"),
        "pair_in_group": in_group,
        "row_pair": np.array([pair_index[(index[s], t)] for s, t in zip(sets, targets)], dtype="int64"),
        "new_coinc": new_coinc,
        "n_explained": np.cumsum(mask[:, :, None] * contains, axis=1),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
