"""Refinement of misfit_tolerant_coincidence_rethink: tolerance of a misfitting
example scales with the evidence behind the guess.

As in the incumbent, people hold a single working guess (one of 45 rules or an
interval of 1..100 with an Erlang-width prior; digit rules discounted and
salient only once two examples share the digit; fitted trust `alpha`), formed
from the first example, kept with probability `stay` while later examples fit
it, dropped with probability `disrupt` when a new example reveals a shared
digit, and a contradicted guess can be kept by dismissing the misfit as a
stray. The one change: each earlier example the guess explains independently
gives probability `tau` of dismissing the misfit, so
tolerance = 1 - (1 - tau)^(number of earlier examples in the guess). A guess
backed by 77 and 17 ("ends in 7") survives the outlier 8 far more often than one
backed by a single example. (For speed, the incumbent's weakly constrained
power-law prior on interval endpoints is dropped: intervals of the same width
are equally plausible wherever they lie.)
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
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
N_HYP = N_RULES + G_MAX

with pm.Model() as model:
    contains = pm.Data("contains", np.ones((1, MAX_N, N_HYP)))  # example k lies in hypothesis
    step_mask = pm.Data("step_mask", np.zeros((1, MAX_N)))  # example k exists
    group_size = pm.Data("group_size", np.ones((1, G_MAX)))  # interval width of each group
    group_count = pm.Data("group_count", np.ones((1, G_MAX)))  # number of intervals in each group
    group_present = pm.Data("group_present", np.zeros((1, G_MAX)))  # 1 for a real (non-padding) group
    new_coinc = pm.Data("new_coinc", np.zeros((1, MAX_N)))  # example k newly reveals a shared digit
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))  # set of each distinct (set, target) pair
    pair_rule_in = pm.Data("pair_rule_in", np.zeros((1, N_RULES)))  # rule contains the pair's target
    pair_group_frac = pm.Data("pair_group_frac", np.zeros((1, G_MAX)))  # share of group's intervals containing it
    row_pair = pm.Data("row_pair", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pt.exp(log_sigma)
    # Log prior weight of surface-digit rules relative to arithmetic rules.
    log_digit_w = pm.Normal("log_digit_w", mu=0.0, sigma=1.5)
    rule_lw = log_digit_w * pt.as_tensor(IS_DIGIT_RULE)
    rule_log_prior = rule_lw - pt.logsumexp(rule_lw)

    size = pt.as_tensor(INTERVAL_SIZE)
    log_norm = pt.logsumexp(pt.log(size) - size / sigma)
    group_log_prior = (pt.log(group_count) + pt.log(group_size) - group_size / sigma - log_norm
                       - 1e4 * (1 - group_present))
    n_sets = group_size.shape[0]
    log_prior = pt.concatenate([
        pt.zeros((n_sets, N_RULES)) + pt.log(lambda_0) + rule_log_prior[None, :],
        pt.log1p(-lambda_0) + group_log_prior,
    ], axis=1)  # (sets, hyp)

    # Probability of keeping a guess that the new example fits.
    stay = pm.Beta("stay", 2, 2)
    # Probability that a newly noticed coincidence makes people abandon a fitting guess.
    disrupt = pm.Beta("disrupt", 2, 2)
    # Per-supporting-example chance of writing a contradicting example off as a stray.
    tau = pm.Beta("tau", 1, 4)
    n_prev = pt.cumsum(step_mask[:, :, None] * contains, axis=1)  # examples 1..k in each hypothesis

    # Fitted trust that examples come from the concept.
    alpha = pm.Beta("alpha", 9, 1)
    hyp_size = pt.concatenate([pt.zeros((n_sets, N_RULES)) + pt.as_tensor(RULE_SIZE)[None, :], group_size], axis=1)
    log_in = pt.log(alpha / hyp_size + (1 - alpha) / 100.0)[:, None, :]  # (sets, 1, hyp)
    log_out = pt.log1p(-alpha) - np.log(100.0)
    step_ll = step_mask[:, :, None] * (contains * log_in + (1 - contains) * log_out)
    prefix_log_lik = pt.cumsum(step_ll, axis=1)  # log lik of examples 1..k

    # Coincidence noticing: a digit rule is available at step k only if at least
    # two of examples 1..k have that digit; otherwise its prior is scaled by
    # exp(log_unnoticed).
    log_unnoticed = pm.Normal("log_unnoticed", mu=0.0, sigma=1.5)
    unnoticed = pt.as_tensor(IS_DIGIT_RULE)[None, None, :] * pt.lt(n_prev[:, :, :N_RULES], 1.5)
    notice_lp = pt.concatenate([log_unnoticed * unnoticed, pt.zeros((n_sets, MAX_N, G_MAX))], axis=2)

    post = pt.special.softmax(log_prior[:, None, :] + notice_lp + prefix_log_lik, axis=2)  # (sets, steps, hyp)

    log_keep_miss = pt.log1p(-tau)
    q = post[:, 0, :]
    for k in range(1, MAX_N):
        stay_k = stay * (1 - disrupt * new_coinc[:, k])[:, None]
        tolerate_k = 1 - pt.exp(n_prev[:, k - 1, :] * log_keep_miss)
        kept = stay_k * q * (contains[:, k, :] + tolerate_k * (1 - contains[:, k, :]))
        updated = kept + (1 - kept.sum(axis=1, keepdims=True)) * post[:, k, :]
        m = step_mask[:, k][:, None]
        q = m * updated + (1 - m) * q

    # Probability of "yes" at each distinct (set, target) pair: rule mass on rules
    # containing the target, plus each interval group's mass spread evenly over
    # its intervals, of which a known share contains the target.
    q_pair = q[pair_set]
    gen = (q_pair[:, :N_RULES] * pair_rule_in).sum(axis=1) + (q_pair[:, N_RULES:] * pair_group_frac).sum(axis=1)
    p_yes = pm.Deterministic("p_yes", pt.clip(gen[row_pair], 1e-6, 1 - 1e-6))
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
        groups.append(group)
        n_g = len(keys)
        assert n_g <= G_MAX
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
    pairs = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    upairs = list(dict.fromkeys(pairs))
    pindex = {p: j for j, p in enumerate(upairs)}
    rule_in = np.zeros((len(upairs), N_RULES))
    group_frac = np.zeros((len(upairs), G_MAX))
    for j, (i, t) in enumerate(upairs):
        rule_in[j] = RULE_MEMBERSHIP[:, t]
        hits = np.bincount(groups[i], weights=INTERVAL_MEMBERSHIP[:, t], minlength=G_MAX)
        group_frac[j] = hits / gcount[i]
    return {
        "contains": contains,
        "step_mask": mask,
        "group_size": gsize,
        "group_count": gcount,
        "group_present": present,
        "new_coinc": new_coinc,
        "pair_set": np.array([p[0] for p in upairs], dtype="int64"),
        "pair_rule_in": rule_in,
        "pair_group_frac": group_frac,
        "row_pair": np.array([pindex[p] for p in pairs], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
