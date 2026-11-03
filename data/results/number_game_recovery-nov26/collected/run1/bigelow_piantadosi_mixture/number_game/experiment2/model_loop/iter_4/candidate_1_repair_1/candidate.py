"""Conservative range stretching: minimal revision of a contradicted range guess.

People hold a single working guess about the concept (one of 45 rules or an
interval of 1..100 with an Erlang-width interval prior, digit rules discounted
by one fitted weight and salient only once their digit is noticed as shared by
two examples, fitted trust `alpha` in the examples), formed from the first
example by Bayesian concept learning and kept with probability `stay` while
later examples fit it; a newly noticed digit coincidence makes them drop a
fitting guess with probability `disrupt` (as in the current best model). The
one change: when a new example falls outside their current range guess, with
fitted probability `stretch` they make the minimal revision -- stretch the
range just far enough to include the new example -- and only otherwise
reconsider from all examples so far. So the range guessed first, and the order
of the examples, shape the final range.

A stretched range always ends at an example, so its mass lives on "lines":
L[j][a] for the range [a, x_j] and R[j][b] for [x_j, b]. Which intervals are
excluded by each example, and the line cell each is stretched onto, is
precomputed once per example set (prepare_observed).
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
N_INT = len(LO)
INTERVAL_SIZE = (HI - LO + 1).astype("float64")
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
N_HYP = N_RULES + G_MAX

with pm.Model() as model:
    contains = pm.Data("contains", np.ones((1, MAX_N, N_HYP)))  # example k lies in rule / interval group
    step_mask = pm.Data("step_mask", np.zeros((1, MAX_N)))
    group_size = pm.Data("group_size", np.ones((1, G_MAX)))
    group_count = pm.Data("group_count", np.ones((1, G_MAX)))  # intervals in the group (1 for padding)
    group_present = pm.Data("group_present", np.zeros((1, G_MAX)))
    interval_group = pm.Data("interval_group", np.zeros((1, N_INT), dtype="int64"))  # flat group id
    new_coinc = pm.Data("new_coinc", np.zeros((1, MAX_N)))
    le = pm.Data("le", np.ones((1, MAX_N, 100)))  # number <= example k
    ge = pm.Data("ge", np.ones((1, MAX_N, 100)))  # number >= example k
    onehot = pm.Data("onehot", np.zeros((1, MAX_N, 100)))  # example k's position
    # For step k: each interval the example excludes (flat group slot) and its stretched line cell.
    src = [pm.Data(f"stretch_src_{k}", np.zeros(1, dtype="int64")) for k in range(1, MAX_N)]
    dst = [pm.Data(f"stretch_dst_{k}", np.zeros(1, dtype="int64")) for k in range(1, MAX_N)]
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    n_sets = interval_group.shape[0]

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pt.exp(log_sigma)
    log_digit_w = pm.Normal("log_digit_w", mu=0.0, sigma=1.5)
    rule_lw = log_digit_w * pt.as_tensor(IS_DIGIT_RULE)
    rule_log_prior = rule_lw - pt.logsumexp(rule_lw)

    size = pt.as_tensor(INTERVAL_SIZE)
    log_norm = pt.logsumexp(pt.log(size) - size / sigma)
    group_log_prior = (pt.log(group_size) - group_size / sigma + pt.log(group_count) - log_norm
                       - 1e4 * (1 - group_present))
    log_prior = pt.concatenate([
        pt.zeros((n_sets, N_RULES)) + pt.log(lambda_0) + rule_log_prior[None, :],
        pt.log1p(-lambda_0) + group_log_prior,
    ], axis=1)

    stay = pm.Beta("stay", 2, 2)
    disrupt = pm.Beta("disrupt", 2, 2)
    # Probability of stretching a contradicted range guess just enough to include the new example.
    stretch = pm.Beta("stretch", 2, 2)

    alpha = pm.Beta("alpha", 9, 1)
    hyp_size = pt.concatenate([pt.zeros((n_sets, N_RULES)) + pt.as_tensor(RULE_SIZE)[None, :], group_size], axis=1)
    log_in = pt.log(alpha / hyp_size + (1 - alpha) / 100.0)[:, None, :]
    log_out = pt.log1p(-alpha) - np.log(100.0)
    step_ll = step_mask[:, :, None] * (contains * log_in + (1 - contains) * log_out)
    prefix_log_lik = pt.cumsum(step_ll, axis=1)

    log_unnoticed = pm.Normal("log_unnoticed", mu=0.0, sigma=1.5)
    n_fit = pt.cumsum(step_mask[:, :, None] * contains[:, :, :N_RULES], axis=1)
    unnoticed = pt.as_tensor(IS_DIGIT_RULE)[None, None, :] * pt.lt(n_fit, 1.5)
    notice_lp = pt.concatenate([log_unnoticed * unnoticed, pt.zeros((n_sets, MAX_N, G_MAX))], axis=2)

    post = pt.special.softmax(log_prior[:, None, :] + notice_lp + prefix_log_lik, axis=2)

    q = post[:, 0, :]
    lines_l, lines_r, ends = [], [], []  # line mass and the example each line ends at
    for k in range(1, MAX_N):
        m = step_mask[:, k][:, None]
        stay_k = (stay * (1 - disrupt * new_coinc[:, k]))[:, None]
        le_k, ge_k, oh_k = le[:, k], ge[:, k], onehot[:, k]
        kept = stay_k * q * contains[:, k, :]
        # Ranges wholly below x become [lo, x]; wholly above become [x, hi] (slot n*G_MAX is empty).
        share = pt.concatenate([(q[:, N_RULES:] / group_count).flatten(), pt.zeros(1)])
        moved = pt.inc_subtensor(pt.zeros(n_sets * 200)[dst[k - 1]], share[src[k - 1]]).reshape((n_sets, 200))
        mv_l = stretch * moved[:, :100]
        mv_r = stretch * moved[:, 100:]
        kept_lines = pt.zeros((n_sets, 1))
        new_l, new_r = [], []
        for L, R, j in zip(lines_l, lines_r, ends):
            c = (oh_k * le[:, j]).sum(axis=1, keepdims=True)  # x_k <= x_j
            d = (oh_k * ge[:, j]).sum(axis=1, keepdims=True)  # x_k >= x_j
            kl = stay_k * L * le_k * c
            kr = stay_k * R * ge_k * d
            mv_l = mv_l + stretch * ((1 - c) * L + d * (R * (1 - ge_k)).sum(axis=1, keepdims=True) * onehot[:, j])
            mv_r = mv_r + stretch * ((1 - d) * R + c * (L * (1 - le_k)).sum(axis=1, keepdims=True) * onehot[:, j])
            kept_lines = kept_lines + kl.sum(axis=1, keepdims=True) + kr.sum(axis=1, keepdims=True)
            new_l.append(m * kl + (1 - m) * L)
            new_r.append(m * kr + (1 - m) * R)
        rest = (1 - kept.sum(axis=1, keepdims=True) - kept_lines
                - mv_l.sum(axis=1, keepdims=True) - mv_r.sum(axis=1, keepdims=True))
        q = m * (kept + rest * post[:, k, :]) + (1 - m) * q
        lines_l = new_l + [m * mv_l]
        lines_r = new_r + [m * mv_r]
        ends.append(k)

    # Spread each group's final mass evenly over its member intervals.
    share = (q[:, N_RULES:] / group_count).flatten()
    q_int = share[interval_group.flatten()].reshape((n_sets, N_INT))
    gen = pt.dot(q[:, :N_RULES], pt.as_tensor(RULE_MEMBERSHIP)) + pt.dot(q_int, pt.as_tensor(INTERVAL_MEMBERSHIP))
    for L, R, j in zip(lines_l, lines_r, ends):
        gen = gen + pt.cumsum(L, axis=1) * le[:, j] + pt.cumsum(R[:, ::-1], axis=1)[:, ::-1] * ge[:, j]
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
    gcount = np.ones((n_sets, G_MAX))
    present = np.zeros((n_sets, G_MAX))
    igroup = np.zeros((n_sets, N_INT), dtype="int64")
    new_coinc = np.zeros((n_sets, MAX_N))
    le = np.ones((n_sets, MAX_N, 100))
    ge = np.ones((n_sets, MAX_N, 100))
    onehot = np.zeros((n_sets, MAX_N, 100))
    src = {k: [np.array([n_sets * G_MAX])] for k in range(1, MAX_N)}  # one dummy entry on the empty slot
    dst = {k: [np.array([0])] for k in range(1, MAX_N)}
    digit_rows = RULE_MEMBERSHIP[IS_DIGIT_RULE > 0]
    for i, s in enumerate(unique):
        inside = np.stack([(LO <= x) & (HI >= x) for x in s], axis=1)
        code = (inside * (1 << np.arange(len(s)))).sum(axis=1)
        keys, group, counts = np.unique(np.stack([INTERVAL_SIZE.astype(int), code], axis=1), axis=0,
                                        return_inverse=True, return_counts=True)
        group = group.reshape(-1)
        n_g = len(keys)
        assert n_g <= G_MAX
        igroup[i] = i * G_MAX + group
        gsize[i, :n_g] = keys[:, 0]
        gcount[i, :n_g] = counts
        present[i, :n_g] = 1.0
        for k, x in enumerate(s):
            contains[i, k, :N_RULES] = RULE_MEMBERSHIP[:, x - 1]
            contains[i, k, N_RULES:] = 0.0
            contains[i, k, N_RULES:N_RULES + n_g] = (keys[:, 1] >> k) & 1
            mask[i, k] = 1.0
            le[i, k] = NUMBERS <= x
            ge[i, k] = NUMBERS >= x
            onehot[i, k, x - 1] = 1.0
            if k > 0:
                below, above = HI < x, LO > x
                src[k].append(i * G_MAX + group[below])
                dst[k].append(i * 200 + LO[below] - 1)
                src[k].append(i * G_MAX + group[above])
                dst[k].append(i * 200 + 100 + HI[above] - 1)
                before = digit_rows[:, [y - 1 for y in s[:k]]].sum(axis=1)
                if np.any((digit_rows[:, x - 1] > 0) & (before == 1)):
                    new_coinc[i, k] = 1.0
    out = {
        "contains": contains,
        "step_mask": mask,
        "group_size": gsize,
        "group_count": gcount,
        "group_present": present,
        "interval_group": igroup,
        "new_coinc": new_coinc,
        "le": le,
        "ge": ge,
        "onehot": onehot,
        "row_set": np.array([index[s] for s in sets], dtype="int64"),
        "row_target": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
    for k in range(1, MAX_N):
        out[f"stretch_src_{k}"] = np.concatenate(src[k]).astype("int64")
        out[f"stretch_dst_{k}"] = np.concatenate(dst[k]).astype("int64")
    return out
