"""Log-ratio similarity gradient on an inferred concept.

People infer the program's concept by Bayesian concept learning from all the
examples together (one of 45 rules, or an interval of 1..100 with a gamma-width
prior of fitted shape; size principle with fitted trust `alpha`; digit rules
discounted and salient only once two examples share the digit). Their
willingness to say the program would produce a target in that concept then
follows its similarity to the nearest example on a logarithmically compressed
mental number line: similarity is the ratio of the two numbers, so the
distance is |log t - log x|, and willingness falls off as
exp(-falloff * d ** 2) with `falloff` (1 / reach ** 2) fitted. The same numerical gap is a small
step among large numbers (90 after 80) but a long leap among small ones
(12 after 2); a doubling or halving (32 or 8 after 16) is the same step at any
magnitude.
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

# Bound per-iteration work (the posterior is well identified; deep trees only
# cost time).
SAMPLER_SETTINGS = {"max_treedepth": 6}

with pm.Model() as model:
    group_size = pm.Data("group_size", np.ones((1, G_MAX)))  # width of the group's intervals
    group_log_size = pm.Data("group_log_size", np.zeros((1, G_MAX)))  # log width
    group_base = pm.Data("group_base", np.zeros((1, G_MAX)))  # log intervals in group (-1e4 for padding)
    n_seen = pm.Data("n_seen", np.ones(1))  # examples shown
    n_explained = pm.Data("n_explained", np.zeros((1, N_HYP)))  # examples in hypothesis
    unnoticed = pm.Data("unnoticed", np.zeros((1, N_HYP)))  # digit rule not a noticed coincidence
    gen_index = pm.Data("gen_index", np.zeros(1, dtype="int64"))  # flat (set, hyp) index of each covering hyp
    gen_weight = pm.Data("gen_weight", np.zeros(1))  # share of that hyp's intervals containing the target
    pair_end = pm.Data("pair_end", np.ones(1, dtype="int64"))  # cumulative entry count per (set, target)
    pair_dist = pm.Data("pair_dist", np.zeros(1))  # log-ratio distance to nearest example
    row_pair = pm.Data("row_pair", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    # Exponential decay rate of the range-width prior (1 / typical width).
    width_rate = pm.HalfNormal("width_rate", sigma=0.1)
    # Log prior weight of surface-digit rules relative to arithmetic rules.
    log_digit_w = pm.Normal("log_digit_w", mu=0.0, sigma=1.5)
    rule_lw = log_digit_w * pt.as_tensor(IS_DIGIT_RULE)
    rule_log_prior = rule_lw - pt.logsumexp(rule_lw)

    # Gamma-width interval prior (fitted shape), summed over each group's intervals.
    log_shape = pm.Normal("log_shape", mu=np.log(2.0), sigma=0.7)
    shape_m1 = pt.exp(log_shape) - 1.0
    log_norm = pt.logsumexp(LOG_WIDTH_COUNT + shape_m1 * np.log(WIDTHS) - WIDTHS * width_rate)
    n_sets = group_size.shape[0]
    group_log_prior = shape_m1 * group_log_size - group_size * width_rate + group_base - log_norm
    log_prior = pt.concatenate([
        pt.zeros((n_sets, N_RULES)) + pt.log(lambda_0) + rule_log_prior[None, :],
        pt.log1p(-lambda_0) + group_log_prior,
    ], axis=1)  # (sets, hyp)

    # Fitted trust that examples come from the concept.
    alpha = pm.Beta("alpha", 9, 1)
    hyp_size = pt.concatenate([pt.zeros((n_sets, N_RULES)) + pt.as_tensor(RULE_SIZE)[None, :], group_size], axis=1)
    log_in = pt.log(alpha / hyp_size + (1 - alpha) / 100.0)  # (sets, hyp)
    log_out = pt.log1p(-alpha) - np.log(100.0)
    log_lik = n_explained * log_in + (n_seen[:, None] - n_explained) * log_out

    # Coincidence noticing: a digit rule is salient only if at least two of the
    # examples have that digit; otherwise its prior is scaled by exp(log_unnoticed).
    log_unnoticed = pm.Normal("log_unnoticed", mu=0.0, sigma=1.5)
    post = pt.special.softmax(log_prior + log_unnoticed * unnoticed + log_lik, axis=1)

    # Probability that the concept contains each (set, target): a segment sum
    # over the hypotheses covering it (a group covers it in proportion to the
    # share of its intervals that contain it).
    contrib = pt.cumsum(post.flatten()[gen_index] * gen_weight)
    padded = pt.concatenate([pt.zeros(1), contrib])
    gen_pair = padded[pair_end] - padded[pt.concatenate([pt.zeros(1, dtype="int64"), pair_end[:-1]])]
    # Extrapolation reluctance: willingness to extend the concept to a target
    # falls off (Gaussian) with its log-ratio distance from the nearest example.
    # `falloff` = 1 / reach ** 2, on the squared log-ratio distance.
    falloff = pm.HalfNormal("falloff", sigma=1.0)
    extend = pt.exp(-falloff * pt.sqr(pair_dist))
    p_yes = pm.Deterministic("p_yes", pt.clip((gen_pair * extend)[row_pair], 1e-6, 1 - 1e-6))
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
    groups = []
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
    pairs = list(dict.fromkeys(zip([index[s] for s in sets], targets)))
    pair_index = {p: j for j, p in enumerate(pairs)}
    gen_index, gen_weight, pair_end = [], [], []
    for i, t in pairs:
        hit = (LO <= t) & (HI >= t)
        share = np.bincount(groups[i][hit], minlength=G_MAX)[:G_MAX] / gcount[i]
        rules = np.flatnonzero(RULE_MEMBERSHIP[:, t - 1])
        grp = np.flatnonzero(share)
        gen_index.extend(i * N_HYP + rules)
        gen_weight.extend(np.ones(len(rules)))
        gen_index.extend(i * N_HYP + N_RULES + grp)
        gen_weight.extend(share[grp])
        pair_end.append(len(gen_index))
    n_explained = np.cumsum(mask[:, :, None] * contains, axis=1)
    unnoticed = np.zeros((n_sets, N_HYP))
    unnoticed[:, :N_RULES] = IS_DIGIT_RULE[None, :] * (n_explained[:, -1, :N_RULES] < 1.5)
    return {
        "group_size": gsize,
        "group_log_size": np.log(gsize),
        "group_base": np.log(gcount) - 1e4 * (1 - present),
        "n_seen": mask.sum(axis=1),
        "n_explained": n_explained[:, -1, :],
        "unnoticed": unnoticed,
        "gen_index": np.array(gen_index, dtype="int64"),
        "gen_weight": np.array(gen_weight, dtype="float64"),
        "pair_end": np.array(pair_end, dtype="int64"),
        "pair_dist": np.array([min(abs(np.log(t) - np.log(x)) for x in unique[i]) for i, t in pairs]),
        "row_pair": np.array([pair_index[(index[s], t)] for s, t in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
