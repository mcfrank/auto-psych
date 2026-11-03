"""Disjunctive rule concepts (concept-space change to the sticky-guess learner).

People's space of candidate concepts includes not only single rules and
numerical ranges, but also "either-or" disjunctions of two familiar rules
("a square or ends in 1", "even or ends in 5"), entertained with a fitted,
complexity-penalised prior share. When the examples do not all fit one rule
people can explain them with a two-rule union instead of writing off an
example as a stray, and then endorse targets in either rule. Otherwise the
learner is the sticky-guess learner: a single working guess drawn after the
first example from the Bayesian posterior (size principle, fitted sampling
trust alpha), kept with probability `stay` while later examples fit it,
redrawn from the posterior given all examples so far when contradicted;
Erlang prior on range width; a fitted discount on digit rules.

A union "A or B" has prior lambda_0 * u * pi(A) pi(B) / Z over the pairs of
rules neither of which contains the other (pi = within-rule prior, with the
digit discount); single rules keep lambda_0 * (1 - u) * pi(R).

For speed, intervals are grouped per example set by (size, which examples
they contain); all members of a group share prior and likelihood, so the
fraction of a group's intervals containing a target is precomputed, and p_yes
is computed once per distinct (set, target) pair.
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


def union_pairs():
    pairs, seen = [], set()
    for a in range(N_RULES):
        for b in range(a + 1, N_RULES):
            ma, mb = RULE_MEMBERSHIP[a] > 0, RULE_MEMBERSHIP[b] > 0
            if (ma <= mb).all() or (mb <= ma).all():
                continue  # the union is just one of the rules
            key = tuple(np.flatnonzero(ma | mb))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((a, b))
    return np.array(pairs, dtype="int64")


UNION_PAIRS = union_pairs()
UNION_MEMBERSHIP = np.maximum(RULE_MEMBERSHIP[UNION_PAIRS[:, 0]], RULE_MEMBERSHIP[UNION_PAIRS[:, 1]])
N_RU = N_RULES + len(UNION_PAIRS)  # rule-like hypotheses (singles then unions)
N_HYP = N_RU + G_MAX
RU_MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, UNION_MEMBERSHIP])
RU_SIZE = RU_MEMBERSHIP.sum(axis=1)
LOG_NORM_SIZES = np.arange(1, 101, dtype="float64")
LOG_NORM_COUNTS = 101.0 - LOG_NORM_SIZES  # number of intervals of each size

with pm.Model() as model:
    contains = pm.Data("contains", np.ones((1, MAX_N, N_HYP)))  # example k lies in hypothesis
    hits = pm.Data("hits", np.zeros((1, MAX_N, N_HYP)))  # how many of examples 1..k lie in it
    step_mask = pm.Data("step_mask", np.zeros((1, MAX_N)))  # example k exists
    group_size = pm.Data("group_size", np.ones((1, G_MAX)))
    group_log_count = pm.Data("group_log_count", np.full((1, G_MAX), -1e4))  # -1e4 for padding
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))  # set of each distinct (set, target)
    pair_target = pm.Data("pair_target", np.zeros(1, dtype="int64"))  # its target - 1
    pair_frac = pm.Data("pair_frac", np.zeros((1, G_MAX)))  # share of each group's intervals holding it
    row_pair = pm.Data("row_pair", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pt.exp(log_sigma)
    # Log prior weight of surface-digit rules relative to arithmetic rules.
    log_digit_w = pm.Normal("log_digit_w", mu=0.0, sigma=1.5)
    rule_lw = log_digit_w * pt.as_tensor(IS_DIGIT_RULE)
    rule_log_prior = rule_lw - pt.logsumexp(rule_lw)
    # Prior share of the rule mass given to two-rule disjunctions (logit scale).
    logit_u = pm.Normal("logit_u", mu=-3.0, sigma=1.5)
    union_lw = rule_log_prior[UNION_PAIRS[:, 0]] + rule_log_prior[UNION_PAIRS[:, 1]]
    union_log_prior = union_lw - pt.logsumexp(union_lw)
    ru_log_prior = pt.concatenate([pt.log(pt.sigmoid(-logit_u)) + rule_log_prior,
                                   pt.log(pt.sigmoid(logit_u)) + union_log_prior])

    # Erlang(2) prior on interval width, normalised over all intervals.
    log_norm = pt.logsumexp(np.log(LOG_NORM_COUNTS) + np.log(LOG_NORM_SIZES) - LOG_NORM_SIZES / sigma)
    group_log_prior = group_log_count + pt.log(group_size) - group_size / sigma - log_norm
    n_sets = contains.shape[0]
    log_prior = pt.concatenate([
        pt.zeros((n_sets, N_RU)) + pt.log(lambda_0) + ru_log_prior[None, :],
        pt.log1p(-lambda_0) + group_log_prior,
    ], axis=1)  # (sets, hyp)

    stay = pm.Beta("stay", 2, 2)
    alpha = pm.Beta("alpha", 9, 1)  # trust that examples come from the concept
    hyp_size = pt.concatenate([pt.zeros((n_sets, N_RU)) + pt.as_tensor(RU_SIZE)[None, :], group_size], axis=1)
    # Each example outside a hypothesis contributes the same log((1-alpha)/100) to
    # every hypothesis, which cancels in the posterior; only the hits matter.
    log_ratio = pt.log1p(alpha / (1 - alpha) * 100.0 / hyp_size)[:, None, :]
    post = pt.special.softmax(log_prior[:, None, :] + hits * log_ratio, axis=2)  # (sets, steps, hyp)

    q = post[:, 0, :]
    for k in range(1, MAX_N):
        kept = stay * q * contains[:, k, :]
        updated = kept + (1 - kept.sum(axis=1, keepdims=True)) * post[:, k, :]
        m = step_mask[:, k][:, None]
        q = m * updated + (1 - m) * q

    gen_ru = pt.dot(q[:, :N_RU], pt.as_tensor(RU_MEMBERSHIP))  # (sets, targets)
    p_pair = gen_ru[pair_set, pair_target] + (q[:, N_RU:][pair_set] * pair_frac).sum(axis=1)
    p_yes = pm.Deterministic("p_yes", pt.clip(p_pair[row_pair], 1e-6, 1 - 1e-6))
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
    glogc = np.full((n_sets, G_MAX), -1e4)
    group_of = []
    for i, s in enumerate(unique):
        inside = np.stack([(LO <= x) & (HI >= x) for x in s], axis=1)
        code = (inside * (1 << np.arange(len(s)))).sum(axis=1)
        keys, group, counts = np.unique(np.stack([INTERVAL_SIZE.astype(int), code], axis=1), axis=0,
                                        return_inverse=True, return_counts=True)
        group = group.reshape(-1)
        n_g = len(keys)
        assert n_g <= G_MAX
        group_of.append((group, counts))
        gsize[i, :n_g] = keys[:, 0]
        glogc[i, :n_g] = np.log(counts)
        for k in range(len(s)):
            contains[i, k, :N_RU] = RU_MEMBERSHIP[:, s[k] - 1]
            contains[i, k, N_RU:] = 0.0
            contains[i, k, N_RU:N_RU + n_g] = (keys[:, 1] >> k) & 1
            mask[i, k] = 1.0
    pairs = list(dict.fromkeys(zip(sets, targets)))
    pindex = {p: j for j, p in enumerate(pairs)}
    pset = np.array([index[s] for s, _ in pairs], dtype="int64")
    frac = np.zeros((len(pairs), G_MAX))
    for j, (s, t) in enumerate(pairs):
        group, counts = group_of[pset[j]]
        has_t = ((LO <= t) & (HI >= t)).astype("float64")
        frac[j, :len(counts)] = np.bincount(group, weights=has_t, minlength=len(counts)) / counts
    return {
        "contains": contains,
        "hits": np.cumsum(contains * mask[:, :, None], axis=1),
        "step_mask": mask,
        "group_size": gsize,
        "group_log_count": glogc,
        "pair_set": pset,
        "pair_target": np.array([t - 1 for _, t in pairs], dtype="int64"),
        "pair_frac": frac,
        "row_pair": np.array([pindex[p] for p in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
