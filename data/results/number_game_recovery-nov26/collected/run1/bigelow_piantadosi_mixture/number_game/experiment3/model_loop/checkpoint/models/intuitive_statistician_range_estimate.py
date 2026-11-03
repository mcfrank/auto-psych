"""Intuitive-statistician range estimate.

People do not generalise along the number line by weighing every possible
interval; when they think the program produces numbers from a range, they
estimate that range like an intuitive statistician from the smallest and largest
examples seen so far, extending it beyond each extreme by a margin
m = base + spread_share * span / n (span = max - min, n = examples seen), and
re-estimate it from all examples as each one arrives. Membership of a target in
the estimated range falls off smoothly (exponentially, scale m) with its
distance beyond the examples' span, so range generalisation is symmetric and order-free.

Rule guesses are formed and kept as in the sticky single-guess learner: one
working guess (one of 45 rules, digit rules discounted and salient only once two
examples share the digit, or "the range"), formed from the first example by
Bayesian concept learning with fitted trust `alpha`, kept with probability
`stay` while later examples fit it, dropped with probability `disrupt` when a
new example reveals a shared digit, and kept despite a contradicting example
with probability 1 - (1 - tolerate) ** (examples it explains). The range guess
always fits (it is re-estimated), and its size for the size principle is its
effective width span + 1 + 2 * m.
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


RULE_NAMES = list(rule_concepts().keys())
RULES = list(rule_concepts().values())
IS_DIGIT_RULE = np.array([n.startswith("ends in") or n.startswith("contains digit") for n in RULE_NAMES], dtype="float64")
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)

with pm.Model() as model:
    contains = pm.Data("contains", np.ones((1, MAX_N, N_RULES)))  # example k lies in rule
    step_mask = pm.Data("step_mask", np.zeros((1, MAX_N)))  # example k exists
    step_span = pm.Data("step_span", np.zeros((1, MAX_N)))  # span of examples 1..k / 100
    step_n = pm.Data("step_n", np.ones((1, MAX_N)))  # k
    new_coinc = pm.Data("new_coinc", np.zeros((1, MAX_N)))  # example k newly reveals a shared digit
    n_explained = pm.Data("n_explained", np.zeros((1, MAX_N, N_RULES)))  # examples 1..k in rule
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))
    pair_target = pm.Data("pair_target", np.zeros(1, dtype="int64"))  # target - 1
    pair_dist = pm.Data("pair_dist", np.zeros(1))  # distance beyond the full set's span / 100
    pair_span = pm.Data("pair_span", np.zeros(1))  # full set's span / 100
    pair_n = pm.Data("pair_n", np.ones(1))  # number of examples in the set
    row_pair = pm.Data("row_pair", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    log_digit_w = pm.Normal("log_digit_w", mu=0.0, sigma=1.5)
    rule_lw = log_digit_w * pt.as_tensor(IS_DIGIT_RULE)
    rule_log_prior = pt.log(lambda_0) + rule_lw - pt.logsumexp(rule_lw)

    stay = pm.Beta("stay", 2, 2)
    disrupt = pm.Beta("disrupt", 2, 2)
    tolerate = pm.Beta("tolerate", 2, 6)
    alpha = pm.Beta("alpha", 9, 1)

    # Intuitive range estimate: margin beyond each extreme (units of 100).
    log_base = pm.Normal("log_base", mu=np.log(0.1), sigma=1.0)
    log_share = pm.Normal("log_share", mu=0.0, sigma=1.0)
    base = pt.exp(log_base)
    share = pt.exp(log_share)

    n_sets = contains.shape[0]
    # Rules: likelihood of examples 1..k.
    log_in = pt.log(alpha / pt.as_tensor(RULE_SIZE) + (1 - alpha) / 100.0)[None, None, :]
    log_out = pt.log1p(-alpha) - np.log(100.0)
    step_ll = step_mask[:, :, None] * (contains * log_in + (1 - contains) * log_out)
    rule_prefix = pt.cumsum(step_ll, axis=1)  # (sets, steps, rules)

    # Range: re-estimated at step k from examples 1..k; all of them lie in it.
    margin_k = base + share * step_span / step_n
    size_k = pt.minimum(100.0 * (step_span + 2.0 * margin_k) + 1.0, 100.0)
    range_prefix = step_n * pt.log(alpha / size_k + (1 - alpha) / 100.0) + pt.log1p(-lambda_0)  # (sets, steps)

    log_unnoticed = pm.Normal("log_unnoticed", mu=0.0, sigma=1.5)
    unnoticed = pt.as_tensor(IS_DIGIT_RULE)[None, None, :] * pt.lt(n_explained, 1.5)
    rule_score = rule_log_prior[None, None, :] + log_unnoticed * unnoticed + rule_prefix
    post = pt.special.softmax(pt.concatenate([rule_score, range_prefix[:, :, None]], axis=2), axis=2)

    fits_all = pt.concatenate([contains, pt.ones((n_sets, MAX_N, 1))], axis=2)
    explained_all = pt.concatenate([n_explained, step_n[:, :, None]], axis=2)
    q = post[:, 0, :]
    for k in range(1, MAX_N):
        stay_k = stay * (1 - disrupt * new_coinc[:, k])[:, None]
        tol_k = 1 - pt.pow(1 - tolerate, explained_all[:, k - 1, :])
        kept = stay_k * q * (fits_all[:, k, :] + tol_k * (1 - fits_all[:, k, :]))
        updated = kept + (1 - kept.sum(axis=1, keepdims=True)) * post[:, k, :]
        m = step_mask[:, k][:, None]
        q = m * updated + (1 - m) * q

    gen_rule = pt.dot(q[:, :N_RULES], pt.as_tensor(RULE_MEMBERSHIP))  # (sets, 100)
    margin = base + share * pair_span / pair_n
    in_range = pt.exp(-pair_dist / margin)
    gen_pair = gen_rule[pair_set, pair_target] + q[pair_set, N_RULES] * in_range
    p_yes = pm.Deterministic("p_yes", pt.clip(gen_pair[row_pair], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    contains = np.ones((n_sets, MAX_N, N_RULES))
    mask = np.zeros((n_sets, MAX_N))
    span = np.zeros((n_sets, MAX_N))
    n = np.ones((n_sets, MAX_N))
    new_coinc = np.zeros((n_sets, MAX_N))
    digit_rows = RULE_MEMBERSHIP[IS_DIGIT_RULE > 0]
    for i, s in enumerate(unique):
        for k in range(MAX_N):
            kk = min(k, len(s) - 1)
            span[i, k] = (max(s[:kk + 1]) - min(s[:kk + 1])) / 100.0
            n[i, k] = kk + 1
        for k in range(len(s)):
            contains[i, k] = RULE_MEMBERSHIP[:, s[k] - 1]
            mask[i, k] = 1.0
        for k in range(1, len(s)):
            before = digit_rows[:, [x - 1 for x in s[:k]]].sum(axis=1)
            if np.any((digit_rows[:, s[k] - 1] > 0) & (before == 1)):
                new_coinc[i, k] = 1.0
    pairs = list(dict.fromkeys(zip([index[s] for s in sets], targets)))
    pair_index = {p: j for j, p in enumerate(pairs)}
    return {
        "contains": contains,
        "step_mask": mask,
        "step_span": span,
        "step_n": n,
        "new_coinc": new_coinc,
        "n_explained": np.cumsum(mask[:, :, None] * contains, axis=1),
        "pair_set": np.array([p[0] for p in pairs], dtype="int64"),
        "pair_target": np.array([p[1] - 1 for p in pairs], dtype="int64"),
        "pair_dist": np.array([max(0, min(unique[i]) - t, t - max(unique[i])) / 100.0 for i, t in pairs]),
        "pair_span": np.array([(max(unique[i]) - min(unique[i])) / 100.0 for i, _ in pairs]),
        "pair_n": np.array([float(len(unique[i])) for i, _ in pairs]),
        "row_pair": np.array([pair_index[(index[s], t)] for s, t in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
