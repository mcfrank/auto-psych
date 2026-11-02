"""Place-value grid block concepts with a sticky working guess.

People represent the numbers 1-100 as cells of a place-value grid (tens digit
by ones digit). Besides arithmetic rules (odd, even, primes, squares, cubes,
powers, multiples), the only concepts they entertain are blocks on that grid:
a contiguous run of tens digits crossed with a contiguous run of ones digits
("the forties", "ends in 6", "ends in 5 to 7", "20s-30s ending in 3 to 7").
Number-line ranges and digit rules are not separate concepts. The concept is
learned by Bayesian concept learning (size principle, fitted trust `alpha` in
the examples, Erlang prior on block size) with a single working guess formed
from the first example and kept with probability `stay` while later examples
fit it, reconsidered from all examples so far otherwise.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4
TENS = NUMBERS // 10  # 0..10
ONES = NUMBERS % 10


def arithmetic_rules():
    rules = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "even": list(range(2, 101, 2)),
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
        "powers of 2, not 32": [2, 4, 8, 16, 64],
        "powers of 2, or 37": [2, 4, 8, 16, 32, 37, 64],
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    return list(rules.values())


RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in arithmetic_rules()], dtype="float64")
N_RULES = len(RULE_MEMBERSHIP)
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)


def grid_blocks():
    seen = set()
    blocks = []
    for a in range(11):
        for b in range(a, 11):
            for c in range(10):
                for d in range(c, 10):
                    mem = (TENS >= a) & (TENS <= b) & (ONES >= c) & (ONES <= d)
                    key = mem.tobytes()
                    if mem.any() and key not in seen:
                        seen.add(key)
                        blocks.append(mem)
    return np.array(blocks, dtype="float64")


BLOCK_MEMBERSHIP = grid_blocks()
BLOCK_SIZE = BLOCK_MEMBERSHIP.sum(axis=1)
G_MAX = 400  # (size, examples-covered) block groups per set, padded
N_HYP = N_RULES + G_MAX
BLOCK_LOG_NORM_SIZES = BLOCK_SIZE  # every block enters the prior's normaliser

with pm.Model() as model:
    contains = pm.Data("contains", np.ones((1, MAX_N, N_HYP)))  # example k lies in hypothesis
    step_mask = pm.Data("step_mask", np.zeros((1, MAX_N)))
    group_size = pm.Data("group_size", np.ones((1, G_MAX)))  # members of each block in the group
    group_log_count = pm.Data("group_log_count", np.zeros((1, G_MAX)))  # log #blocks (-1e4 padding)
    group_mem = pm.Data("group_mem", np.zeros((1, G_MAX, 100)))  # mean membership of the group's blocks
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior share of arithmetic rules versus grid blocks.
    lambda_0 = pm.Beta("lambda_0", 2, 2)
    # Erlang preference for a typical block size.
    log_sigma = pm.Normal("log_sigma", mu=np.log(10.0), sigma=1.0)
    sigma = pt.exp(log_sigma)
    all_sizes = pt.as_tensor(BLOCK_LOG_NORM_SIZES)
    log_norm = pt.logsumexp(pt.log(all_sizes) - all_sizes / sigma)
    group_log_prior = group_log_count + pt.log(group_size) - group_size / sigma - log_norm
    n_sets = group_size.shape[0]
    log_prior = pt.concatenate([
        pt.zeros((n_sets, N_RULES)) + pt.log(lambda_0) - np.log(N_RULES),
        pt.log1p(-lambda_0) + group_log_prior,
    ], axis=1)  # (sets, hyp)

    # Trust that examples come from the concept; stickiness of a fitting guess.
    alpha = pm.Beta("alpha", 9, 1)
    stay = pm.Beta("stay", 2, 2)

    hyp_size = pt.concatenate([pt.zeros((n_sets, N_RULES)) + pt.as_tensor(RULE_SIZE)[None, :], group_size], axis=1)
    log_in = pt.log(alpha / hyp_size + (1 - alpha) / 100.0)[:, None, :]
    log_out = pt.log1p(-alpha) - np.log(100.0)
    step_ll = step_mask[:, :, None] * (contains * log_in + (1 - contains) * log_out)
    prefix_log_lik = pt.cumsum(step_ll, axis=1)
    post = pt.special.softmax(log_prior[:, None, :] + prefix_log_lik, axis=2)  # (sets, steps, hyp)

    q = post[:, 0, :]
    for k in range(1, MAX_N):
        kept = stay * q * contains[:, k, :]
        updated = kept + (1 - kept.sum(axis=1, keepdims=True)) * post[:, k, :]
        m = step_mask[:, k][:, None]
        q = m * updated + (1 - m) * q

    gen = pt.dot(q[:, :N_RULES], pt.as_tensor(RULE_MEMBERSHIP)) + pt.batched_dot(q[:, N_RULES:], group_mem)
    p_yes = pm.Deterministic("p_yes", pt.clip(gen[row_set, row_target], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`.

    Blocks of equal size covering the same examples have equal prior and
    likelihood in a set, so they are pooled; the group's generalisation is the
    mean membership of its blocks."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    contains = np.ones((n_sets, MAX_N, N_HYP))
    mask = np.zeros((n_sets, MAX_N))
    gsize = np.ones((n_sets, G_MAX))
    glogc = np.full((n_sets, G_MAX), -1e4)
    gmem = np.zeros((n_sets, G_MAX, 100))
    for i, s in enumerate(unique):
        s = s[:MAX_N]
        inside = BLOCK_MEMBERSHIP[:, [x - 1 for x in s]]  # (blocks, examples)
        code = (inside * (1 << np.arange(len(s)))).sum(axis=1).astype(int)
        keys, group = np.unique(np.stack([BLOCK_SIZE.astype(int), code], axis=1), axis=0, return_inverse=True)
        group = group.reshape(-1)
        n_g = len(keys)
        assert n_g <= G_MAX
        counts = np.bincount(group, minlength=n_g)
        gsize[i, :n_g] = keys[:, 0]
        glogc[i, :n_g] = np.log(counts)
        np.add.at(gmem[i], group, BLOCK_MEMBERSHIP)
        gmem[i, :n_g] /= counts[:, None]
        for k, x in enumerate(s):
            contains[i, k, :N_RULES] = RULE_MEMBERSHIP[:, x - 1]
            contains[i, k, N_RULES:] = 0.0
            contains[i, k, N_RULES:N_RULES + n_g] = (keys[:, 1] >> k) & 1
            mask[i, k] = 1.0
    return {
        "contains": contains,
        "step_mask": mask,
        "group_size": gsize,
        "group_log_count": glogc,
        "group_mem": gmem,
        "row_set": np.array([index[s] for s in sets], dtype="int64"),
        "row_target": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
