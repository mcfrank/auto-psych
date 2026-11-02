"""Serial-position-weighted Bayesian concept learning.

People infer the hidden concept by Bayesian generalization over rules and
intervals, but they do not weigh the examples equally: how much an example
counts as evidence depends on where it came in the order shown. One
serial-position gradient, inferred from the data, can favour the first examples
(primacy, gamma > 0) or the most recent ones (recency, gamma < 0).

The hypothesis space and prior are those of the Bigelow & Piantadosi (2016)
Independent model (45 rules with a weight inferred for each, a rule share
lambda_0, and every interval [m, n] of 1..100 with prior (1 - lambda_0) / 100^2).
Each example is drawn from the concept with probability 0.9 (size principle),
else uniformly. Example k of n contributes w_k * log P(x_k | h), with
w = n * softmax(-gamma * (k - 1)): total evidence stays at n examples and only
its spread over serial positions changes (gamma = 0 is equal weighting). "Yes"
is the posterior probability that the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
MAX_K = 4
NUMBERS = np.arange(1, 101)


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


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
# log P(x | h) = log(0.001) + in(x, h) * log(1 + 900 / |h|): the gain over the
# out-of-concept floor; the floor, n * log(0.001), is common to every hypothesis.
RULE_GAIN = np.log1p(ALPHA * 100 / (1 - ALPHA) / RULE_MEMBERSHIP.sum(axis=1))
LOG_INTERVAL_PRIOR = -2 * np.log(100)

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0


def prepare_observed(rows):
    """Per set: rule gains by example position. Intervals are grouped, per set,
    by size and by which examples they contain (their likelihood depends only on
    that); for each target asked about with the set, how many intervals of each
    group contain it. Everything is padded to dense (set, target, group) arrays."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    s_index = {s: i for i, s in enumerate(unique)}
    targets = [list(dict.fromkeys(int(r["target"]) for s, r in zip(sets, rows) if s == u)) for u in unique]
    t_index = [{t: j for j, t in enumerate(ts)} for ts in targets]
    n_t = max(len(ts) for ts in targets)

    per_set = []
    for si, s in enumerate(unique):
        xs = s[:MAX_K]
        groups = {}
        for m in range(1, 101):
            for n in range(m, 101):
                key = (n - m + 1, tuple(int(m <= x <= n) for x in xs) + (0,) * (MAX_K - len(xs)))
                entry = groups.setdefault(key, np.zeros(1 + len(targets[si])))
                entry[0] += 1
                for j, t in enumerate(targets[si]):
                    entry[1 + j] += m <= t <= n
        per_set.append(groups)
    n_g = max(len(g) for g in per_set)

    S = len(unique)
    rule_gain = np.zeros((S, MAX_K, N_RULES))
    pos_mask = np.zeros((S, MAX_K))
    g_gain = np.zeros((S, n_g, MAX_K))
    g_count = np.zeros((S, n_g))
    t_count = np.zeros((S, n_t, n_g))
    t_rule = np.zeros((S, n_t, N_RULES))
    for si, s in enumerate(unique):
        for k, x in enumerate(s[:MAX_K]):
            rule_gain[si, k] = RULE_MEMBERSHIP[:, x - 1] * RULE_GAIN
            pos_mask[si, k] = 1.0
        for gi, ((size, inside), counts) in enumerate(per_set[si].items()):
            g_gain[si, gi] = np.array(inside) * np.log1p(ALPHA * 100 / (1 - ALPHA) / size)
            g_count[si, gi] = counts[0]
            t_count[si, :len(targets[si]), gi] = counts[1:]
        for j, t in enumerate(targets[si]):
            t_rule[si, j] = RULE_MEMBERSHIP[:, t - 1]
    return {
        "rule_gain": rule_gain,
        "pos_mask": pos_mask,
        "g_gain": g_gain,
        "g_count": g_count,
        "t_count": t_count,
        "t_rule": t_rule,
        "row_cell": np.array([s_index[s] * n_t + t_index[s_index[s]][int(r["target"])]
                              for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    rule_gain = pm.Data("rule_gain", np.zeros((1, MAX_K, N_RULES)))  # (sets, K, rules)
    pos_mask = pm.Data("pos_mask", np.ones((1, MAX_K)))  # (sets, K): example present
    g_gain = pm.Data("g_gain", np.zeros((1, 1, MAX_K)))  # (sets, groups, K)
    g_count = pm.Data("g_count", np.ones((1, 1)))  # (sets, groups): #intervals in group
    t_count = pm.Data("t_count", np.zeros((1, 1, 1)))  # (sets, targets, groups): ... containing target
    t_rule = pm.Data("t_rule", np.zeros((1, 1, N_RULES)))  # (sets, targets, rules): rule contains target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * n_targets + target slot
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    # Serial-position gradient: > 0 primacy, < 0 recency, 0 equal weighting.
    gamma = pm.Normal("gamma", mu=0.0, sigma=1.0)

    n_ex = pos_mask.sum(axis=1, keepdims=True)
    raw = pt.exp(-gamma * pt.arange(MAX_K)[None, :]) * pos_mask
    w = n_ex * raw / raw.sum(axis=1, keepdims=True)  # (sets, K), sums to n

    # Joint relative to the shared floor n * log(0.001); bounded, so exp is safe.
    rule_score = pt.exp(pt.log(lambda_0) + pt.log(rule_weights)[None, :]
                        + (w[:, :, None] * rule_gain).sum(axis=1))  # (sets, rules)
    int_score = pt.exp(pt.log1p(-lambda_0) + LOG_INTERVAL_PRIOR
                       + (w[:, None, :] * g_gain).sum(axis=2))  # (sets, groups), per interval
    z = rule_score.sum(axis=1) + (int_score * g_count).sum(axis=1)  # (sets,)
    in_concept = (pt.batched_dot(t_rule, rule_score) + pt.batched_dot(t_count, int_score)) / z[:, None]  # (sets, targets)
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
