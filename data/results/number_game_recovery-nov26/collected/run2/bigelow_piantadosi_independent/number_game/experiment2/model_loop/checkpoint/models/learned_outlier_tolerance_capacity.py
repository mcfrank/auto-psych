"""Learned outlier tolerance in capacity-limited family-prior concept learning.

Refinement of `capacity_limited_evidence_integration`: people infer the hidden
concept by Bayesian learning over 45 rules (separate prior plausibilities for
even, odd and the other kinds of rule) and all intervals of 1-100, with the
summed example evidence scaled by beta * n ** (-gamma) (capacity-limited
integration). The single change: the probability that an example is a stray,
uniformly random number (fixed at 0.1 in the incumbent) is a free parameter
`noise`. Low noise means every example must lie in the concept, so ranges not
covering all examples are excluded and mixed-parity sets are generalized
sharply within the span of the examples.

The example likelihood of a hypothesis depends only on how many examples it
contains (k) and its size, so intervals are grouped by (k, size), which is
independent of the noise level and computed once in prepare_observed.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4


def rule_concepts():
    """(family, members) for each rule; families index FAMILIES."""
    primes = [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))]
    rules = [
        (0, [n for n in NUMBERS if n % 2 == 1]),
        (1, list(range(2, 101, 2))),
        (2, primes),
        (3, [n * n for n in range(1, 11)]),
        (3, [n**3 for n in range(1, 5)]),
        (4, [2, 4, 8, 16, 64]),
        (4, [2, 4, 8, 16, 32, 37, 64]),
    ]
    rules += [(5, list(range(m, 101, m))) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rules += [(4, [m**k for k in range(1, 8) if m**k <= 100]) for m in range(2, 11)]
    rules += [(6, [n for n in NUMBERS if n % 10 == d]) for d in range(10)]
    rules += [(7, [n for n in NUMBERS if str(d) in str(n)]) for d in range(10)]
    return rules


FAMILIES = ["odd", "even", "primes", "squares_cubes", "powers", "multiples", "final_digit", "contains_digit"]
_RULES = rule_concepts()
RULE_FAMILY = np.array([f for f, _ in _RULES], dtype="int64")
N_RULES = len(_RULES)
N_FAM = len(FAMILIES)
FAMILY_LOG_COUNT = np.log(np.bincount(RULE_FAMILY, minlength=N_FAM)).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for _, c in _RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
INTERVALS = np.array([(m, n) for m in NUMBERS for n in range(m, 101)])
N_INTERVALS = len(INTERVALS)
INTERVAL_MEMBERSHIP = (NUMBERS[None, :] >= INTERVALS[:, :1]) & (NUMBERS[None, :] <= INTERVALS[:, 1:])
INTERVAL_MEMBERSHIP = INTERVAL_MEMBERSHIP.astype("float64")
INTERVAL_SIZE = (INTERVALS[:, 1] - INTERVALS[:, 0] + 1).astype("int64")
# Interval groups: (k examples covered, size); group id = k * 100 + size - 1.
N_GROUPS = (MAX_N + 1) * 100
GROUP_K = np.repeat(np.arange(MAX_N + 1), 100).astype("float64")
GROUP_SIZE = np.tile(np.arange(1, 101), MAX_N + 1).astype("float64")

with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples in rule
    group_log_size = pm.Data("group_log_size", np.zeros((1, N_GROUPS)))  # (sets, groups)
    set_n = pm.Data("set_n", np.ones(1))  # number of examples per set
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))  # (set-target pairs,)
    pair_rule_in = pm.Data("pair_rule_in", np.zeros((1, N_RULES)))  # (pairs, rules)
    pair_frac = pm.Data("pair_frac", np.zeros((1, N_GROUPS)))  # (pairs, groups)
    row_pair = pm.Data("row_pair", np.zeros(1, dtype="int64"))  # (rows,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_logit = pm.Normal("family_logit", mu=0.0, sigma=2.5, shape=N_FAM)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)  # sampling strength of a single example
    gamma = pm.Normal("gamma", mu=0.0, sigma=0.5)  # capacity limit
    noise_logit = pm.Normal("noise_logit", mu=-2.2, sigma=1.5)  # belief in stray examples
    noise = pm.Deterministic("noise", pm.math.sigmoid(noise_logit))

    log_norm = pt.logsumexp(pt.concatenate([family_logit, pt.zeros(1)]))
    log_rule_prior = family_logit[RULE_FAMILY] - FAMILY_LOG_COUNT[RULE_FAMILY] - log_norm
    log_interval_prior = -log_norm - np.log(N_INTERVALS)

    log_out = pt.log(noise / 100.0)
    n_col = set_n[:, None]
    rule_ll = rule_k * pt.log((1 - noise) / RULE_SIZE + noise / 100.0)[None, :] + (n_col - rule_k) * log_out
    group_ll = GROUP_K[None, :] * pt.log((1 - noise) / GROUP_SIZE + noise / 100.0)[None, :] + (
        n_col - GROUP_K[None, :]
    ) * log_out

    beta_set = beta * pt.exp(-gamma * pt.log(n_col))
    log_rules = log_rule_prior[None, :] + beta_set * rule_ll
    log_groups = log_interval_prior + group_log_size + beta_set * group_ll
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    post_pair = posterior[pair_set]
    in_concept = (post_pair[:, :N_RULES] * pair_rule_in).sum(axis=1) + (post_pair[:, N_RULES:] * pair_frac).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-6, 1 - 1e-6)[row_pair])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`.

    Empty (k, size) groups get a log size of -1e4 (no posterior mass).
    """
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}

    rule_k = np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique])
    group_ids, counts = [], np.zeros((len(unique), N_GROUPS))
    for i, s in enumerate(unique):
        k = INTERVAL_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1).astype("int64")
        g = k * 100 + INTERVAL_SIZE - 1
        group_ids.append(g)
        counts[i] = np.bincount(g, minlength=N_GROUPS)
    log_size = np.where(counts > 0, np.log(np.maximum(counts, 1)), -1e4)
    keys = [(s, int(r["target"])) for s, r in zip(sets, rows)]
    pairs = list(dict.fromkeys(keys))
    pair_index = {p: j for j, p in enumerate(pairs)}
    frac = np.zeros((len(pairs), N_GROUPS))
    rule_in = np.zeros((len(pairs), N_RULES))
    for j, (s, t) in enumerate(pairs):
        i = set_index[s]
        frac[j] = np.bincount(group_ids[i], weights=INTERVAL_MEMBERSHIP[:, t - 1], minlength=N_GROUPS)
        frac[j] /= np.maximum(counts[i], 1)
        rule_in[j] = RULE_MEMBERSHIP[:, t - 1]
    return {
        "rule_k": rule_k,
        "group_log_size": log_size,
        "set_n": np.array([len(s) for s in unique], dtype="float64"),
        "pair_set": np.array([set_index[s] for s, _ in pairs], dtype="int64"),
        "pair_rule_in": rule_in,
        "pair_frac": frac,
        "row_pair": np.array([pair_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(float(r.get("rating", 0))) for r in rows], dtype="int64"),
    }
