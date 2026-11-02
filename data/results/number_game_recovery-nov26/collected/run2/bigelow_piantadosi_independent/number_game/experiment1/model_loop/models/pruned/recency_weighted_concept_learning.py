"""Recency-weighted concept learning: people infer the hidden concept by
Bayesian inference over 45 mathematical rules (each with its own learned prior
plausibility) and all intervals of 1..100 with the size principle, but they read
the examples in order and their memory of earlier examples fades. Each example's
log-likelihood is weighted by decay**(number of examples shown after it), so the
most recent examples dominate the inferred concept (decay = 1 recovers the
order-blind learner). "Yes" is the posterior probability that the concept
contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
MAX_EX = 4
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


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
# Intervals [lo, hi] of 1..100 (all equally likely a priori).
LO, HI = np.triu_indices(100)
LO, HI = LO + 1, HI + 1
SIZE = HI - LO + 1
INTERVAL_MEMBERSHIP = (NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])

with pm.Model() as model:
    # Per distinct set, concepts are the 45 rules followed by groups of
    # intervals that share a size and the same subset of examples (identical
    # likelihood under any decay, so grouping is exact). lag_log_lik holds each
    # example's log-likelihood by lag (0 = last example shown), zero-padded.
    lag_log_lik = pm.Data("lag_log_lik", np.zeros((1, MAX_EX, N_RULES + 1)))
    group_log_count = pm.Data("group_log_count", np.zeros((1, 1)))
    # Distinct (set, target) cells: the set each belongs to and the target's
    # membership in each concept; row_cell maps every trial to its cell.
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_member = pm.Data("cell_member", np.zeros((1, N_RULES + 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    rule_logit = pm.ZeroSumNormal("rule_logit", sigma=1.0, shape=N_RULES)
    # Memory decay per intervening example: 1 = no forgetting.
    decay = pm.Beta("decay", 2.0, 1.0)

    weights = decay ** pt.arange(MAX_EX)
    set_log_lik = pt.tensordot(lag_log_lik, weights, axes=[[1], [0]])  # (sets, concepts)

    log_rule_prior = pt.log(lambda_0) + pt.special.log_softmax(rule_logit)
    log_interval_prior = pt.log1p(-lambda_0) - 2 * np.log(100) + group_log_count
    log_prior = pt.concatenate(
        [pt.broadcast_to(log_rule_prior[None, :], (set_log_lik.shape[0], N_RULES)), log_interval_prior], axis=1
    )
    log_joint = log_prior + set_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[cell_set] * cell_member).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    groups = []
    for s in unique:
        ex = np.array(s[::-1])  # lag order
        inside = (ex[None, :] >= LO[:, None]) & (ex[None, :] <= HI[:, None])
        key = SIZE * 16 + (inside * (2 ** np.arange(len(ex)))).sum(axis=1)
        _, first, inverse, counts = np.unique(key, return_index=True, return_inverse=True, return_counts=True)
        member = np.zeros((len(counts), 100))
        np.add.at(member, inverse.ravel(), INTERVAL_MEMBERSHIP)
        loglik = np.where(inside[first], np.log(ALPHA / SIZE[first] + (1 - ALPHA) / 100)[:, None], np.log((1 - ALPHA) / 100))
        groups.append((ex, loglik, counts, member / counts[:, None]))
    n_groups = max(len(g[2]) for g in groups)
    n_conc = N_RULES + n_groups
    lag = np.zeros((len(unique), MAX_EX, n_conc))
    log_count = np.full((len(unique), n_groups), -1e4)
    member = np.zeros((len(unique), n_conc, 100))
    for i, (ex, loglik, counts, mem) in enumerate(groups):
        g = len(counts)
        lag[i, : len(ex), :N_RULES] = RULE_LOG_LIK[:, ex - 1].T
        lag[i, : len(ex), N_RULES : N_RULES + g] = loglik.T
        log_count[i, :g] = np.log(counts)
        member[i, :N_RULES] = RULE_MEMBERSHIP
        member[i, N_RULES : N_RULES + g] = mem
    pairs = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(pairs))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_set = np.array([c[0] for c in cells], dtype="int64")
    cell_target = np.array([c[1] for c in cells], dtype="int64")
    return {
        "lag_log_lik": lag,
        "group_log_count": log_count,
        "cell_set": cell_set,
        "cell_member": member[cell_set, :, cell_target],
        "row_cell": np.array([cell_index[c] for c in pairs], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
