"""Recency-weighted evidence in the tempered rule-family-prior learner.

Refinement of `rule_family_prior_tempered_mixture`: the learner considers the
same 45 rules (prior structured by kind of rule, each kind's log-plausibility
inferred) and all intervals of 1..100; each example comes from the concept with
probability 0.9 (uniformly) and from 1..100 otherwise, and the size-principle
evidence is tempered by a sampling strength beta. The one change: examples are
read in order and memory of earlier ones fades, so the log-likelihood of an
example shown k examples before the last is weighted by beta * decay**k
(decay = 1 recovers the order-blind model). "Yes" is the posterior probability
that the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
MAX_EX = 4
NUMBERS = np.arange(1, 101)


def rule_concepts():
    """(family, members) for each rule; families index FAMILIES."""
    primes = [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))]
    rules = [
        (0, [n for n in NUMBERS if n % 2 == 1]),
        (0, list(range(2, 101, 2))),
        (1, primes),
        (2, [n * n for n in range(1, 11)]),
        (2, [n**3 for n in range(1, 5)]),
        (3, [2, 4, 8, 16, 64]),
        (3, [2, 4, 8, 16, 32, 37, 64]),
    ]
    rules += [(4, list(range(m, 101, m))) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rules += [(3, [m**k for k in range(1, 8) if m**k <= 100]) for m in range(2, 11)]
    rules += [(5, [n for n in NUMBERS if n % 10 == d]) for d in range(10)]
    rules += [(6, [n for n in NUMBERS if str(d) in str(n)]) for d in range(10)]
    return rules


FAMILIES = ["parity", "primes", "squares_cubes", "powers", "multiples", "final_digit", "contains_digit"]
_RULES = rule_concepts()
RULE_FAMILY = np.array([f for f, _ in _RULES], dtype="int64")
N_RULES = len(_RULES)
N_FAM = len(FAMILIES)
FAMILY_LOG_COUNT = np.log(np.bincount(RULE_FAMILY, minlength=N_FAM)).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for _, c in _RULES], dtype="float64")
RULE_EX_LL = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
LO, HI = np.triu_indices(100)
LO, HI = LO + 1, HI + 1
SIZE = HI - LO + 1
N_INTERVALS = len(LO)
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")

with pm.Model() as model:
    # Per distinct set: each example's log-likelihood by lag (0 = last shown),
    # for the 45 rules then the interval groups (zero-padded lags/groups).
    lag_log_lik = pm.Data("lag_log_lik", np.zeros((1, MAX_EX, N_RULES + 1)))
    group_log_size = pm.Data("group_log_size", np.zeros((1, 1)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_member = pm.Data("cell_member", np.zeros((1, N_RULES + 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_logit = pm.Normal("family_logit", mu=0.0, sigma=2.5, shape=N_FAM)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)  # sampling strength
    decay = pm.Beta("decay", 2.0, 1.0)  # memory retained per later example

    weights = beta * decay ** pt.arange(MAX_EX)
    set_log_lik = pt.tensordot(lag_log_lik, weights, axes=[[1], [0]])  # (sets, concepts)

    log_norm = pt.logsumexp(pt.concatenate([family_logit, pt.zeros(1)]))
    log_rule_prior = family_logit[RULE_FAMILY] - FAMILY_LOG_COUNT[RULE_FAMILY] - log_norm
    log_interval_prior = -log_norm - np.log(N_INTERVALS) + group_log_size
    log_prior = pt.concatenate(
        [pt.broadcast_to(log_rule_prior[None, :], (set_log_lik.shape[0], N_RULES)), log_interval_prior], axis=1
    )
    log_joint = log_prior + set_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[cell_set] * cell_member).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`.

    Intervals sharing a size and the same subset of examples have identical
    lag-wise likelihoods, so they are grouped exactly per distinct set; padding
    groups get a log size of -1e4 (no posterior mass).
    """
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
        loglik = np.where(
            inside[first], np.log(ALPHA / SIZE[first] + (1 - ALPHA) / 100)[:, None], np.log((1 - ALPHA) / 100)
        )
        groups.append((ex, loglik, counts, member / counts[:, None]))
    n_groups = max(len(g[2]) for g in groups)
    n_conc = N_RULES + n_groups
    lag = np.zeros((len(unique), MAX_EX, n_conc))
    log_size = np.full((len(unique), n_groups), -1e4)
    member = np.zeros((len(unique), n_conc, 100))
    for i, (ex, loglik, counts, mem) in enumerate(groups):
        g = len(counts)
        lag[i, : len(ex), :N_RULES] = RULE_EX_LL[:, ex - 1].T
        lag[i, : len(ex), N_RULES : N_RULES + g] = loglik.T
        log_size[i, :g] = np.log(counts)
        member[i, :N_RULES] = RULE_MEMBERSHIP
        member[i, N_RULES : N_RULES + g] = mem
    pairs = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(pairs))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_set = np.array([c[0] for c in cells], dtype="int64")
    cell_target = np.array([c[1] for c in cells], dtype="int64")
    return {
        "lag_log_lik": lag,
        "group_log_size": log_size,
        "cell_set": cell_set,
        "cell_member": member[cell_set, :, cell_target],
        "row_cell": np.array([cell_index[c] for c in pairs], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
