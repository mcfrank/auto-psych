"""First-example win-stay, lose-sample concept learning.

People do not weigh all examples together; they form a guess about the
concept from the first example they see and then test it against the later
examples. If every later example fits the guess, they keep it; only when an
example contradicts it do they abandon it and re-infer the concept from all
the examples. Broad concepts suggested by the first example therefore persist
more than an ideal learner allows, and example order matters.

Concepts and the base learner follow `rule_family_prior_tempered_mixture`: 45
rules with a prior by kind of rule, all intervals of 1..100, examples drawn
from the concept with probability 0.9, a tempered size principle (beta). The
guess is sampled from the posterior given the first example, p1; it is kept
if it contains every example, else resampled from the posterior given all
examples, pN. The concept distribution is
    q(h) = p1(h) c(h) + (1 - sum_h p1(h) c(h)) pN(h),
and "yes" is the probability under q that the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


def rule_concepts():
    """(family, members) for each rule."""
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


_RULES = rule_concepts()
RULE_FAMILY = np.array([f for f, _ in _RULES], dtype="int64")
N_RULES = len(_RULES)
N_FAM = 7
FAMILY_LOG_COUNT = np.log(np.bincount(RULE_FAMILY, minlength=N_FAM)).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for _, c in _RULES], dtype="float64")
RULE_EX_LL = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
SIZE = HI - LO + 1
N_INTERVALS = len(LO)
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")

with pm.Model() as model:
    # Per distinct set: concepts are the 45 rules then groups of intervals with
    # the same size and the same subset of examples inside (exact grouping).
    first_log_lik = pm.Data("first_log_lik", np.zeros((1, N_RULES + 1)))  # (sets, concepts)
    full_log_lik = pm.Data("full_log_lik", np.zeros((1, N_RULES + 1)))  # (sets, concepts)
    consistent = pm.Data("consistent", np.zeros((1, N_RULES + 1)))  # contains every example
    group_log_count = pm.Data("group_log_count", np.zeros((1, 1)))  # (sets, interval groups)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_member = pm.Data("cell_member", np.zeros((1, N_RULES + 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_logit = pm.Normal("family_logit", mu=0.0, sigma=2.5, shape=N_FAM)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)  # sampling strength

    log_norm = pt.logsumexp(pt.concatenate([family_logit, pt.zeros(1)]))
    log_rule_prior = family_logit[RULE_FAMILY] - FAMILY_LOG_COUNT[RULE_FAMILY] - log_norm
    log_interval_prior = -log_norm - np.log(N_INTERVALS) + group_log_count
    log_prior = pt.concatenate(
        [pt.broadcast_to(log_rule_prior[None, :], (group_log_count.shape[0], N_RULES)), log_interval_prior], axis=1
    )

    def posterior(log_lik):
        log_joint = log_prior + beta * log_lik
        return pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))

    p_first = posterior(first_log_lik)
    p_full = posterior(full_log_lik)
    kept = p_first * consistent
    q = kept + (1.0 - kept.sum(axis=1, keepdims=True)) * p_full
    in_concept = (q[cell_set] * cell_member).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    groups = []
    for s in unique:
        ex = np.array(s)
        inside = (ex[None, :] >= LO[:, None]) & (ex[None, :] <= HI[:, None])
        key = SIZE * 16 + (inside * (2 ** np.arange(len(ex)))).sum(axis=1)
        _, first, inverse, counts = np.unique(key, return_index=True, return_inverse=True, return_counts=True)
        member = np.zeros((len(counts), 100))
        np.add.at(member, inverse.ravel(), INTERVAL_MEMBERSHIP)
        ins = inside[first]
        ll = np.log(ALPHA * ins / SIZE[first][:, None] + (1 - ALPHA) / 100)
        groups.append((ex, ll, ins.all(axis=1), counts, member / counts[:, None]))
    n_groups = max(len(g[3]) for g in groups)
    n_conc = N_RULES + n_groups
    first_ll = np.zeros((len(unique), n_conc))
    full_ll = np.zeros((len(unique), n_conc))
    cons = np.zeros((len(unique), n_conc))
    log_count = np.full((len(unique), n_groups), -1e4)
    member = np.zeros((len(unique), n_conc, 100))
    for i, (ex, ll, c, counts, mem) in enumerate(groups):
        g = len(counts)
        first_ll[i, :N_RULES] = RULE_EX_LL[:, ex[0] - 1]
        full_ll[i, :N_RULES] = RULE_EX_LL[:, ex - 1].sum(axis=1)
        cons[i, :N_RULES] = RULE_MEMBERSHIP[:, ex - 1].all(axis=1)
        first_ll[i, N_RULES : N_RULES + g] = ll[:, 0]
        full_ll[i, N_RULES : N_RULES + g] = ll.sum(axis=1)
        cons[i, N_RULES : N_RULES + g] = c
        log_count[i, :g] = np.log(counts)
        member[i, :N_RULES] = RULE_MEMBERSHIP
        member[i, N_RULES : N_RULES + g] = mem
    pairs = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(pairs))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_set = np.array([c[0] for c in cells], dtype="int64")
    cell_target = np.array([c[1] for c in cells], dtype="int64")
    return {
        "first_log_lik": first_ll,
        "full_log_lik": full_ll,
        "consistent": cons,
        "group_log_count": log_count,
        "cell_set": cell_set,
        "cell_member": member[cell_set, :, cell_target],
        "row_cell": np.array([cell_index[c] for c in pairs], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
