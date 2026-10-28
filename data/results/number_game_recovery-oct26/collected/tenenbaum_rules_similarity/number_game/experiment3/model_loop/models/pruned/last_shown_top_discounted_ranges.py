"""Last-shown top marking with range-discounted size principle: refinement of
upper_edge_discounted_range_size. Bayesian concept learning over 42 rules
(kind-level prior, primes excluding 1), every interval of 1-100 (Erlang width
prior, decade-boundary bonus) and range-restricted rules, 10% uniform random
examples, "yes" = posterior probability the concept contains the target; the
size principle applies at full strength to rules but only partly to plain
ranges (effective size a geometric interpolation, inferred sharpness, between
true width and 100). The one change, taken from last_example_marks_range_top:
the top-edge evidence is generative and read from the order of presentation
instead of a constant bonus - with inferred odds the program ends its
demonstration on its range's top number, so when the largest example is shown
last a range ending there is credited by 1 + odds / (chance a random draw from
that range, judged at its effective size, hits that number), and a set whose
largest example comes earlier gives no edge credit.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

EPSILON = 0.1
NUMBERS = np.arange(1, 101)


def concept_sets():
    rules = [
        [2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36],
        [7, 49], [8, 64], [9, 81], [10, 100],
    ]
    rules += [[i for i in NUMBERS if i % n == 0] for n in range(3, 13)]
    rules += [[i for i in NUMBERS if i % 10 == d] for d in range(1, 10)]
    rules += [[i for i in NUMBERS if str(d) in str(i)] for d in range(1, 10)]
    # Primes, excluding 1 (the one change from range_restricted_rule_concepts).
    rules.append([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    rules.append([i for i in NUMBERS if i % 2 == 0])
    rules.append([i for i in NUMBERS if i % 2 == 1])
    rules.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    rules.append([1, 8, 27, 64])
    kinds = [0] * 9 + [1] * 10 + [2] * 9 + [3] * 9 + [4] * 5
    bounds = [(m, n) for m in range(1, 101) for n in range(m, 101)]
    return rules, kinds, bounds


RULES, RULE_KINDS, BOUNDS = concept_sets()
N_RULES = len(RULES)
N_KINDS = 5
KIND_MATRIX = np.eye(N_KINDS)[RULE_KINDS]
KIND_COUNTS = KIND_MATRIX.sum(axis=0)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
INTERVAL_MEMBERSHIP = np.array([(NUMBERS >= m) & (NUMBERS <= n) for m, n in BOUNDS], dtype="float64")
INTERVAL_WIDTH = np.array([n - m + 1 for m, n in BOUNDS])
DECADE_BOUNDED = np.array([(m - 1) % 10 == 0 and n % 10 == 0 for m, n in BOUNDS], dtype="int64")
GROUP_KEYS, GROUP_OF_INTERVAL, GROUP_SIZES = np.unique(
    np.stack([INTERVAL_WIDTH, DECADE_BOUNDED], axis=1), axis=0, return_inverse=True, return_counts=True
)
GROUP_OF_INTERVAL = GROUP_OF_INTERVAL.ravel()
GROUP_WIDTHS = GROUP_KEYS[:, 0].astype("float64")
GROUP_ALIGNED = GROUP_KEYS[:, 1].astype("float64")
N_GROUPS = len(GROUP_KEYS)
GROUP_MEMBERS = [np.flatnonzero(GROUP_OF_INTERVAL == g) for g in range(N_GROUPS)]

# Size of each conjunction "rule r within interval I" (rules x intervals).
CONJ_SIZE = RULE_MEMBERSHIP @ INTERVAL_MEMBERSHIP.T
NEG = -1e4
MAX_SLOTS = 12
KMAX = 5
BLOCK = 8
N_COLS = N_RULES + 2 * N_GROUPS * KMAX + MAX_SLOTS * N_GROUPS
K_RANGE = np.arange(KMAX, dtype="float64")
INTERVAL_TOP = np.array([n for _, n in BOUNDS])


def log_lik(member, size, examples):
    """Size-principle log-likelihood of the examples; member: (..., 100), size: (...)."""
    safe = np.maximum(size, 1.0)[..., None]
    per = np.log(member[..., examples - 1] * (1 - EPSILON) / safe + EPSILON / 100)
    return per.sum(axis=-1)


def set_quantities(examples):
    """Per-set evidence of rules, interval groups and conjunction groups (for
    the rules consistent with every example), and their within-group
    predictive membership over 1-100."""
    rule_ll = log_lik(RULE_MEMBERSHIP, RULE_MEMBERSHIP.sum(1), examples)
    int_ll = log_lik(INTERVAL_MEMBERSHIP, INTERVAL_MEMBERSHIP.sum(1), examples)
    consistent = np.flatnonzero(RULE_MEMBERSHIP[:, examples - 1].min(axis=1) > 0)[:MAX_SLOTS]
    slot_rule = np.zeros(MAX_SLOTS, dtype="int64")
    slot_rule[: len(consistent)] = consistent
    slot_used = np.arange(MAX_SLOTS) < len(consistent)
    rm = RULE_MEMBERSHIP[slot_rule]
    size = CONJ_SIZE[slot_rule]
    in_both = rm[:, None, examples - 1] * INTERVAL_MEMBERSHIP[None, :, examples - 1]
    conj_ll = np.log(in_both * (1 - EPSILON) / np.maximum(size, 1.0)[..., None] + EPSILON / 100).sum(-1)
    conj_ll = np.where(size > 0, conj_ll, NEG)
    int_ev = np.empty(N_GROUPS)
    int_mem = np.empty((N_GROUPS, 100))
    conj_ev = np.empty((MAX_SLOTS, N_GROUPS))
    conj_mem = np.empty((MAX_SLOTS, N_GROUPS, 100))
    for g, idx in enumerate(GROUP_MEMBERS):
        ll = int_ll[idx]
        int_ev[g] = logsumexp(ll)
        int_mem[g] = np.exp(ll - int_ev[g]) @ INTERVAL_MEMBERSHIP[idx]
        cl = conj_ll[:, idx]
        ev = logsumexp(cl, axis=1)
        conj_ev[:, g] = ev
        w = np.exp(cl - ev[:, None])
        conj_mem[:, g] = (w @ INTERVAL_MEMBERSHIP[idx]) * rm
    conj_ev[~slot_used] = NEG
    conj_mem[~slot_used] = 0.0
    return rule_ll, int_ev, int_mem, conj_ev, conj_mem, slot_rule


def k_resolved(examples):
    """Intervals grouped by (top at largest example?, group, number of
    examples inside): log count and mean membership over 1-100."""
    k = INTERVAL_MEMBERSHIP[:, examples - 1].sum(axis=1).astype(int)
    at_top = (INTERVAL_TOP == examples.max()).astype(int)
    lc = np.full((2, N_GROUPS, KMAX), NEG)
    mem = np.zeros((2, N_GROUPS, KMAX, 100))
    for g, idx in enumerate(GROUP_MEMBERS):
        for a in (0, 1):
            for kk in np.unique(k[idx]):
                sel = idx[(k[idx] == kk) & (at_top[idx] == a)]
                if len(sel) == 0:
                    continue
                lc[a, g, kk] = np.log(len(sel))
                mem[a, g, kk] = INTERVAL_MEMBERSHIP[sel].mean(axis=0)
    return lc, mem


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    group_k_log_count = pm.Data("group_k_log_count", np.zeros((1, 2, N_GROUPS, KMAX)))
    group_k_n = pm.Data("group_k_n", np.zeros((1, KMAX)))
    conj_log_evidence = pm.Data("conj_log_evidence", np.zeros((1, MAX_SLOTS, N_GROUPS)))
    top_is_last = pm.Data("top_is_last", np.zeros(1))
    slot_rule = pm.Data("slot_rule", np.zeros((1, MAX_SLOTS), dtype="int64"))
    # Targets of one set in blocks of BLOCK: membership of each target in each hypothesis column.
    block_set = pm.Data("block_set", np.zeros(1, dtype="int64"))
    block_in_hypothesis = pm.Data("block_in_hypothesis", np.zeros((1, BLOCK, N_COLS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior shares of plain rules, plain ranges and range-restricted rules.
    shares = pm.Dirichlet("shares", a=np.array([2.0, 2.0, 1.0]))
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    rule_weights = pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS)
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    decade_boost = pm.Normal("decade_boost", mu=1.0, sigma=1.5)
    # Odds that the program ends its demonstration on its range's top number.
    top_odds = pm.LogNormal("top_odds", mu=np.log(0.1), sigma=1.5)

    log_interval = np.log(GROUP_WIDTHS) - GROUP_WIDTHS / width_scale + decade_boost * GROUP_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval + np.log(GROUP_SIZES))
    log_rw = pt.log(rule_weights)

    log_rules = pt.log(shares[0]) + log_rw[None, :] + rule_log_lik
    sharpness = pm.Beta("sharpness", alpha=2.0, beta=2.0)
    log_w_eff = sharpness * np.log(GROUP_WIDTHS) + (1 - sharpness) * np.log(100.0)
    log_in = pt.log((1 - EPSILON) * pt.exp(-log_w_eff) + EPSILON / 100)
    # (sets, at_top, groups, k): log count of intervals with k examples inside + their likelihood
    gk = group_k_log_count + K_RANGE[None, None, None, :] * log_in[None, None, :, None] + (
        group_k_n[:, None, None, :] - K_RANGE[None, None, None, :]) * np.log(EPSILON / 100)
    group_log_evidence = pt.logsumexp(gk, axis=3)
    # Marking credit: only when the largest example is shown last.
    top_credit = pt.log1p(top_odds * top_is_last[:, None] * pt.exp(-log_in)[None, :])
    top_shift = pt.stack([pt.zeros_like(top_credit), top_credit], axis=1)
    log_groups = (
        pt.log(shares[1]) + log_interval_prior[None, None, :] + group_log_evidence + top_shift
    ).reshape((rule_log_lik.shape[0], 2 * N_GROUPS))
    log_conj = (
        pt.log(shares[2]) + log_rw[slot_rule][:, :, None] + log_interval_prior[None, None, :] + conj_log_evidence
    ).reshape((log_rules.shape[0], MAX_SLOTS * N_GROUPS))
    log_joint = pt.concatenate([log_rules, log_groups, log_conj], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    n_sets = rule_log_lik.shape[0]
    gk_w = pt.exp(gk - group_log_evidence[:, :, :, None])
    group_post = posterior[:, N_RULES:N_RULES + 2 * N_GROUPS].reshape((n_sets, 2, N_GROUPS))
    weights = pt.concatenate(
        [
            posterior[:, :N_RULES],
            (group_post[:, :, :, None] * gk_w).reshape((n_sets, 2 * N_GROUPS * KMAX)),
            posterior[:, N_RULES + 2 * N_GROUPS:],
        ],
        axis=1,
    )
    in_concept = pt.batched_dot(block_in_hypothesis, weights[block_set][:, :, None]).reshape((-1,))

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    by_set = {}
    for s, t in dict.fromkeys(zip(sets, targets)):
        by_set.setdefault(s, []).append(t)
    rule_ll, conj_ev, slots, gkc, gkn, lasts = [], [], [], [], [], []
    block_set, blocks, slot_of = [], [], {}
    for si, s in enumerate(unique):
        lc, kmem = k_resolved(np.array(s))
        gkc.append(lc)
        gkn.append(np.full(KMAX, float(len(s))))
        lasts.append(1.0 if s[-1] == max(s) else 0.0)
        r_ll, _, _, c_ev, c_mem, slot = set_quantities(np.array(s))
        rule_ll.append(r_ll)
        conj_ev.append(c_ev)
        slots.append(slot)
        ts = by_set[s]
        for b0 in range(0, len(ts), BLOCK):
            block = np.zeros((BLOCK, N_COLS))
            for j, t in enumerate(ts[b0:b0 + BLOCK]):
                block[j] = np.concatenate(
                    [RULE_MEMBERSHIP[:, t - 1], kmem[:, :, :, t - 1].reshape(-1), c_mem[:, :, t - 1].reshape(-1)]
                )
                slot_of[(s, t)] = len(blocks) * BLOCK + j
            blocks.append(block)
            block_set.append(si)
    return {
        "rule_log_lik": np.stack(rule_ll),
        "group_k_log_count": np.stack(gkc),
        "group_k_n": np.stack(gkn),
        "conj_log_evidence": np.stack(conj_ev),
        "slot_rule": np.stack(slots),
        "top_is_last": np.array(lasts),
        "block_set": np.array(block_set, dtype="int64"),
        "block_in_hypothesis": np.stack(blocks),
        "row_cell": np.array([slot_of[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
