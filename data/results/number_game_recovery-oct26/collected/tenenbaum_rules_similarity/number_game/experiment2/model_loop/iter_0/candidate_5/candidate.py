"""Random forgetting with range-restricted rules: refinement of
random_example_forgetting. People do Bayesian concept learning over 42 rules
(kind-level prior), every interval of 1-100 (Erlang width prior, decade-boundary
bonus) and -- the one change, grafted from range_restricted_rule_concepts --
conjunctions "rule r within interval I" with their own prior share (prior
share * rule prior(r) * interval prior(I)), using the size principle with 10%
uniform examples. Each shown example is independently forgotten with an
inferred probability (at least one is kept); "yes" is the average over the
remembered subsets of the posterior probability that the concept contains the
target.

Implementation note: given each subset's (constant) evidence tables, the
posterior predictive is linear in the prior weights, so numerator and
denominator are computed as matrix-vector products with precomputed tables.
"""

import itertools

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
    rules.append([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
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

CONJ_SIZE = RULE_MEMBERSHIP @ INTERVAL_MEMBERSHIP.T
NEG = -1e4
MAX_SLOTS = 12


def log_lik(member, size, examples):
    safe = np.maximum(size, 1.0)[..., None]
    per = np.log(member[..., examples - 1] * (1 - EPSILON) / safe + EPSILON / 100)
    return per.sum(axis=-1)


def set_quantities(examples):
    """Evidence of rules, interval groups and conjunction groups for one
    remembered subset, and their within-group membership over 1-100."""
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


with pm.Model() as model:
    # Per remembered subset: scaled evidence tables (exp(log evidence - offset)).
    sub_rule = pm.Data("sub_rule", np.zeros((1, N_RULES)))
    sub_group = pm.Data("sub_group", np.zeros((1, N_GROUPS)))
    sub_conj = pm.Data("sub_conj", np.zeros((1 * MAX_SLOTS, N_GROUPS)))
    sub_slot_rule = pm.Data("sub_slot_rule", np.zeros((1, MAX_SLOTS), dtype="int64"))
    # Per (cell, subset) pair: the same tables times membership of the target.
    pair_rule = pm.Data("pair_rule", np.zeros((1, N_RULES)))
    pair_group = pm.Data("pair_group", np.zeros((1, N_GROUPS)))
    pair_conj = pm.Data("pair_conj", np.zeros((1 * MAX_SLOTS, N_GROUPS)))
    pair_subset = pm.Data("pair_subset", np.zeros(1, dtype="int64"))
    pair_cell = pm.Data("pair_cell", np.zeros(1, dtype="int64"))
    pair_kept = pm.Data("pair_kept", np.ones(1, dtype="float64"))
    pair_lost = pm.Data("pair_lost", np.zeros(1, dtype="float64"))
    cell_set_size = pm.Data("cell_set_size", np.ones(1, dtype="float64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior shares of plain rules, plain ranges and range-restricted rules.
    shares = pm.Dirichlet("shares", a=np.array([2.0, 2.0, 1.0]))
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    rule_weights = pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS)
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    decade_boost = pm.Normal("decade_boost", mu=1.0, sigma=1.5)
    # Probability that any one example is lost from memory.
    forget = pm.Beta("forget", 1.5, 4.0)

    log_interval = np.log(GROUP_WIDTHS) - GROUP_WIDTHS / width_scale + decade_boost * GROUP_ALIGNED
    interval_prior = pt.exp(log_interval - pt.logsumexp(log_interval + np.log(GROUP_SIZES)))

    def mass(rule_tab, group_tab, conj_tab, slot_w):
        conj = pt.dot(conj_tab, interval_prior).reshape((slot_w.shape[0], MAX_SLOTS))
        return (
            shares[0] * pt.dot(rule_tab, rule_weights)
            + shares[1] * pt.dot(group_tab, interval_prior)
            + shares[2] * (conj * slot_w).sum(axis=1)
        )

    sub_slot_w = rule_weights[sub_slot_rule]
    denom = mass(sub_rule, sub_group, sub_conj, sub_slot_w)
    numer = mass(pair_rule, pair_group, pair_conj, sub_slot_w[pair_subset])
    pair_yes = numer / denom[pair_subset]

    # Subset weight forget^lost (1-forget)^kept, renormalized over non-empty subsets.
    w = pt.exp(pair_lost * pt.log(forget) + pair_kept * pt.log1p(-forget))
    norm = 1.0 - forget ** cell_set_size
    cell_yes = pt.inc_subtensor(pt.zeros(cell_set_size.shape[0])[pair_cell], w * pair_yes) / norm

    p_yes = pm.Deterministic("p_yes", pt.clip(cell_yes[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}

    subsets, subset_list, cell_pairs = {}, [], []
    for s, _ in cells:
        pairs = []
        n = len(s)
        for k in range(1, n + 1):
            for idx in itertools.combinations(range(n), k):
                key = tuple(sorted(s[i] for i in idx))
                if key not in subsets:
                    subsets[key] = len(subset_list)
                    subset_list.append(key)
                pairs.append((subsets[key], k, n - k))
        cell_pairs.append(pairs)

    quantities = []
    s_rule, s_group, s_conj, s_slot = [], [], [], []
    for key in subset_list:
        r_ll, i_ev, i_mem, c_ev, c_mem, slot = set_quantities(np.array(key))
        offset = max(r_ll.max(), i_ev.max(), c_ev.max())
        er, eg, ec = np.exp(r_ll - offset), np.exp(i_ev - offset), np.exp(c_ev - offset)
        quantities.append((er, eg, i_mem, ec, c_mem))
        s_rule.append(er)
        s_group.append(eg)
        s_conj.append(ec)
        s_slot.append(slot)

    p_rule, p_group, p_conj = [], [], []
    p_subset, p_cell, p_kept, p_lost = [], [], [], []
    for c, ((s, t), pairs) in enumerate(zip(cells, cell_pairs)):
        for sub, kept, lost in pairs:
            er, eg, i_mem, ec, c_mem = quantities[sub]
            p_rule.append(er * RULE_MEMBERSHIP[:, t - 1])
            p_group.append(eg * i_mem[:, t - 1])
            p_conj.append(ec * c_mem[:, :, t - 1])
            p_subset.append(sub)
            p_cell.append(c)
            p_kept.append(kept)
            p_lost.append(lost)

    return {
        "sub_rule": np.stack(s_rule),
        "sub_group": np.stack(s_group),
        "sub_conj": np.concatenate(s_conj, axis=0),
        "sub_slot_rule": np.stack(s_slot).astype("int64"),
        "pair_rule": np.stack(p_rule),
        "pair_group": np.stack(p_group),
        "pair_conj": np.concatenate(p_conj, axis=0),
        "pair_subset": np.array(p_subset, dtype="int64"),
        "pair_cell": np.array(p_cell, dtype="int64"),
        "pair_kept": np.array(p_kept, dtype="float64"),
        "pair_lost": np.array(p_lost, dtype="float64"),
        "cell_set_size": np.array([float(len(s)) for s, _ in cells]),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
