"""Parity-representative sequence edges: refinement of
sequence_opens_bottom_ends_top. Bayesian concept learning over 42 rules
(kind-level prior, primes excluding 1), every interval of 1-100 (Erlang width
prior, decade-boundary bonus) and range-restricted rules, size principle with
10% uniform examples, "yes" = posterior probability the concept contains the
target; a sequence ending on its largest example marks its range's top and a
sequence of several opening on its smallest marks its bottom (generative
credit 1 + odds / chance of hitting that number). The one change, taken from
parity_representativeness_bias: people judge how likely the examples are under
a concept partly by how representative they look of it. A concept holding both
odd and even numbers (a range, multiples of 3, squares, ...) is expected to
yield a parity-mixed sample, so each example beyond the first that repeats a
parity shared by the whole set costs an inferred extra log-penalty under it;
parity-pure concepts (even, odd, multiples of 4, ...) pay nothing.
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
# Probability a random draw from a range of each width group hits a given member.
GROUP_DRAW_PROB = (1 - EPSILON) / GROUP_WIDTHS + EPSILON / 100
# Concepts containing both odd and even numbers (whose samples should look mixed).
RULE_MIXED = np.array([len({x % 2 for x in c}) > 1 for c in RULES], dtype="float64")
GROUP_MIXED = (GROUP_WIDTHS >= 2).astype("float64")


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
    # Plain ranges split by whether their top is the largest example and
    # whether their bottom is the smallest (part = at_top + 2 * at_bottom).
    at_top = np.array([n == examples.max() for _, n in BOUNDS])
    at_bot = np.array([m == examples.min() for m, _ in BOUNDS])
    part_of = at_top.astype(int) + 2 * at_bot.astype(int)
    int_ev = np.full((4, N_GROUPS), NEG)
    int_mem = np.zeros((4, N_GROUPS, 100))
    conj_ev = np.empty((MAX_SLOTS, N_GROUPS))
    conj_mem = np.empty((MAX_SLOTS, N_GROUPS, 100))
    for g, idx in enumerate(GROUP_MEMBERS):
        for k in range(4):
            part = idx[part_of[idx] == k]
            if len(part) == 0:
                continue
            ll = int_ll[part]
            int_ev[k, g] = logsumexp(ll)
            int_mem[k, g] = np.exp(ll - int_ev[k, g]) @ INTERVAL_MEMBERSHIP[part]
        cl = conj_ll[:, idx]
        ev = logsumexp(cl, axis=1)
        conj_ev[:, g] = ev
        w = np.exp(cl - ev[:, None])
        conj_mem[:, g] = (w @ INTERVAL_MEMBERSHIP[idx]) * rm
    conj_ev[~slot_used] = NEG
    conj_mem[~slot_used] = 0.0
    return rule_ll, int_ev, int_mem, conj_ev, conj_mem, slot_rule


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    group_log_evidence = pm.Data("group_log_evidence", np.zeros((1, 4, N_GROUPS)))
    conj_log_evidence = pm.Data("conj_log_evidence", np.zeros((1, MAX_SLOTS, N_GROUPS)))
    top_share = pm.Data("top_share", np.zeros(1))
    bottom_share = pm.Data("bottom_share", np.zeros(1))
    parity_repeats = pm.Data("parity_repeats", np.zeros(1))
    slot_rule = pm.Data("slot_rule", np.zeros((1, MAX_SLOTS), dtype="int64"))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_in_hypothesis = pm.Data("cell_in_hypothesis", np.zeros((1, N_RULES + 4 * N_GROUPS + MAX_SLOTS * N_GROUPS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior shares of plain rules, plain ranges and range-restricted rules.
    shares = pm.Dirichlet("shares", a=np.array([2.0, 2.0, 1.0]))
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    rule_weights = pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS)
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    decade_boost = pm.Normal("decade_boost", mu=1.0, sigma=1.5)
    # Odds that the program deliberately ends on its range's top, and that it
    # deliberately opens on its range's bottom.
    top_odds = pm.LogNormal("top_odds", mu=np.log(0.1), sigma=1.5)
    bottom_odds = pm.LogNormal("bottom_odds", mu=np.log(0.1), sigma=1.5)

    # Extra log-penalty per repeated shared parity under a parity-mixed concept.
    parity_penalty = pm.HalfNormal("parity_penalty", sigma=1.0)
    unrep = parity_penalty * parity_repeats[:, None]

    log_interval = np.log(GROUP_WIDTHS) - GROUP_WIDTHS / width_scale + decade_boost * GROUP_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval + np.log(GROUP_SIZES))
    log_rw = pt.log(rule_weights)

    log_rules = pt.log(shares[0]) + log_rw[None, :] + rule_log_lik - unrep * RULE_MIXED[None, :]
    top_credit = pt.log1p(top_odds * top_share[:, None] / GROUP_DRAW_PROB[None, :])
    bottom_credit = pt.log1p(bottom_odds * bottom_share[:, None] / GROUP_DRAW_PROB[None, :])
    zero = pt.zeros_like(top_credit)
    top_shift = pt.stack([zero, top_credit, bottom_credit, top_credit + bottom_credit], axis=1)
    log_groups = (
        pt.log(shares[1]) + log_interval_prior[None, None, :] + group_log_evidence + top_shift
        - unrep[:, :, None] * GROUP_MIXED[None, None, :]
    ).reshape((rule_log_lik.shape[0], 4 * N_GROUPS))
    log_conj = (
        pt.log(shares[2]) + log_rw[slot_rule][:, :, None] + log_interval_prior[None, None, :] + conj_log_evidence
        - unrep[:, :, None] * pt.as_tensor_variable(RULE_MIXED)[slot_rule][:, :, None]
    ).reshape((log_rules.shape[0], MAX_SLOTS * N_GROUPS))
    log_joint = pt.concatenate([log_rules, log_groups, log_conj], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[cell_set] * cell_in_hypothesis).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    by_set = {}
    for ci, (s, t) in enumerate(cells):
        by_set.setdefault(s, []).append((ci, t))
    rule_ll, int_ev, conj_ev, slots, tops, bots, repeats = [], [], [], [], [], [], []
    cell_rows = [None] * len(cells)
    for s in unique:
        r_ll, i_ev, i_mem, c_ev, c_mem, slot = set_quantities(np.array(s))
        rule_ll.append(r_ll)
        int_ev.append(i_ev)
        conj_ev.append(c_ev)
        slots.append(slot)
        # The top is marked when the sequence ends on its largest example; the
        # bottom when a sequence of several opens on its smallest.
        tops.append(1.0 if s[-1] == max(s) else 0.0)
        bots.append(1.0 if len(s) > 1 and s[0] == min(s) and min(s) < max(s) else 0.0)
        repeats.append(float(len(s) - 1) if len({x % 2 for x in s}) == 1 else 0.0)
        for ci, t in by_set[s]:
            cell_rows[ci] = np.concatenate(
                [RULE_MEMBERSHIP[:, t - 1], i_mem[:, :, t - 1].reshape(-1), c_mem[:, :, t - 1].reshape(-1)]
            )
    return {
        "rule_log_lik": np.stack(rule_ll),
        "group_log_evidence": np.stack(int_ev),
        "conj_log_evidence": np.stack(conj_ev),
        "slot_rule": np.stack(slots),
        "top_share": np.array(tops),
        "bottom_share": np.array(bots),
        "parity_repeats": np.array(repeats),
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_in_hypothesis": np.stack(cell_rows),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
