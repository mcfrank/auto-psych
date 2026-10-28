"""Spread-sampled ranges. People do the same Bayesian concept learning as
last_example_marks_range_top (arithmetic rules with primes excluding 1, every
range of 1-100, range-restricted rules, the last-shown largest example marking
a range's top), but with an inferred probability they believe the program
spreads a range's examples evenly across it (the range is cut into as many
equal stretches as there are examples and one example is drawn from each)
instead of drawing them at random. Several evenly spaced examples are then far
likelier under the range they span than under a rule, so "yes" inside the span
of widely, evenly spaced sets (20,35,68,92) rises regardless of parity, while
ranges reaching well past the outermost examples, or covering clumped examples,
get no such boost.

Base: Bayesian concept learning over 42 rules (kind-level prior), every
interval of 1-100 (Erlang width prior, decade-boundary bonus) and
range-restricted rules, size principle with 10% uniform examples, "yes" =
posterior probability the concept contains the target; a range ending at a
last-shown largest example gets credit 1 + odds / (chance a draw hits it).
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
LOG_RULE_SIZE = np.log(RULE_MEMBERSHIP.sum(1))
BOUND_LO = np.array([m for m, _ in BOUNDS])
CONJ_SIZE = RULE_MEMBERSHIP @ INTERVAL_MEMBERSHIP.T
NEG = -1e4
MAX_SLOTS = 12
# Probability a random draw from a range of each width group hits a given member.
GROUP_DRAW_PROB = (1 - EPSILON) / GROUP_WIDTHS + EPSILON / 100


def log_lik(member, size, examples):
    """Size-principle log-likelihood of the examples; member: (..., 100), size: (...)."""
    safe = np.maximum(size, 1.0)[..., None]
    per = np.log(member[..., examples - 1] * (1 - EPSILON) / safe + EPSILON / 100)
    return per.sum(axis=-1)


def spread_log_lik(examples):
    """Log-likelihood of the examples for every range if the program spreads its
    examples across the range: the range is cut into n equal strata and one
    example is drawn from each (10% of examples still uniform on 1-100)."""
    n = len(examples)
    width = INTERVAL_WIDTH.astype("float64")
    offset = examples[None, :] - BOUND_LO[:, None]
    inside = (offset >= 0) & (offset < width[:, None])
    stratum = np.where(inside, np.floor(offset * n / width[:, None]), -1).astype("int64")
    occupancy = np.stack([(stratum == j).sum(1) for j in range(n)], axis=1)
    alone = inside & (np.take_along_axis(occupancy, np.clip(stratum, 0, n - 1), axis=1) == 1)
    per = np.log(alone * (1 - EPSILON) * n / width[:, None] + EPSILON / 100)
    return per.sum(axis=1)


def set_quantities(examples):
    """Per-set evidence of rules, interval groups and conjunction groups (for
    the rules consistent with every example), and their within-group
    predictive membership over 1-100."""
    rule_ll = log_lik(RULE_MEMBERSHIP, RULE_MEMBERSHIP.sum(1), examples)
    int_ll = log_lik(INTERVAL_MEMBERSHIP, INTERVAL_MEMBERSHIP.sum(1), examples)
    spread_ll = spread_log_lik(examples)
    consistent = np.flatnonzero(RULE_MEMBERSHIP[:, examples - 1].min(axis=1) > 0)[:MAX_SLOTS]
    slot_rule = np.zeros(MAX_SLOTS, dtype="int64")
    slot_rule[: len(consistent)] = consistent
    slot_used = np.arange(MAX_SLOTS) < len(consistent)
    rm = RULE_MEMBERSHIP[slot_rule]
    size = CONJ_SIZE[slot_rule]
    in_both = rm[:, None, examples - 1] * INTERVAL_MEMBERSHIP[None, :, examples - 1]
    conj_ll = np.log(in_both * (1 - EPSILON) / np.maximum(size, 1.0)[..., None] + EPSILON / 100).sum(-1)
    conj_ll = np.where(size > 0, conj_ll, NEG)
    # Plain ranges split by whether their upper end is the largest example.
    at_top = np.array([n == examples.max() for _, n in BOUNDS])
    int_ev = np.full((2, 2, N_GROUPS), NEG)
    int_mem = np.zeros((2, 2, N_GROUPS, 100))
    conj_ev = np.empty((MAX_SLOTS, N_GROUPS))
    conj_mem = np.empty((MAX_SLOTS, N_GROUPS, 100))
    for g, idx in enumerate(GROUP_MEMBERS):
        for k, part in enumerate([idx[~at_top[idx]], idx[at_top[idx]]]):
            if len(part) == 0:
                continue
            for mode, lls in enumerate([int_ll, spread_ll]):
                ll = lls[part]
                ev = logsumexp(ll)
                if ev < NEG / 2:
                    continue
                int_ev[mode, k, g] = ev
                int_mem[mode, k, g] = np.exp(ll - ev) @ INTERVAL_MEMBERSHIP[part]
        cl = conj_ll[:, idx]
        ev = logsumexp(cl, axis=1)
        conj_ev[:, g] = ev
        w = np.exp(cl - ev[:, None])
        conj_mem[:, g] = (w @ INTERVAL_MEMBERSHIP[idx]) * rm
    conj_ev[~slot_used] = NEG
    conj_mem[~slot_used] = 0.0
    return rule_ll, int_ev, int_mem, conj_ev, conj_mem, slot_rule


with pm.Model() as model:
    rule_hits = pm.Data("rule_hits", np.zeros((1, N_RULES)))
    n_examples = pm.Data("n_examples", np.ones(1))
    group_log_evidence = pm.Data("group_log_evidence", np.zeros((1, 2, N_GROUPS)))
    spread_log_evidence = pm.Data("spread_log_evidence", np.zeros((1, 2, N_GROUPS)))
    conj_log_evidence = pm.Data("conj_log_evidence", np.zeros((1, MAX_SLOTS, N_GROUPS)))
    top_share = pm.Data("top_share", np.zeros(1))
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
    # Odds that the program deliberately shows its range's top number.
    top_odds = pm.LogNormal("top_odds", mu=np.log(0.1), sigma=1.5)
    # Probability people think the examples were spread across the range.
    spread_prob = pm.Beta("spread_prob", alpha=2.0, beta=2.0)

    per_hit = np.log((1 - EPSILON) * np.exp(-LOG_RULE_SIZE) + EPSILON / 100)
    rule_log_lik = rule_hits * per_hit[None, :] + (n_examples[:, None] - rule_hits) * np.log(EPSILON / 100)

    log_interval = np.log(GROUP_WIDTHS) - GROUP_WIDTHS / width_scale + decade_boost * GROUP_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval + np.log(GROUP_SIZES))
    log_rw = pt.log(rule_weights)

    log_rules = pt.log(shares[0]) + log_rw[None, :] + rule_log_lik
    top_credit = pt.log1p(top_odds * top_share[:, None] / GROUP_DRAW_PROB[None, :])
    top_shift = pt.stack([pt.zeros_like(top_credit), top_credit], axis=1)
    # Under spread sampling a draw from the top stratum hits the top number
    # with probability n / width rather than 1 / width.
    spread_draw = pt.clip(
        (1 - EPSILON) * n_examples[:, None] / GROUP_WIDTHS[None, :] + EPSILON / 100, 1e-6, 1.0
    )
    spread_credit = pt.log1p(top_odds * top_share[:, None] / spread_draw)
    spread_shift = pt.stack([pt.zeros_like(spread_credit), spread_credit], axis=1)
    log_groups = (
        pt.log(shares[1]) + pt.log1p(-spread_prob) + log_interval_prior[None, None, :] + group_log_evidence + top_shift
    ).reshape((rule_log_lik.shape[0], 2 * N_GROUPS))
    log_spread = (
        pt.log(shares[1]) + pt.log(spread_prob) + log_interval_prior[None, None, :] + spread_log_evidence + spread_shift
    ).reshape((rule_log_lik.shape[0], 2 * N_GROUPS))
    log_conj = (
        pt.log(shares[2]) + log_rw[slot_rule][:, :, None] + log_interval_prior[None, None, :] + conj_log_evidence
    ).reshape((log_rules.shape[0], MAX_SLOTS * N_GROUPS))
    log_joint = pt.concatenate([log_rules, log_groups, log_spread, log_conj], axis=1)
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
    rule_ll, int_ev, conj_ev, slots, tops = [], [], [], [], []
    hits = [RULE_MEMBERSHIP[:, np.array(s) - 1].sum(1) for s in unique]
    cell_rows = [None] * len(cells)
    for s in unique:
        r_ll, i_ev, i_mem, c_ev, c_mem, slot = set_quantities(np.array(s))
        rule_ll.append(r_ll)
        int_ev.append(i_ev)
        conj_ev.append(c_ev)
        slots.append(slot)
        # Top marked only when the sequence ends on its largest example.
        tops.append(1.0 if s[-1] == max(s) else 0.0)
        for ci, t in by_set[s]:
            cell_rows[ci] = np.concatenate(
                [RULE_MEMBERSHIP[:, t - 1], i_mem[0, :, :, t - 1].reshape(-1), i_mem[1, :, :, t - 1].reshape(-1),
                 c_mem[:, :, t - 1].reshape(-1)]
            )
    return {
        "rule_hits": np.stack(hits),
        "n_examples": np.array([float(len(s)) for s in unique]),
        "group_log_evidence": np.stack([e[0] for e in int_ev]),
        "spread_log_evidence": np.stack([e[1] for e in int_ev]),
        "conj_log_evidence": np.stack(conj_ev),
        "slot_rule": np.stack(slots),
        "top_share": np.array(tops),
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_in_hypothesis": np.stack(cell_rows),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
