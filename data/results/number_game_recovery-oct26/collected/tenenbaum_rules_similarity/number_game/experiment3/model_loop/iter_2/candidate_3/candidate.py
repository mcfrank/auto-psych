"""Uncertain sampling on compressed_rule_extension (refinement of the incumbent).
Bayesian concept learning over 42 rules (kind-level prior, primes excluding 1),
every interval of 1-100 (Erlang width prior, decade-boundary bonus) and
range-restricted rules, 10% uniform examples, rule sizes perceived as
N**rule_size_power, the last-shown largest example marking a range's top,
"yes" = posterior probability the concept contains the target. The one change,
taken from uncertain_sampling_assumption: people are unsure how the examples
were chosen. With inferred probability strong_prob they were drawn from the
program's concept (size principle); otherwise they are merely numbers the
program accepts (weak sampling: every concept containing them keeps its prior
plausibility). "Yes" averages over both readings, so a third or fourth example
tightens generalization beyond the examples' span less than an ideal learner
predicts, and broad concepts keep a floor of support.
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
LOG_OUT = np.log(EPSILON / 100)
LOG_WEAK_HIT = np.log((1 - EPSILON) / 100 + EPSILON / 100)
# Chance one draw hits a given member of a range of each width group: strong, weak.
LOG_RANGE_HIT = [np.log((1 - EPSILON) / GROUP_WIDTHS + EPSILON / 100), np.full(N_GROUPS, LOG_WEAK_HIT)]


def set_quantities(examples):
    """Per-set evidence of rules, interval groups and conjunction groups (for
    the rules consistent with every example), and their within-group
    predictive membership over 1-100."""
    consistent = np.flatnonzero(RULE_MEMBERSHIP[:, examples - 1].min(axis=1) > 0)[:MAX_SLOTS]
    slot_rule = np.zeros(MAX_SLOTS, dtype="int64")
    slot_rule[: len(consistent)] = consistent
    slot_used = np.arange(MAX_SLOTS) < len(consistent)
    rm = RULE_MEMBERSHIP[slot_rule]
    size = CONJ_SIZE[slot_rule]
    in_both = rm[:, None, examples - 1] * INTERVAL_MEMBERSHIP[None, :, examples - 1]
    conj_ll = []
    for denom in (np.maximum(size, 1.0), np.full(size.shape, 100.0)):  # strong, weak
        ll = np.log(in_both * (1 - EPSILON) / denom[..., None] + EPSILON / 100).sum(-1)
        conj_ll.append(np.where(size > 0, ll, NEG))
    # Plain ranges split by whether their upper end is the largest example;
    # evidence per width group under strong and weak sampling.
    at_top = np.array([n == examples.max() for _, n in BOUNDS])
    hits = INTERVAL_MEMBERSHIP[:, examples - 1].sum(axis=1)
    n_out = len(examples) - hits
    int_ev = np.full((2, 2, N_GROUPS), NEG)
    int_mem = np.zeros((2, 2, N_GROUPS, 100))
    conj_ev = np.empty((2, MAX_SLOTS, N_GROUPS))
    conj_mem = np.empty((2, MAX_SLOTS, N_GROUPS, 100))
    for g, idx in enumerate(GROUP_MEMBERS):
        for k, part in enumerate([idx[~at_top[idx]], idx[at_top[idx]]]):
            if len(part) == 0:
                continue
            for mode in range(2):
                ll = hits[part] * LOG_RANGE_HIT[mode][g] + n_out[part] * LOG_OUT
                int_ev[mode, k, g] = logsumexp(ll)
                int_mem[mode, k, g] = np.exp(ll - int_ev[mode, k, g]) @ INTERVAL_MEMBERSHIP[part]
        for mode in range(2):
            cl = conj_ll[mode][:, idx]
            ev = logsumexp(cl, axis=1)
            conj_ev[mode, :, g] = ev
            w = np.exp(cl - ev[:, None])
            conj_mem[mode, :, g] = (w @ INTERVAL_MEMBERSHIP[idx]) * rm
    conj_ev[:, ~slot_used] = NEG
    conj_mem[:, ~slot_used] = 0.0
    return int_ev, int_mem, conj_ev, conj_mem, slot_rule


with pm.Model() as model:
    rule_count_in = pm.Data("rule_count_in", np.zeros((1, N_RULES)))
    rule_count_out = pm.Data("rule_count_out", np.zeros((1, N_RULES)))
    group_log_evidence = pm.Data("group_log_evidence", np.zeros((1, 2, 2, N_GROUPS)))
    conj_log_evidence = pm.Data("conj_log_evidence", np.zeros((1, 2, MAX_SLOTS, N_GROUPS)))
    top_share = pm.Data("top_share", np.zeros(1))
    slot_rule = pm.Data("slot_rule", np.zeros((1, MAX_SLOTS), dtype="int64"))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_in_hypothesis = pm.Data("cell_in_hypothesis", np.zeros((1, 2 * (N_RULES + 2 * N_GROUPS + MAX_SLOTS * N_GROUPS))))
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

    log_interval = np.log(GROUP_WIDTHS) - GROUP_WIDTHS / width_scale + decade_boost * GROUP_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval + np.log(GROUP_SIZES))
    log_rw = pt.log(rule_weights)
    # Probability the examples were drawn from the concept (strong sampling).
    strong_prob = pm.Beta("strong_prob", alpha=2.0, beta=2.0)
    log_mode = [pt.log(strong_prob), pt.log1p(-strong_prob)]
    # Perceived size of each rule's extension: N ** power (power < 1 compresses).
    rule_size_power = pm.Beta("rule_size_power", alpha=4.0, beta=2.0)
    log_rule_size = np.log(RULE_MEMBERSHIP.sum(1))
    # Chance one draw hits a given member of each rule: strong, weak.
    rule_hits = [pt.log((1 - EPSILON) * pt.exp(-rule_size_power * log_rule_size) + EPSILON / 100), pt.zeros(N_RULES) + LOG_WEAK_HIT]
    n_sets = rule_count_in.shape[0]
    blocks = []
    for mode in range(2):
        rule_log_lik = rule_count_in * rule_hits[mode][None, :] + rule_count_out * LOG_OUT
        blocks.append(log_mode[mode] + pt.log(shares[0]) + log_rw[None, :] + rule_log_lik)
        # Generative top credit: how unlikely a random draw was to hit the top.
        top_credit = pt.log1p(top_odds * top_share[:, None] / np.exp(LOG_RANGE_HIT[mode])[None, :])
        top_shift = pt.stack([pt.zeros_like(top_credit), top_credit], axis=1)
        log_groups = (
            log_mode[mode] + pt.log(shares[1]) + log_interval_prior[None, None, :]
            + group_log_evidence[:, mode] + top_shift
        )
        blocks.append(log_groups.reshape((n_sets, 2 * N_GROUPS)))
        log_conj = (
            log_mode[mode] + pt.log(shares[2]) + log_rw[slot_rule][:, :, None]
            + log_interval_prior[None, None, :] + conj_log_evidence[:, mode]
        )
        blocks.append(log_conj.reshape((n_sets, MAX_SLOTS * N_GROUPS)))
    log_joint = pt.concatenate(blocks, axis=1)
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
    c_in, c_out, int_ev, conj_ev, slots, tops = [], [], [], [], [], []
    cell_rows = [None] * len(cells)
    for s in unique:
        i_ev, i_mem, c_ev, c_mem, slot = set_quantities(np.array(s))
        hits = RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1)
        c_in.append(hits)
        c_out.append(len(s) - hits)
        int_ev.append(i_ev)
        conj_ev.append(c_ev)
        slots.append(slot)
        # Top marked only when the sequence ends on its largest example.
        tops.append(1.0 if s[-1] == max(s) else 0.0)
        for ci, t in by_set[s]:
            cell_rows[ci] = np.concatenate([
                np.concatenate([RULE_MEMBERSHIP[:, t - 1], i_mem[m, :, :, t - 1].reshape(-1), c_mem[m, :, :, t - 1].reshape(-1)])
                for m in range(2)
            ])
    return {
        "rule_count_in": np.stack(c_in),
        "rule_count_out": np.stack(c_out),
        "group_log_evidence": np.stack(int_ev),
        "conj_log_evidence": np.stack(conj_ev),
        "slot_rule": np.stack(slots),
        "top_share": np.array(tops),
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_in_hypothesis": np.stack(cell_rows),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
