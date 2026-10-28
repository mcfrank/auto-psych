"""Noise-rate-uncertain range-restricted concepts: refinement of
range_restricted_rule_concepts. Bayesian concept learning over 42 rules (kind-
level prior), every interval of 1-100 (Erlang width prior with a decade-
boundary bonus) and conjunctions "rule r within interval I", size principle,
"yes" = posterior probability the concept contains the target. The one change:
instead of assuming exactly 10% of examples are random, people are unsure how
many examples are random noise and infer the noise rate jointly with the
concept for each set, under an inferred prior over noise levels (2%, 10%, 30%,
60% of examples random). So a set with one stray example (8,24,48,63) can still
be read as a tight rule plus noise, while a set that fits a tight concept
exactly drives the noise belief down and sharpens generalization.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pytensor import sparse
from scipy.special import logsumexp

NOISE_LEVELS = np.array([0.02, 0.1, 0.3, 0.6])
N_LEVELS = len(NOISE_LEVELS)
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

# Size of each conjunction "rule r within interval I" (rules x intervals).
CONJ_SIZE = RULE_MEMBERSHIP @ INTERVAL_MEMBERSHIP.T
NEG = -1e4
MAX_SLOTS = 12


def log_lik(member, size, examples, eps):
    """Size-principle log-likelihood of the examples, a share eps of them drawn
    uniformly from 1-100; member: (..., 100), size: (...)."""
    safe = np.maximum(size, 1.0)[..., None]
    per = np.log(member[..., examples - 1] * (1 - eps) / safe + eps / 100)
    return per.sum(axis=-1)


def set_quantities(examples, eps):
    """Per-set evidence of rules, interval groups and conjunction groups (for
    the rules consistent with every example), and their within-group
    predictive membership over 1-100."""
    rule_ll = log_lik(RULE_MEMBERSHIP, RULE_MEMBERSHIP.sum(1), examples, eps)
    int_ll = log_lik(INTERVAL_MEMBERSHIP, INTERVAL_MEMBERSHIP.sum(1), examples, eps)
    consistent = np.flatnonzero(RULE_MEMBERSHIP[:, examples - 1].min(axis=1) > 0)[:MAX_SLOTS]
    slot_rule = np.zeros(MAX_SLOTS, dtype="int64")
    slot_rule[: len(consistent)] = consistent
    slot_used = np.arange(MAX_SLOTS) < len(consistent)
    rm = RULE_MEMBERSHIP[slot_rule]
    size = CONJ_SIZE[slot_rule]
    in_both = rm[:, None, examples - 1] * INTERVAL_MEMBERSHIP[None, :, examples - 1]
    conj_ll = np.log(in_both * (1 - eps) / np.maximum(size, 1.0)[..., None] + eps / 100).sum(-1)
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


N_HYP = N_RULES + N_GROUPS + MAX_SLOTS * N_GROUPS
# Hypotheses whose log evidence trails the set's best by more than this are
# dropped (their posterior is below e^-45 of the best's under any plausible
# prior), so the posterior is held only over the live (set, noise level,
# hypothesis) entries.
PRUNE = 45.0

with pm.Model() as model:
    # Live hypothesis entries, sorted by set: their log evidence, noise level,
    # type (0 rule, 1 range, 2 range-restricted rule), rule and range group.
    entry_evidence = pm.Data("entry_evidence", np.zeros(1, dtype="float64"))
    entry_level = pm.Data("entry_level", np.zeros(1, dtype="int64"))
    entry_type = pm.Data("entry_type", np.zeros(1, dtype="int64"))
    entry_rule = pm.Data("entry_rule", np.zeros(1, dtype="int64"))
    entry_group = pm.Data("entry_group", np.zeros(1, dtype="int64"))
    entry_set = pm.Data("entry_set", np.zeros(1, dtype="int64"))
    set_start = pm.Data("set_start", np.zeros(2, dtype="int64"))
    # Per cell (set, target): the live entries whose concept may contain the
    # target, with the within-hypothesis probability that it does.
    cell_start = pm.Data("cell_start", np.zeros(2, dtype="int64"))
    cell_entry = pm.Data("cell_entry", np.zeros(1, dtype="int64"))
    cell_entry_in = pm.Data("cell_entry_in", np.zeros(1, dtype="float64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior shares of plain rules, plain ranges and range-restricted rules.
    shares = pm.Dirichlet("shares", a=np.array([2.0, 2.0, 1.0]))
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    rule_weights = pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS)
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    decade_boost = pm.Normal("decade_boost", mu=1.0, sigma=1.5)
    # People's prior belief over how many of the examples are random noise.
    noise_prior = pm.Dirichlet("noise_prior", a=np.full(N_LEVELS, 1.5))

    log_interval = np.log(GROUP_WIDTHS) - GROUP_WIDTHS / width_scale + decade_boost * GROUP_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval + np.log(GROUP_SIZES))
    has_rule = pt.neq(entry_type, 1)
    has_group = pt.neq(entry_type, 0)
    log_prior = (
        pt.log(shares)[entry_type]
        + has_rule * pt.log(rule_weights)[entry_rule]
        + has_group * log_interval_prior[entry_group]
        + pt.log(noise_prior)[entry_level]
    )
    n_entries = entry_evidence.shape[0]
    n_sets = set_start.shape[0] - 1
    weight = pt.exp(entry_evidence + log_prior)
    # Sum of weights per set (entries are contiguous by set).
    by_set = sparse.CSR(pt.ones(n_entries), pt.arange(n_entries), set_start, pt.stack([n_sets, n_entries]))
    total = sparse.structured_dot(by_set, weight[:, None])[:, 0]
    posterior = weight / total[entry_set]
    table = sparse.CSR(cell_entry_in, cell_entry, cell_start, pt.stack([cell_start.shape[0] - 1, n_entries]))
    in_concept = sparse.structured_dot(table, posterior[:, None])[:, 0]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    by_set = {}
    for ci, (s, t) in enumerate(cells):
        by_set.setdefault(s, []).append((ci, t))
    # Generic layout of one noise level's hypotheses: type, group, slot.
    h_type = np.concatenate([np.zeros(N_RULES), np.ones(N_GROUPS), np.full(MAX_SLOTS * N_GROUPS, 2)]).astype("int64")
    h_group = np.concatenate([np.zeros(N_RULES), np.arange(N_GROUPS), np.tile(np.arange(N_GROUPS), MAX_SLOTS)]).astype("int64")
    h_slot = np.concatenate([np.zeros(N_RULES + N_GROUPS), np.repeat(np.arange(MAX_SLOTS), N_GROUPS)]).astype("int64")
    ev_l, lv_l, ty_l, ru_l, gr_l, se_l, set_counts = [], [], [], [], [], [], []
    cell_parts = [None] * len(cells)
    offset = 0
    for si, s in enumerate(unique):
        per_level = [set_quantities(np.array(s), eps) for eps in NOISE_LEVELS]
        slot = per_level[0][5]
        h_rule = np.concatenate([np.arange(N_RULES), np.zeros(N_GROUPS, dtype="int64"), slot[h_slot[N_RULES + N_GROUPS:]]])
        evidence = np.stack([np.concatenate([q[0], q[1], q[3].reshape(-1)]) for q in per_level])
        live_level, live_hyp = np.nonzero(evidence > evidence.max() - PRUNE)
        ev_l.append(evidence[live_level, live_hyp])
        lv_l.append(live_level)
        ty_l.append(h_type[live_hyp])
        ru_l.append(h_rule[live_hyp])
        gr_l.append(h_group[live_hyp])
        se_l.append(np.full(len(live_hyp), si))
        set_counts.append(len(live_hyp))
        for ci, t in by_set[s]:
            member = np.stack(
                [np.concatenate([RULE_MEMBERSHIP[:, t - 1], q[2][:, t - 1], q[4][:, :, t - 1].reshape(-1)]) for q in per_level]
            )
            val = member[live_level, live_hyp]
            keep = np.flatnonzero(val > 0)
            cell_parts[ci] = (offset + keep, val[keep])
        offset += len(live_hyp)
    # Shift each set's log evidence by its maximum, so weights stay finite.
    evidence = np.concatenate(ev_l)
    entry_set = np.concatenate(se_l)
    set_max = np.array([e.max() for e in ev_l])
    return {
        "entry_evidence": evidence - set_max[entry_set],
        "entry_level": np.concatenate(lv_l).astype("int64"),
        "entry_type": np.concatenate(ty_l).astype("int64"),
        "entry_rule": np.concatenate(ru_l).astype("int64"),
        "entry_group": np.concatenate(gr_l).astype("int64"),
        "entry_set": entry_set.astype("int64"),
        "set_start": np.concatenate([[0], np.cumsum(set_counts)]).astype("int64"),
        "cell_start": np.concatenate([[0], np.cumsum([len(p[0]) for p in cell_parts])]).astype("int64"),
        "cell_entry": np.concatenate([p[0] for p in cell_parts]).astype("int64"),
        "cell_entry_in": np.concatenate([p[1] for p in cell_parts]).astype("float64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
