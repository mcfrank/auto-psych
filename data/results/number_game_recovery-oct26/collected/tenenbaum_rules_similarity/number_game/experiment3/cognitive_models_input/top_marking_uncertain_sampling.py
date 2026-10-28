"""Top-marking uncertain sampling: refinement of top_marking_range_sampling.
Bayesian concept learning over 42 rules (kind-level prior, primes excluding 1),
every interval of 1-100 (Erlang width prior, decade-boundary bonus) and
range-restricted rules, 10% uniform examples, "yes" = posterior probability the
concept contains the target; with inferred odds the program deliberately shows
its range's top number and draws the other examples at random, so a range
ending at the largest example is credited by 1 + odds * (share of examples
equal to the top) / (chance a random example is that number). The one change,
grafted from uncertain_sampling_assumption: people are unsure how the examples
were generated - with an inferred probability they assume strong sampling
(examples drawn from the concept: size principle), otherwise weak sampling
(examples merely accepted by the concept: likelihood ignores the concept's
size, a random example hitting any number with chance 1/100). The posterior
runs jointly over concept and sampling assumption, so broad concepts keep a
floor of support however many examples fit a narrower one.
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
# Row 0: strong sampling; row 1: weak sampling (any accepted number, 1/100).
GROUP_DRAW_PROB = np.stack([(1 - EPSILON) / GROUP_WIDTHS + EPSILON / 100, np.full(len(GROUP_WIDTHS), 0.01)])
N_HYP = N_RULES + 2 * N_GROUPS + MAX_SLOTS * N_GROUPS
# Targets of one set are handled in blocks of this many (one block per row).
BLOCK = 8
SAMPLER_SETTINGS = {"target_accept": 0.9}


def log_lik(member, size, examples, strong=True):
    """Log-likelihood of the examples under strong (size principle) or weak
    sampling; member: (..., 100), size: (...)."""
    safe = (np.maximum(size, 1.0) if strong else np.full(np.shape(size), 100.0))[..., None]
    per = np.log(member[..., examples - 1] * (1 - EPSILON) / safe + EPSILON / 100)
    return per.sum(axis=-1)


def set_quantities(examples, strong):
    """Per-set evidence of rules, interval groups and conjunction groups (for
    the rules consistent with every example), and their within-group
    predictive membership over 1-100."""
    rule_ll = log_lik(RULE_MEMBERSHIP, RULE_MEMBERSHIP.sum(1), examples, strong)
    int_ll = log_lik(INTERVAL_MEMBERSHIP, INTERVAL_MEMBERSHIP.sum(1), examples, strong)
    consistent = np.flatnonzero(RULE_MEMBERSHIP[:, examples - 1].min(axis=1) > 0)[:MAX_SLOTS]
    slot_rule = np.zeros(MAX_SLOTS, dtype="int64")
    slot_rule[: len(consistent)] = consistent
    slot_used = np.arange(MAX_SLOTS) < len(consistent)
    rm = RULE_MEMBERSHIP[slot_rule]
    size = CONJ_SIZE[slot_rule]
    in_both = rm[:, None, examples - 1] * INTERVAL_MEMBERSHIP[None, :, examples - 1]
    norm = np.maximum(size, 1.0) if strong else np.full(size.shape, 100.0)
    conj_ll = np.log(in_both * (1 - EPSILON) / norm[..., None] + EPSILON / 100).sum(-1)
    conj_ll = np.where(size > 0, conj_ll, NEG)
    # Plain ranges split by whether their upper end is the largest example.
    at_top = np.array([n == examples.max() for _, n in BOUNDS])
    int_ev = np.full((2, N_GROUPS), NEG)
    int_mem = np.zeros((2, N_GROUPS, 100))
    conj_ev = np.empty((MAX_SLOTS, N_GROUPS))
    conj_mem = np.empty((MAX_SLOTS, N_GROUPS, 100))
    for g, idx in enumerate(GROUP_MEMBERS):
        for k, part in enumerate([idx[~at_top[idx]], idx[at_top[idx]]]):
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
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, 2, N_RULES)))
    group_log_evidence = pm.Data("group_log_evidence", np.zeros((1, 2, 2, N_GROUPS)))
    conj_log_evidence = pm.Data("conj_log_evidence", np.zeros((1, 2, MAX_SLOTS, N_GROUPS)))
    top_share = pm.Data("top_share", np.zeros(1))
    slot_rule = pm.Data("slot_rule", np.zeros((1, MAX_SLOTS), dtype="int64"))
    cell_in_hypothesis = pm.Data("cell_in_hypothesis", np.zeros((1, BLOCK, 2 * N_HYP)))
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
    # Probability people assume the examples were sampled from the concept.
    strong_prob = pm.Beta("strong_prob", alpha=2.0, beta=2.0)

    log_interval = np.log(GROUP_WIDTHS) - GROUP_WIDTHS / width_scale + decade_boost * GROUP_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval + np.log(GROUP_SIZES))
    log_rw = pt.log(rule_weights)
    log_mode = pt.stack([pt.log(strong_prob), pt.log1p(-strong_prob)])
    n_sets = rule_log_lik.shape[0]

    log_rules = pt.log(shares[0]) + log_rw[None, None, :] + rule_log_lik
    top_credit = pt.log1p(top_odds * top_share[:, None, None] / GROUP_DRAW_PROB[None, :, :])
    top_shift = pt.stack([pt.zeros_like(top_credit), top_credit], axis=2)
    log_groups = (
        pt.log(shares[1]) + log_interval_prior[None, None, None, :] + group_log_evidence + top_shift
    ).reshape((n_sets, 2, 2 * N_GROUPS))
    log_conj = (
        pt.log(shares[2]) + log_rw[slot_rule][:, None, :, None] + log_interval_prior[None, None, None, :]
        + conj_log_evidence
    ).reshape((n_sets, 2, MAX_SLOTS * N_GROUPS))
    log_joint = (pt.concatenate([log_rules, log_groups, log_conj], axis=2) + log_mode[None, :, None]).reshape(
        (n_sets, 2 * N_HYP)
    )
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = pt.batched_dot(cell_in_hypothesis, posterior[:, :, None]).reshape((-1,))

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Per-set quantities, one row per block of up to BLOCK distinct targets
    of a set (a set with many targets fills several rows)."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    by_set = {}
    for s, t in dict.fromkeys(zip(sets, targets)):
        by_set.setdefault(s, []).append(t)
    rule_ll, int_ev, conj_ev, slots, tops, blocks = [], [], [], [], [], []
    cell_index = {}
    for s, ts in by_set.items():
        modes = [set_quantities(np.array(s), strong) for strong in (True, False)]
        for b in range(0, len(ts), BLOCK):
            block = np.zeros((BLOCK, 2 * N_HYP))
            for j, t in enumerate(ts[b : b + BLOCK]):
                cell_index[(s, t)] = len(blocks) * BLOCK + j
                block[j] = np.concatenate(
                    [
                        np.concatenate([RULE_MEMBERSHIP[:, t - 1], m[2][:, :, t - 1].reshape(-1), m[4][:, :, t - 1].reshape(-1)])
                        for m in modes
                    ]
                )
            blocks.append(block)
            rule_ll.append(np.stack([m[0] for m in modes]))
            int_ev.append(np.stack([m[1] for m in modes]))
            conj_ev.append(np.stack([m[3] for m in modes]))
            slots.append(modes[0][5])
            tops.append(float(np.mean(np.array(s) == max(s))))
    return {
        "rule_log_lik": np.stack(rule_ll),
        "group_log_evidence": np.stack(int_ev),
        "conj_log_evidence": np.stack(conj_ev),
        "slot_rule": np.stack(slots),
        "top_share": np.array(tops),
        "cell_in_hypothesis": np.stack(blocks),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
