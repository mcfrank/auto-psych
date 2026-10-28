"""Concept-confusion decision noise: people do plain Bayesian concept learning
over 42 arithmetic rules (kind-level prior, primes excluding 1) and every range
of 1-100 (Erlang width prior with a decade-boundary preference, size principle
with 10% random examples), but how consistently they act on their membership
belief depends on how sure they are which concept the program has. The
log-odds of "yes" is the log-odds of the membership belief times a response
gain that shrinks (or grows) with the normalised entropy of the posterior over
concepts, by an inferred degree: when many concepts stay plausible, answers
become noisier and drift toward 50/50.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
ENTROPY_CENTRE = 0.3  # typical normalised posterior entropy over concepts


def concept_sets():
    rules = [
        [2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36],
        [7, 49], [8, 64], [9, 81], [10, 100],
    ]
    rules += [[i for i in NUMBERS if i % n == 0] for n in range(3, 13)]
    rules += [[i for i in NUMBERS if i % 10 == d] for d in range(1, 10)]
    rules += [[i for i in NUMBERS if str(d) in str(i)] for d in range(1, 10)]
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
RULE_SIZE = RULE_MEMBERSHIP.sum(1)
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


def log_lik(member, size, examples):
    safe = np.maximum(size, 1.0)[..., None]
    per = np.log(member[..., examples - 1] * (1 - EPSILON) / safe + EPSILON / 100)
    return per.sum(axis=-1)


def set_quantities(examples):
    rule_ll = log_lik(RULE_MEMBERSHIP, RULE_SIZE, examples)
    int_ll = log_lik(INTERVAL_MEMBERSHIP, INTERVAL_MEMBERSHIP.sum(1), examples)
    int_ev = np.empty(N_GROUPS)
    int_mem = np.empty((N_GROUPS, 100))
    for g, idx in enumerate(GROUP_MEMBERS):
        ll = int_ll[idx]
        int_ev[g] = logsumexp(ll)
        int_mem[g] = np.exp(ll - int_ev[g]) @ INTERVAL_MEMBERSHIP[idx]
    return rule_ll, int_ev, int_mem


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    group_log_evidence = pm.Data("group_log_evidence", np.zeros((1, N_GROUPS)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_in_hypothesis = pm.Data("cell_in_hypothesis", np.zeros((1, N_RULES + N_GROUPS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    rule_share = pm.Beta("rule_share", alpha=2.0, beta=2.0)
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    rule_weights = pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS)
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    decade_boost = pm.Normal("decade_boost", mu=1.0, sigma=1.5)
    # Decision stage: response gain at typical confusion, and how strongly
    # confusion about the concept degrades it.
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    confusion_slope = pm.Normal("confusion_slope", mu=0.0, sigma=2.0)

    log_interval = np.log(GROUP_WIDTHS) - GROUP_WIDTHS / width_scale + decade_boost * GROUP_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval + np.log(GROUP_SIZES))
    log_rules = pt.log(rule_share) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_groups = pt.log1p(-rule_share) + log_interval_prior[None, :] + group_log_evidence
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))

    log_post = log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True)
    entropy = -(posterior * log_post).sum(axis=1) / np.log(N_RULES + N_GROUPS)
    gain = pt.exp(log_gain - confusion_slope * (entropy - ENTROPY_CENTRE))
    post_set = posterior[cell_set]
    belief = pt.clip((post_set * cell_in_hypothesis).sum(axis=1), 1e-6, 1 - 1e-6)
    cell_logit = gain[cell_set] * (pt.log(belief) - pt.log1p(-belief))

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(cell_logit)[row_cell], 1e-6, 1 - 1e-6))
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
    rule_ll, int_ev = [], []
    cell_rows = [None] * len(cells)
    for s in unique:
        r_ll, i_ev, i_mem = set_quantities(np.array(s))
        rule_ll.append(r_ll)
        int_ev.append(i_ev)
        for ci, t in by_set[s]:
            cell_rows[ci] = np.concatenate([RULE_MEMBERSHIP[:, t - 1], i_mem[:, t - 1]])
    return {
        "rule_log_lik": np.stack(rule_ll),
        "group_log_evidence": np.stack(int_ev),
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_in_hypothesis": np.stack(cell_rows),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r.get("rating", 0)) for r in rows], dtype="int64"),
    }
