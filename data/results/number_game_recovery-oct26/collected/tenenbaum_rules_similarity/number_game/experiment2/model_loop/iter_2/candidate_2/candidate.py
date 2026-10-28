"""Two-cluster range concepts. People do Bayesian concept learning over 42
arithmetic rules (primes excluding 1, kind-level prior) and every interval of
1-100 (Erlang width prior, decade-boundary bonus, an evidence bonus for a range
whose top is the largest example), size principle with 10% uniform examples,
"yes" = posterior probability the concept contains the target. The one new
claim: people can also conceive of the program's numbers as two separate
clusters -- the union of two equally wide, non-adjacent stretches of 1-100 --
a kind of concept with its own inferred prior share and an Erlang prior over
cluster width sharing the ranges' width scale. Widely separated examples
(10,80) are then explained by two clusters, so "yes" falls near each example
and "no" in the empty gap between them, where the best model (one wide range
or a rule) says "yes". Range-restricted rules are omitted (they carried ~1% of
the prior and slowed sampling ~50x).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_CLUSTER_WIDTH = 25

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

NEG = -1e4

# Two-cluster concepts [a, a+w-1] U [b, b+w-1] with b >= a+w+1, by width w.
CLUSTER_WIDTHS = np.arange(1, MAX_CLUSTER_WIDTH + 1).astype("float64")
CLUSTER_MEMBERSHIP = []
for w in range(1, MAX_CLUSTER_WIDTH + 1):
    pairs = [(a, b) for a in range(1, 102 - w) for b in range(a + w + 1, 102 - w)]
    lo = np.array(pairs)
    mem = ((NUMBERS >= lo[:, :1]) & (NUMBERS <= lo[:, :1] + w - 1)) | (
        (NUMBERS >= lo[:, 1:]) & (NUMBERS <= lo[:, 1:] + w - 1)
    )
    CLUSTER_MEMBERSHIP.append(mem)
CLUSTER_COUNTS = np.array([len(m) for m in CLUSTER_MEMBERSHIP], dtype="float64")
N_CW = MAX_CLUSTER_WIDTH


def log_lik(member, size, examples):
    """Size-principle log-likelihood of the examples; member: (..., 100), size: (...)."""
    safe = np.maximum(size, 1.0)[..., None]
    per = np.log(member[..., examples - 1] * (1 - EPSILON) / safe + EPSILON / 100)
    return per.sum(axis=-1)


def set_quantities(examples):
    """Per-set evidence of rules, interval groups (split by whether the top
    is the largest example) and cluster-width groups, with within-group
    predictive membership over 1-100."""
    rule_ll = log_lik(RULE_MEMBERSHIP, RULE_MEMBERSHIP.sum(1), examples)
    int_ll = log_lik(INTERVAL_MEMBERSHIP, INTERVAL_MEMBERSHIP.sum(1), examples)
    at_top = np.array([n == examples.max() for _, n in BOUNDS])
    int_ev = np.full((2, N_GROUPS), NEG)
    int_mem = np.zeros((2, N_GROUPS, 100))
    for g, idx in enumerate(GROUP_MEMBERS):
        for k, part in enumerate([idx[~at_top[idx]], idx[at_top[idx]]]):
            if len(part) == 0:
                continue
            ll = int_ll[part]
            int_ev[k, g] = logsumexp(ll)
            int_mem[k, g] = np.exp(ll - int_ev[k, g]) @ INTERVAL_MEMBERSHIP[part]
    cl_ev = np.empty(N_CW)
    cl_mem = np.empty((N_CW, 100))
    for i, mem in enumerate(CLUSTER_MEMBERSHIP):
        w = i + 1
        hit = mem[:, examples - 1]
        ll = np.log(hit * (1 - EPSILON) / (2 * w) + EPSILON / 100).sum(axis=1)
        cl_ev[i] = logsumexp(ll) - np.log(len(mem))
        cl_mem[i] = np.exp(ll - logsumexp(ll)) @ mem
    return rule_ll, int_ev, int_mem, cl_ev, cl_mem


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    group_log_evidence = pm.Data("group_log_evidence", np.zeros((1, 2, N_GROUPS)))
    cluster_log_evidence = pm.Data("cluster_log_evidence", np.zeros((1, N_CW)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_in_hypothesis = pm.Data("cell_in_hypothesis", np.zeros((1, N_RULES + 2 * N_GROUPS + N_CW)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior shares of rules, single ranges and two-cluster concepts.
    shares = pm.Dirichlet("shares", a=np.array([2.0, 2.0, 1.0]))
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    rule_weights = pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS)
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    decade_boost = pm.Normal("decade_boost", mu=1.0, sigma=1.5)
    top_bonus = pm.Normal("top_bonus", mu=0.0, sigma=1.5)

    log_interval = np.log(GROUP_WIDTHS) - GROUP_WIDTHS / width_scale + decade_boost * GROUP_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval + np.log(GROUP_SIZES))
    # Erlang prior over cluster width (total covered = 2w), uniform within width.
    log_cw = np.log(2 * CLUSTER_WIDTHS) - 2 * CLUSTER_WIDTHS / width_scale
    log_cw_prior = log_cw - pt.logsumexp(log_cw)

    log_rules = pt.log(shares[0]) + pt.log(rule_weights)[None, :] + rule_log_lik
    top_shift = pt.stack([pt.zeros(()), top_bonus])[None, :, None]
    log_groups = (
        pt.log(shares[1]) + log_interval_prior[None, None, :] + group_log_evidence + top_shift
    ).reshape((rule_log_lik.shape[0], 2 * N_GROUPS))
    log_clusters = pt.log(shares[2]) + log_cw_prior[None, :] + cluster_log_evidence
    log_joint = pt.concatenate([log_rules, log_groups, log_clusters], axis=1)
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
    rule_ll, int_ev, cl_ev = [], [], []
    cell_rows = [None] * len(cells)
    for s in unique:
        r_ll, i_ev, i_mem, c_ev, c_mem = set_quantities(np.array(s))
        rule_ll.append(r_ll)
        int_ev.append(i_ev)
        cl_ev.append(c_ev)
        for ci, t in by_set[s]:
            cell_rows[ci] = np.concatenate([RULE_MEMBERSHIP[:, t - 1], i_mem[:, :, t - 1].reshape(-1), c_mem[:, t - 1]])
    return {
        "rule_log_lik": np.stack(rule_ll),
        "group_log_evidence": np.stack(int_ev),
        "cluster_log_evidence": np.stack(cl_ev),
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_in_hypothesis": np.stack(cell_rows),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
