"""People do Bayesian concept learning over 42 arithmetic rules (primes
excluding 1, kind-level rule prior) and every range of 1-100 (Erlang width
prior with a decade-boundary preference, 10% random examples, "yes" = posterior
probability the concept contains the target), but in the size principle they
judge a range's size on a compressed, logarithmic mental number line: the
perceived size of [m, n] is its share of log(x + c) between 1 and 100, where
the compression offset c is inferred (c -> infinity gives the true width; small
c makes ranges among large numbers feel smaller). Ranges among large numbers
are therefore favoured more strongly by the size principle, so generalization
is tighter around high examples than around low ones. The prior is unchanged.

The compression is inferred as a smooth parameter on the log-offset scale; set
evidence and membership are precomputed on a fine grid of log offsets and
interpolated with a C1-smooth cubic (Keys) kernel, so the likelihood has no
kinks in the compression parameter.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
# Grid of log compression offsets, log c from -0.5 (c ~ 0.6, strong
# compression) to 7.5 (c ~ 1800, close to true width); the end points pad the
# cubic kernel so the parameter stays within [0, 7].
LOG_OFFSETS = np.arange(-0.5, 7.51, 0.5)
OFFSETS = np.exp(LOG_OFFSETS)
N_GRID = len(OFFSETS)


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
INTERVAL_MEMBERSHIP = np.array([(NUMBERS >= m) & (NUMBERS <= n) for m, n in BOUNDS], dtype="float64")
INTERVAL_LO = np.array([m for m, _ in BOUNDS], dtype="float64")
INTERVAL_HI = np.array([n for _, n in BOUNDS], dtype="float64")
INTERVAL_WIDTH = INTERVAL_HI - INTERVAL_LO + 1
DECADE_BOUNDED = np.array([(m - 1) % 10 == 0 and n % 10 == 0 for m, n in BOUNDS], dtype="int64")
GROUP_KEYS, GROUP_OF_INTERVAL, GROUP_SIZES = np.unique(
    np.stack([INTERVAL_WIDTH.astype("int64"), DECADE_BOUNDED], axis=1), axis=0, return_inverse=True, return_counts=True
)
GROUP_OF_INTERVAL = GROUP_OF_INTERVAL.ravel()
GROUP_WIDTHS = GROUP_KEYS[:, 0].astype("float64")
GROUP_ALIGNED = GROUP_KEYS[:, 1].astype("float64")
N_GROUPS = len(GROUP_KEYS)
GROUP_MEMBERS = [np.flatnonzero(GROUP_OF_INTERVAL == g) for g in range(N_GROUPS)]


def perceived_size(c):
    """Range size on a log(x + c) number line, scaled so 1-100 has size 100."""
    span = np.log(INTERVAL_HI + c) - np.log(INTERVAL_LO - 1 + c)
    return 100.0 * span / (np.log(100 + c) - np.log(c))


PERCEIVED_SIZES = [perceived_size(c) for c in OFFSETS]


def log_lik(member, size, examples):
    """Size-principle log-likelihood of the examples under each concept."""
    return np.log(member[:, examples - 1] * (1 - EPSILON) / size[:, None] + EPSILON / 100).sum(axis=1)


def set_quantities(examples):
    """Rule log-likelihoods, and for each compression offset and interval group
    (same width and decade alignment) the log evidence and evidence-weighted
    membership over 1-100."""
    rule_ll = log_lik(RULE_MEMBERSHIP, RULE_MEMBERSHIP.sum(axis=1), examples)
    int_ev = np.empty((N_GRID, N_GROUPS))
    int_mem = np.empty((N_GRID, N_GROUPS, 100))
    for k, size in enumerate(PERCEIVED_SIZES):
        int_ll = log_lik(INTERVAL_MEMBERSHIP, size, examples)
        for g, idx in enumerate(GROUP_MEMBERS):
            ll = int_ll[idx]
            int_ev[k, g] = logsumexp(ll)
            int_mem[k, g] = np.exp(ll - int_ev[k, g]) @ INTERVAL_MEMBERSHIP[idx]
    return rule_ll, int_ev, int_mem


SAMPLER_SETTINGS = {"target_accept": 0.9}

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    group_log_evidence = pm.Data("group_log_evidence", np.zeros((1, N_GRID, N_GROUPS)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_group_member = pm.Data("cell_group_member", np.zeros((1, N_GRID, N_GROUPS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    rule_share = pm.Beta("rule_share", alpha=2.0, beta=2.0)
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    rule_weights = pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS)
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    decade_boost = pm.Normal("decade_boost", mu=1.0, sigma=1.5)
    # The distortion: log of the offset c of the log(x + c) number line used to
    # size ranges, smoothly confined to [0, 7] (small = strong compression).
    compression_raw = pm.Normal("compression_raw", mu=0.0, sigma=1.5)
    log_offset = pm.Deterministic("log_offset", 7.0 * pm.math.sigmoid(compression_raw))

    pos = (log_offset - LOG_OFFSETS[0]) / 0.5
    d = pt.abs(pos - np.arange(N_GRID, dtype="float64"))
    grid_weights = pt.switch(
        d <= 1.0, 1.5 * d**3 - 2.5 * d**2 + 1.0,
        pt.switch(d < 2.0, -0.5 * d**3 + 2.5 * d**2 - 4.0 * d + 2.0, 0.0),
    )
    group_ev = pt.tensordot(group_log_evidence, grid_weights, axes=[[1], [0]])
    group_mem = pt.tensordot(cell_group_member, grid_weights, axes=[[1], [0]])

    log_interval = np.log(GROUP_WIDTHS) - GROUP_WIDTHS / width_scale + decade_boost * GROUP_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval + np.log(GROUP_SIZES))
    log_rules = pt.log(rule_share) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_groups = pt.log(1 - rule_share) + log_interval_prior[None, :] + group_ev
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_in = pt.concatenate([cell_rule_member, group_mem], axis=1)
    belief = (posterior[cell_set] * cell_in).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(belief[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    per_set = {s: set_quantities(np.array(s)) for s in unique}
    return {
        "rule_log_lik": np.stack([per_set[s][0] for s in unique]),
        "group_log_evidence": np.stack([per_set[s][1] for s in unique]),
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_group_member": np.stack([per_set[s][2][:, :, t - 1] for s, t in cells]),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
