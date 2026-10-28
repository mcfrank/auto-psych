"""People store each example as a separate exemplar and judge the target by its
similarity to each one, where similarity to one exemplar is Shepard's
consequential-region generalization (how likely a familiar number concept — an
arithmetic rule, or a range with a preference for moderate, decade-bounded
widths — that contains the exemplar also contains the target). The per-exemplar
similarities are combined conjunctively as a soft minimum (a generalized mean
with negative inferred exponent), so the target must resemble every example,
but the examples are never pooled as joint evidence under the size principle.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_EX = 4


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
INTERVAL_WIDTH = np.array([n - m + 1 for m, n in BOUNDS])
DECADE_BOUNDED = np.array([(m - 1) % 10 == 0 and n % 10 == 0 for m, n in BOUNDS], dtype="int64")
GROUP_KEYS, GROUP_OF_INTERVAL, GROUP_SIZES = np.unique(
    np.stack([INTERVAL_WIDTH, DECADE_BOUNDED], axis=1), axis=0, return_inverse=True, return_counts=True
)
GROUP_OF_INTERVAL = GROUP_OF_INTERVAL.ravel()
GROUP_WIDTHS = GROUP_KEYS[:, 0].astype("float64")
GROUP_ALIGNED = GROUP_KEYS[:, 1].astype("float64")
N_GROUPS = len(GROUP_KEYS)


def single_example_lik(member):
    """Likelihood of one exemplar x under each concept (size principle, 10% random)."""
    size = np.maximum(member.sum(axis=1), 1.0)[:, None]
    return member * (1 - EPSILON) / size + EPSILON / 100


RULE_LIK = single_example_lik(RULE_MEMBERSHIP)  # (rules, x)
RULE_LIK_MEM = RULE_LIK[:, :, None] * RULE_MEMBERSHIP[:, None, :]  # (rules, x, t)
_INT_LIK = single_example_lik(INTERVAL_MEMBERSHIP)
GROUP_EVIDENCE = np.zeros((N_GROUPS, 100))
GROUP_LIK_MEM = np.zeros((N_GROUPS, 100, 100))
for _g in range(N_GROUPS):
    _idx = np.flatnonzero(GROUP_OF_INTERVAL == _g)
    GROUP_EVIDENCE[_g] = _INT_LIK[_idx].sum(axis=0)
    GROUP_LIK_MEM[_g] = _INT_LIK[_idx].T @ INTERVAL_MEMBERSHIP[_idx]
RULE_LIK_MEM = RULE_LIK_MEM.reshape(N_RULES, -1)
GROUP_LIK_MEM = GROUP_LIK_MEM.reshape(N_GROUPS, -1)


with pm.Model() as model:
    cell_examples = pm.Data("cell_examples", np.zeros((1, MAX_EX), dtype="int64"))
    cell_mask = pm.Data("cell_mask", np.ones((1, MAX_EX)))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    rule_share = pm.Beta("rule_share", alpha=2.0, beta=2.0)
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    rule_prior = rule_share * pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS)
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    decade_boost = pm.Normal("decade_boost", mu=1.0, sigma=1.5)
    log_interval = np.log(GROUP_WIDTHS) - GROUP_WIDTHS / width_scale + decade_boost * GROUP_ALIGNED
    interval_prior = (1 - rule_share) * pt.exp(log_interval - pt.logsumexp(log_interval + np.log(GROUP_SIZES)))

    # Shepard generalization from each single exemplar x to each target t.
    numer = pt.dot(rule_prior, RULE_LIK_MEM) + pt.dot(interval_prior, GROUP_LIK_MEM)
    denom = pt.dot(rule_prior, RULE_LIK) + pt.dot(interval_prior, GROUP_EVIDENCE)
    sim = pt.clip(numer.reshape((100, 100)) / denom[:, None], 1e-9, 1.0)
    log_sim = pt.log(sim)

    # Soft-minimum combination: generalized mean with exponent -strictness.
    strictness = pm.LogNormal("strictness", mu=0.0, sigma=1.0)
    ls = log_sim[cell_examples, cell_target[:, None]]  # (cells, MAX_EX)
    n_ex = cell_mask.sum(axis=1)
    z = -strictness * ls + pt.log(cell_mask + 1e-30)
    log_mean = -(pt.logsumexp(z, axis=1) - pt.log(n_ex)) / strictness
    set_sim = pt.exp(log_mean)

    # Similarity mapped to a yes-rate between a floor and a ceiling.
    floor = pm.Beta("floor", alpha=1.0, beta=9.0)
    ceiling = pm.Beta("ceiling", alpha=9.0, beta=1.0)
    p_cell = floor + (ceiling - floor) * set_sim
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    ex = np.zeros((len(cells), MAX_EX), dtype="int64")
    mask = np.zeros((len(cells), MAX_EX))
    for i, (s, _) in enumerate(cells):
        ex[i, : len(s)] = np.array(s) - 1
        mask[i, : len(s)] = 1.0
    return {
        "cell_examples": ex,
        "cell_mask": mask,
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
