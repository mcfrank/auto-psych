"""Exemplar-memory account: each example is stored as a separate exemplar and
the target is judged by its average similarity to the stored exemplars. The
similarity of target y to exemplar x is Shepard's consequential-region
generalization, P(y in C | x alone), over familiar concepts C (42 arithmetic
rules, and every interval of 1-100 with an Erlang prior over width; size
principle, 10% random examples). Similarities are averaged across exemplars
(GCM-style summed similarity), not combined as joint evidence about one concept.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

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
    rules.append([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    rules.append([i for i in NUMBERS if i % 2 == 0])
    rules.append([i for i in NUMBERS if i % 2 == 1])
    rules.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    rules.append([1, 8, 27, 64])
    intervals = [list(range(m, n + 1)) for m in range(1, 101) for n in range(m, 101)]
    return rules, intervals


RULES, INTERVALS = concept_sets()
N_RULES = len(RULES)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_WIDTH = np.array([len(c) for c in INTERVALS])
GROUP_WIDTHS, GROUP_OF_INTERVAL, GROUP_SIZES = np.unique(
    INTERVAL_WIDTH, return_inverse=True, return_counts=True
)
N_GROUPS = len(GROUP_WIDTHS)
N_HYP = N_RULES + N_GROUPS

# Single-exemplar quantities for every number 1-100 (index = number - 1):
# rule log-likelihoods, per-width-group interval evidence, and per group the
# share of that group's evidence-weighted intervals containing each target.
EX_RULE_LOG_LIK = EXAMPLE_LOG_LIK[:N_RULES].T  # (100, N_RULES)
_int_ll = EXAMPLE_LOG_LIK[N_RULES:].T  # (100, N_INTERVALS)
EX_GROUP_LOG_EV = np.empty((100, N_GROUPS))
EX_GROUP_IN = np.empty((100, N_GROUPS, 100))
for _g in range(N_GROUPS):
    _m = GROUP_OF_INTERVAL == _g
    EX_GROUP_LOG_EV[:, _g] = logsumexp(_int_ll[:, _m], axis=1)
    _w = np.exp(_int_ll[:, _m] - EX_GROUP_LOG_EV[:, _g : _g + 1])
    EX_GROUP_IN[:, _g] = _w @ MEMBERSHIP[N_RULES:][_m]


with pm.Model() as model:
    ex_rule_log_lik = pm.Data("ex_rule_log_lik", np.zeros((1, N_RULES)))
    ex_group_log_ev = pm.Data("ex_group_log_ev", np.zeros((1, N_GROUPS)))
    slot_exemplar = pm.Data("slot_exemplar", np.zeros((1, MAX_EX), dtype="int64"))
    slot_mask = pm.Data("slot_mask", np.ones((1, MAX_EX)))
    slot_in_hypothesis = pm.Data("slot_in_hypothesis", np.zeros((1, MAX_EX, N_HYP)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)

    log_width = np.log(GROUP_WIDTHS.astype("float64")) - GROUP_WIDTHS / width_scale
    log_interval_prior = log_width - pt.logsumexp(log_width + np.log(GROUP_SIZES))

    # Consequential-region posterior given each single exemplar.
    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + ex_rule_log_lik
    log_groups = pt.log1p(-p_mix) + log_interval_prior[None, :] + ex_group_log_ev
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))

    # Shepard similarity of target to each stored exemplar, then averaged.
    similarity = (posterior[slot_exemplar] * slot_in_hypothesis).sum(axis=2)
    mean_sim = (similarity * slot_mask).sum(axis=1) / slot_mask.sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(mean_sim[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    exemplars = sorted({x for s, _ in cells for x in s})
    ex_index = {x: i for i, x in enumerate(exemplars)}
    n = len(cells)
    slot_ex = np.zeros((n, MAX_EX), dtype="int64")
    mask = np.zeros((n, MAX_EX))
    in_hyp = np.zeros((n, MAX_EX, N_HYP))
    for c, (s, t) in enumerate(cells):
        for k, x in enumerate(s[:MAX_EX]):
            slot_ex[c, k] = ex_index[x]
            mask[c, k] = 1.0
            in_hyp[c, k, :N_RULES] = RULE_MEMBERSHIP[:, t - 1]
            in_hyp[c, k, N_RULES:] = EX_GROUP_IN[x - 1, :, t - 1]
    ex_rows = np.array(exemplars) - 1
    return {
        "ex_rule_log_lik": EX_RULE_LOG_LIK[ex_rows],
        "ex_group_log_ev": EX_GROUP_LOG_EV[ex_rows],
        "slot_exemplar": slot_ex,
        "slot_mask": mask,
        "slot_in_hypothesis": in_hyp,
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
