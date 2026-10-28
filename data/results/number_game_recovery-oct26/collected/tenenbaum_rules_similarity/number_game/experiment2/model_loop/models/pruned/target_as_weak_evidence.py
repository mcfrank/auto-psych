"""Target as weak evidence: Bayesian concept learning as in
family_pooled_rule_prior (42 rules with a kind-level prior, every interval of
1-100 with an Erlang width prior and a decade-boundary bonus, size principle
with 10% uniform examples), with one distortion — the question itself is taken
as evidence. People partly treat the queried target as if the program had
produced it: the target enters the likelihood as an extra example raised to an
inferred weight w in (0, 1) before "yes" = posterior probability that the
concept contains the target. With w = 0 this is the ideal learner.
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
INTERVALS = [list(range(m, n + 1)) for m, n in BOUNDS]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
RULE_SIZES = RULE_MEMBERSHIP.sum(axis=1)

INTERVAL_WIDTH = np.array([n - m + 1 for m, n in BOUNDS])
DECADE_BOUNDED = np.array([(m - 1) % 10 == 0 and n % 10 == 0 for m, n in BOUNDS], dtype="int64")
GROUP_KEYS, GROUP_OF_INTERVAL, GROUP_SIZES = np.unique(
    np.stack([INTERVAL_WIDTH, DECADE_BOUNDED], axis=1), axis=0, return_inverse=True, return_counts=True
)
GROUP_OF_INTERVAL = GROUP_OF_INTERVAL.ravel()
GROUP_WIDTHS = GROUP_KEYS[:, 0].astype("float64")
GROUP_ALIGNED = GROUP_KEYS[:, 1].astype("float64")
N_GROUPS = len(GROUP_KEYS)

# Log ratio of the target's likelihood when a hypothesis contains it versus
# when it does not ((1 - eps) / |h| + eps / 100 against eps / 100); every
# interval in a group has the same width, so one value per group.
HYP_SIZES = np.concatenate([RULE_SIZES, GROUP_WIDTHS])
LOG_TARGET_RATIO = np.log(((1 - EPSILON) / HYP_SIZES + EPSILON / 100) / (EPSILON / 100))

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    group_log_evidence = pm.Data("group_log_evidence", np.zeros((1, N_GROUPS)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_in_hypothesis = pm.Data("cell_in_hypothesis", np.zeros((1, N_RULES + N_GROUPS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    rule_weights = pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS)
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    decade_boost = pm.Normal("decade_boost", mu=1.0, sigma=1.5)
    # How much of an example's evidential weight the queried target carries.
    target_weight = pm.Beta("target_weight", 1.0, 2.0)

    log_interval = np.log(GROUP_WIDTHS) - GROUP_WIDTHS / width_scale + decade_boost * GROUP_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval + np.log(GROUP_SIZES))

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_groups = pt.log1p(-p_mix) + log_interval_prior[None, :] + group_log_evidence
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))

    # Tempered target likelihood (relative to a non-member's): members of a
    # hypothesis get (ratio)^w, non-members 1.
    boost = pt.exp(target_weight * LOG_TARGET_RATIO)[None, :]
    post_cell = posterior[cell_set]
    member = cell_in_hypothesis
    yes_mass = (post_cell * member * boost).sum(axis=1)
    total_mass = yes_mass + (post_cell * (1.0 - member)).sum(axis=1)
    in_concept = yes_mass / total_mass

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_log_lik = set_log_lik[:, N_RULES:]
    log_evidence = np.empty((len(unique), N_GROUPS))
    in_concept = np.empty((len(unique), N_GROUPS, 100))
    for g in range(N_GROUPS):
        members = GROUP_OF_INTERVAL == g
        log_evidence[:, g] = logsumexp(interval_log_lik[:, members], axis=1)
        weights = np.exp(interval_log_lik[:, members] - log_evidence[:, g : g + 1])
        in_concept[:, g] = weights @ MEMBERSHIP[N_RULES:][members]
    cell_set = np.array([index[s] for s, _ in cells], dtype="int64")
    cell_target = np.array([t - 1 for _, t in cells])
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "group_log_evidence": log_evidence,
        "cell_set": cell_set,
        "cell_in_hypothesis": np.concatenate(
            [RULE_MEMBERSHIP[:, cell_target].T, in_concept[cell_set, :, cell_target]], axis=1
        ),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
