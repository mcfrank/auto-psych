"""Refinement of erlang_interval_size_prior: Bayesian concept learning over
42 rules and every interval of 1-100 (Erlang width prior with inferred scale,
size principle, 10% of examples drawn uniformly, "yes" = posterior probability
the concept contains the target). The one change: for sets of 2+ distinct
examples, people assume the examples tend to mark the ends of the program's
range, so each endpoint of an interval that coincides with the smallest or
largest example multiplies that interval's evidence by exp(edge_bonus), with
edge_bonus inferred. The interval exactly spanning the examples is favoured,
sharpening the generalization edge at the span.
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
    bounds = [(m, n) for m in range(1, 101) for n in range(m, 101)]
    intervals = [list(range(m, n + 1)) for m, n in bounds]
    return rules, intervals, np.array(bounds)


RULES, INTERVALS, INTERVAL_BOUNDS = concept_sets()
N_RULES, N_INTERVALS = len(RULES), len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]

# Intervals sharing a width and a number of example-marked ends (0, 1, 2)
# share prior and bonus, so per set they act as one hypothesis.
INTERVAL_WIDTH = INTERVAL_BOUNDS[:, 1] - INTERVAL_BOUNDS[:, 0] + 1
GROUP_WIDTHS, WIDTH_OF_INTERVAL, WIDTH_SIZES = np.unique(
    INTERVAL_WIDTH, return_inverse=True, return_counts=True
)
N_WIDTHS = len(GROUP_WIDTHS)
N_GROUPS = 3 * N_WIDTHS
GROUP_WIDTH = np.repeat(GROUP_WIDTHS, 3).astype("float64")
GROUP_EDGES = np.tile(np.arange(3), N_WIDTHS).astype("float64")
EMPTY = -1e4

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    group_log_evidence = pm.Data("group_log_evidence", np.zeros((1, N_GROUPS)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_in_hypothesis = pm.Data("cell_in_hypothesis", np.zeros((1, N_RULES + N_GROUPS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    edge_bonus = pm.HalfNormal("edge_bonus", sigma=2.0)

    # Erlang over width, per interval; normalised over all 5,050 intervals.
    log_width = np.log(GROUP_WIDTHS.astype("float64")) - GROUP_WIDTHS / width_scale
    log_interval_prior = log_width - pt.logsumexp(log_width + np.log(WIDTH_SIZES))
    log_group_prior = pt.repeat(log_interval_prior, 3) + edge_bonus * GROUP_EDGES

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_groups = pt.log1p(-p_mix) + log_group_prior[None, :] + group_log_evidence
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
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
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_log_lik = set_log_lik[:, N_RULES:]
    log_evidence = np.full((len(unique), N_GROUPS), EMPTY)
    in_concept = np.zeros((len(unique), N_GROUPS, 100))
    for i, s in enumerate(unique):
        lo, hi = min(s), max(s)
        if lo == hi:
            edges = np.zeros(N_INTERVALS, dtype="int64")
        else:
            edges = (INTERVAL_BOUNDS[:, 0] == lo).astype("int64") + (INTERVAL_BOUNDS[:, 1] == hi)
        group = WIDTH_OF_INTERVAL * 3 + edges
        for g in np.unique(group):
            members = group == g
            ll = interval_log_lik[i, members]
            log_evidence[i, g] = logsumexp(ll)
            in_concept[i, g] = np.exp(ll - log_evidence[i, g]) @ INTERVAL_MEMBERSHIP[members]
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
