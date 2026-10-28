"""Refinement of bigelow_piantadosi_interval: Bayesian concept learning over
42 rules and every interval of 1-100 (size principle, 10% of examples drawn
uniformly, "yes" = posterior probability the concept contains the target),
where the interval prior depends on the interval's width, not its midpoint:
an Erlang density over width s, p(s) ∝ s exp(-s / sigma), with sigma inferred
(Tenenbaum's number-game prior). Rules keep a p_mix share and Dirichlet weights.

Refinement: a response lapse. With probability `lapse` the answer ignores the
posterior and is "yes" with a guessing bias `guess`; otherwise it is the
posterior probability that the concept contains the target.
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
    intervals = [list(range(m, n + 1)) for m in range(1, 101) for n in range(m, 101)]
    return rules, intervals


RULES, INTERVALS = concept_sets()
N_RULES, N_INTERVALS = len(RULES), len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]

# Intervals of one width share a prior, so per set they act as one hypothesis
# (total evidence, and per target the share from intervals containing it).
INTERVAL_WIDTH = np.array([len(c) for c in INTERVALS])
GROUP_WIDTHS, GROUP_OF_INTERVAL, GROUP_SIZES = np.unique(
    INTERVAL_WIDTH, return_inverse=True, return_counts=True
)
N_GROUPS = len(GROUP_WIDTHS)

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
    lapse = pm.Beta("lapse", 1.5, 6.0)
    guess = pm.Beta("guess", 2, 2)

    # Erlang over width, per interval; normalised over all 5,050 intervals.
    log_width = np.log(GROUP_WIDTHS.astype("float64")) - GROUP_WIDTHS / width_scale
    log_interval_prior = log_width - pt.logsumexp(log_width + np.log(GROUP_SIZES))

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_groups = pt.log1p(-p_mix) + log_interval_prior[None, :] + group_log_evidence
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[cell_set] * cell_in_hypothesis).sum(axis=1)

    p_concept = (1 - lapse) * in_concept + lapse * guess
    p_yes = pm.Deterministic("p_yes", pt.clip(p_concept[row_cell], 1e-6, 1 - 1e-6))
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
