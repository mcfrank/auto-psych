"""Refinement of bigelow_piantadosi_independent: Bayesian concept learning over
42 rules (with a prior weight inferred per rule, p_mix for the rules as a group)
and every interval of 1-100, where the prior over intervals is proportional to
(width) ** gamma with gamma inferred, instead of flat. Wide-interval priors
(gamma > 0) offset the size principle and yield broader generalization beyond
the examples' range; gamma = 0 recovers the incumbent. Examples are sampled
from the concept (size principle), 10% uniformly from 1-100, and "yes" is the
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
N_RULES = len(RULES)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_WIDTH = np.array([len(c) for c in INTERVALS])  # 1..100
WIDTHS = np.arange(1, 101)
LOG_WIDTHS = np.log(WIDTHS.astype("float64"))
LOG_COUNT_OF_WIDTH = np.log(101.0 - WIDTHS)  # intervals of width L: 101 - L

# Intervals of the same width share a prior, so per example set they collapse
# into 100 width-groups: summed evidence per width, and per (trial, width) the
# share of that evidence from intervals containing the target.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    width_log_evidence = pm.Data("width_log_evidence", np.zeros((1, 100)))  # (sets, widths)
    width_in_concept = pm.Data("width_in_concept", np.zeros((1, 100)))  # (trials, widths)
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    gamma = pm.Normal("gamma", mu=0.0, sigma=1.5)

    # Per-interval log prior of an interval of width L, normalised over all intervals.
    log_width_prior = gamma * LOG_WIDTHS - pt.logsumexp(LOG_COUNT_OF_WIDTH + gamma * LOG_WIDTHS)
    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_widths = pt.log1p(-p_mix) + log_width_prior[None, :] + width_log_evidence
    log_joint = pt.concatenate([log_rules, log_widths], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, rules + 100)
    rule_part = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)).reshape((-1,))[row_cell]
    interval_part = (posterior[:, N_RULES:][row_set] * width_in_concept).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(rule_part + interval_part, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_log_lik = set_log_lik[:, N_RULES:]  # (sets, intervals)
    width_ev = np.full((len(unique), 100), -np.inf)
    share = np.zeros((len(unique), 100, 100))  # (sets, widths, targets)
    for w in WIDTHS:
        cols = np.where(INTERVAL_WIDTH == w)[0]
        ll = interval_log_lik[:, cols]
        ev = logsumexp(ll, axis=1)
        width_ev[:, w - 1] = ev
        share[:, w - 1, :] = np.exp(ll - ev[:, None]) @ MEMBERSHIP[N_RULES + cols]
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "width_log_evidence": width_ev,
        "width_in_concept": share[set_idx, :, tgt],
        "row_set": set_idx,
        "row_cell": set_idx * 100 + tgt,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
