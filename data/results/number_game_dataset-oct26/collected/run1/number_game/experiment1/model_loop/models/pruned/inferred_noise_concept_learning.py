"""Refinement of bigelow_piantadosi_independent: Bayesian concept learning over
42 rules (each with its own prior weight, sharing p_mix) and all 5,050
intervals of 1..100, with examples drawn from the concept by the size
principle -- but the probability that an example is random noise (drawn
uniformly from 1..100 rather than from the concept) is a free parameter,
epsilon, instead of being fixed at 0.1. A learner who believes examples are
rarely noisy will not explain away an example that breaks a broad rule, so
rich, rule-less sets are attributed to the interval spanning them and numbers
outside their range are rejected. "Yes" is the posterior probability that the
concept contains the target.

Rule likelihoods are exact in epsilon (they depend only on how many examples
each rule contains). The interval part -- all 5,050 intervals share one prior,
so they act as one hypothesis with total evidence and per-target in-concept
share -- is precomputed on a grid of logit(epsilon) and linearly interpolated.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

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
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
INT_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INT_SIZE = INT_MEMBERSHIP.sum(axis=1, keepdims=True)

# Grid over logit(epsilon), epsilon from ~0.002 to ~0.6.
LOGIT_GRID = np.linspace(-6.0, 0.5, 27)
EPS_GRID = 1.0 / (1.0 + np.exp(-LOGIT_GRID))
G = len(LOGIT_GRID)
STEP = LOGIT_GRID[1] - LOGIT_GRID[0]

with pm.Model() as model:
    rule_hits = pm.Data("rule_hits", np.zeros((1, N_RULES)))  # (sets, rules): examples in rule
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    int_log_ev = pm.Data("int_log_ev", np.zeros((1, G)))  # (sets, grid)
    int_in_concept = pm.Data("int_in_concept", np.zeros((1, G, 100)))  # (sets, grid, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    logit_eps = pm.Normal("logit_eps", mu=-2.2, sigma=1.0)
    eps = pm.Deterministic("epsilon", pm.math.sigmoid(logit_eps))

    # Exact rule likelihood for this epsilon.
    log_in = pt.log((1 - eps) / pt.as_tensor(RULE_SIZE) + eps / 100.0)[None, :]
    log_out = pt.log(eps / 100.0)
    rule_log_lik = rule_hits * log_in + (n_examples[:, None] - rule_hits) * log_out

    # Interpolated interval evidence and in-concept share.
    pos = pt.clip((logit_eps - LOGIT_GRID[0]) / STEP, 0.0, G - 1.000001)
    lo = pt.cast(pt.floor(pos), "int64")
    w = pos - lo
    interval_log_evidence = (1 - w) * int_log_ev[:, lo] + w * int_log_ev[:, lo + 1]
    interval_in_concept = (1 - w) * int_in_concept[:, lo, :] + w * int_in_concept[:, lo + 1, :]

    log_rules = pt.log(p_mix) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_intervals = pt.log1p(-p_mix) - np.log(N_INTERVALS) + interval_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + posterior[:, N_RULES:] * interval_in_concept

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    counts = np.zeros((len(unique), 100))
    for i, s in enumerate(unique):
        for x in s:
            counts[i, x - 1] += 1
    log_ev = np.zeros((len(unique), G))
    in_c = np.zeros((len(unique), G, 100))
    for g, e in enumerate(EPS_GRID):
        ex_ll = np.log(INT_MEMBERSHIP * (1 - e) / INT_SIZE + e / 100.0)  # (intervals, 100)
        set_ll = counts @ ex_ll.T  # (sets, intervals)
        ev = logsumexp(set_ll, axis=1)
        log_ev[:, g] = ev
        in_c[:, g, :] = np.exp(set_ll - ev[:, None]) @ INT_MEMBERSHIP
    return {
        "rule_hits": counts @ RULE_MEMBERSHIP.T,
        "n_examples": counts.sum(axis=1),
        "int_log_ev": log_ev,
        "int_in_concept": in_c,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
