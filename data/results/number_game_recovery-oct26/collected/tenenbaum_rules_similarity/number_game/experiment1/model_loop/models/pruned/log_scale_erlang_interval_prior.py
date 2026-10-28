"""Refinement of erlang_interval_size_prior: Bayesian concept learning over
42 rules and every interval of 1-100 (size principle, 10% of examples drawn
uniformly, "yes" = posterior probability the concept contains the target),
with an Erlang prior over interval width whose scale is inferred. The one
change: width is perceived on a logarithmic number line, w = log((n + 1) / m)
for the interval [m, n], so an interval of fixed span counts as narrower among
large numbers and people generalize further from large examples than small.
The size principle still uses the interval's actual number of members.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

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
    return rules, intervals, bounds


RULES, INTERVALS, BOUNDS = concept_sets()
N_RULES, N_INTERVALS = len(RULES), len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)

# Perceived (log-scale) width of each interval [m, n].
LOG_WIDTH = np.array([np.log((n + 1.0) / m) for m, n in BOUNDS])

# The posterior predictive is linear in the prior: per set, the evidence is
# sum_h prior_h L_h, and per (set, target) cell the "yes" mass is
# sum_h prior_h L_h [target in h]. Both weight tables are precomputed with each
# set's likelihoods scaled by its largest one.
with pm.Model() as model:
    set_rule_lik = pm.Data("set_rule_lik", np.zeros((1, N_RULES)))
    set_interval_lik = pm.Data("set_interval_lik", np.zeros((1, N_INTERVALS)))
    cell_rule_yes = pm.Data("cell_rule_yes", np.zeros((1, N_RULES)))
    cell_interval_yes = pm.Data("cell_interval_yes", np.zeros((1, N_INTERVALS)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    width_scale = pm.LogNormal("width_scale", mu=np.log(0.5), sigma=1.0)

    # Erlang over log-scale width, normalised over all 5,050 intervals.
    log_width = np.log(LOG_WIDTH) - LOG_WIDTH / width_scale
    interval_prior = (1 - p_mix) * pt.exp(log_width - pt.logsumexp(log_width))
    rule_prior = p_mix * rule_weights

    evidence = pt.dot(set_rule_lik, rule_prior) + pt.dot(set_interval_lik, interval_prior)
    yes_mass = pt.dot(cell_rule_yes, rule_prior) + pt.dot(cell_interval_yes, interval_prior)
    p_cell = yes_mass / evidence[cell_set]
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    lik = np.exp(log_lik - log_lik.max(axis=1, keepdims=True))
    cell_set = np.array([index[s] for s, _ in cells], dtype="int64")
    cell_target = np.array([t - 1 for _, t in cells])
    cell_yes = lik[cell_set] * MEMBERSHIP[:, cell_target].T
    return {
        "set_rule_lik": lik[:, :N_RULES],
        "set_interval_lik": lik[:, N_RULES:],
        "cell_rule_yes": cell_yes[:, :N_RULES],
        "cell_interval_yes": cell_yes[:, N_RULES:],
        "cell_set": cell_set,
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
