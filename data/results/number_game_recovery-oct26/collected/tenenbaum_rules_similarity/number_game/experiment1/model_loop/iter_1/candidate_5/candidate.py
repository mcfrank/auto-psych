"""Refinement of bigelow_piantadosi_interval: Bayesian concept learning over
42 rules and every interval of 1-100 (size principle, 10% of examples drawn
uniformly, "yes" = posterior probability the concept contains the target),
where the interval prior keeps the Beta density over the interval's midpoint
and adds the one change: it also depends on the interval's width, via an
Erlang density over width s, p(s) ∝ s exp(-s / sigma), with sigma inferred
(component taken from erlang_interval_size_prior). An interval's prior is the
product of the two, normalised over all 5,050 intervals: people's prior over
ranges depends both on where a range lies and how wide it is.
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
    intervals = [list(range(m, n + 1)) for m in range(1, 101) for n in range(m, 101)]
    return rules, intervals


RULES, INTERVALS = concept_sets()
N_RULES, N_INTERVALS = len(RULES), len(INTERVALS)
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)

# Midpoint grid point g = m + n - 2 of 199 (as in bigelow_piantadosi_interval)
# and width n - m + 1 of every interval.
MIDPOINT_GRID = np.clip(np.arange(199) / 198, 1e-5, 1 - 1e-5)
INTERVAL_GRID_POINT = np.array([c[0] + c[-1] - 2 for c in INTERVALS])
INTERVAL_WIDTH = np.array([len(c) for c in INTERVALS], dtype="float64")

with pm.Model() as model:
    # P(D | h) for each set, relative to the set's best hypothesis (so in
    # (0, 1]); it does not depend on any parameter, so it is computed once.
    # The cell_ arrays are the same evidence for each (set, target) cell,
    # zeroed for hypotheses that do not contain the target.
    rule_evidence = pm.Data("rule_evidence", np.ones((1, N_RULES)))  # (sets, rules)
    interval_evidence = pm.Data("interval_evidence", np.ones((1, N_INTERVALS)))  # (sets, intervals)
    cell_rule_evidence = pm.Data("cell_rule_evidence", np.ones((1, N_RULES)))  # (cells, rules)
    cell_interval_evidence = pm.Data("cell_interval_evidence", np.ones((1, N_INTERVALS)))  # (cells, intervals)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)

    # Interval prior: Beta density at the midpoint times Erlang over width.
    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    width = pt.as_tensor(INTERVAL_WIDTH)
    log_prior = log_density[pt.as_tensor(INTERVAL_GRID_POINT)] + pt.log(width) - width / width_scale
    interval_prior = pt.exp(log_prior - pt.logsumexp(log_prior))

    # Posterior = prior x evidence, normalised per set; "yes" = its share on
    # concepts containing the target.
    total = p_mix * pt.dot(rule_evidence, rule_weights) + (1 - p_mix) * pt.dot(interval_evidence, interval_prior)
    in_target = p_mix * pt.dot(cell_rule_evidence, rule_weights) + (1 - p_mix) * pt.dot(
        cell_interval_evidence, interval_prior
    )
    in_concept = in_target / total[cell_set]

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
    evidence = np.exp(set_log_lik - set_log_lik.max(axis=1, keepdims=True))
    cell_set = np.array([index[s] for s, _ in cells], dtype="int64")
    cell_target = np.array([t - 1 for _, t in cells], dtype="int64")
    cell_evidence = evidence[cell_set] * MEMBERSHIP[:, cell_target].T
    return {
        "rule_evidence": evidence[:, :N_RULES],
        "interval_evidence": evidence[:, N_RULES:],
        "cell_rule_evidence": cell_evidence[:, :N_RULES],
        "cell_interval_evidence": cell_evidence[:, N_RULES:],
        "cell_set": cell_set,
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
