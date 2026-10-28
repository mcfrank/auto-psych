"""Graded similarity neighbourhoods instead of crisp numeric ranges.

People do not represent magnitude concepts as crisp ranges with hard edges; a
numeric range is a similarity neighbourhood on the mental number line: a number
belongs fully if it lies inside and partially otherwise, with membership
exp(-d / width), d its distance to the range's nearest edge and width an
inferred similarity gradient. The graded membership sets both how well a range
explains each example (size principle over the summed membership) and how far
people generalize from it. Arithmetic rules keep exact membership. Everything
else (42 weighted rules with a size-dependent prior, evidence tempered
separately for rules and ranges, 10% noisy examples, first-example evidence
multiplier, decision gain and criterion) is as in
rule_size_prior_anchored_evidence; width -> 0 recovers it.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)


def rule_sets():
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
    return rules


RULES = rule_sets()
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_EXAMPLE_LOG_LIK = np.log(
    RULE_MEMBERSHIP * (1 - EPSILON) / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100
)
RULE_LOG_SIZE = np.log(RULE_MEMBERSHIP.sum(axis=1))
RULE_LOG_SIZE_C = RULE_LOG_SIZE - RULE_LOG_SIZE.mean()

LO, HI = np.array([(m, n) for m in range(1, 101) for n in range(m, 101)]).T
# Distance from each number to each interval (0 inside): (100, intervals).
DIST_T = np.ascontiguousarray(
    (np.maximum(LO[None, :] - NUMBERS[:, None], 0) + np.maximum(NUMBERS[:, None] - HI[None, :], 0)).astype("float64")
)
N_INTERVALS = len(LO)
LOG_N_INTERVALS = np.log(N_INTERVALS)

with pm.Model() as model:
    first_onehot = pm.Data("first_onehot", np.zeros((1, 100)))  # (sets, numbers): first example
    later_counts = pm.Data("later_counts", np.zeros((1, 100)))  # (sets, numbers): later examples
    cell_flat = pm.Data("cell_flat", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # trial -> cell
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 4, 4)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    rule_evidence_weight = pm.LogNormal("rule_evidence_weight", mu=0.0, sigma=0.5)
    range_evidence_weight = pm.LogNormal("range_evidence_weight", mu=0.0, sigma=0.5)
    log_gain = pm.Normal("log_gain", mu=0.0, sigma=0.5)
    bias = pm.Normal("bias", mu=0.0, sigma=1.0)
    anchor_weight = pm.LogNormal("anchor_weight", mu=0.0, sigma=0.75)
    size_exponent = pm.Normal("size_exponent", mu=0.0, sigma=0.5)
    # The mechanism: similarity gradient width (number-line units) of range edges.
    edge_width = pm.LogNormal("edge_width", mu=np.log(1.0), sigma=1.0)

    member_t = pt.exp(-pt.as_tensor(DIST_T) / edge_width)  # (100, intervals): graded membership
    size = member_t.sum(axis=0, keepdims=True)  # perceived size, fringe included
    interval_lik_t = pt.log((1 - EPSILON) * member_t / size + EPSILON / 100)  # (100, intervals)

    counts = anchor_weight * first_onehot + later_counts  # (sets, numbers): evidence weight per number
    rule_ev = counts @ pt.as_tensor(RULE_EXAMPLE_LOG_LIK.T)  # (sets, rules)
    range_ev = counts @ interval_lik_t  # (sets, intervals)

    log_rule_prior = pt.log(rule_weights) + size_exponent * RULE_LOG_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)
    log_rules = pt.log(p_mix) + log_rule_prior[None, :] + rule_evidence_weight * rule_ev
    log_ranges = pt.log1p(-p_mix) - LOG_N_INTERVALS + range_evidence_weight * range_ev
    log_norm = pt.logaddexp(
        pt.logsumexp(log_rules, axis=1, keepdims=True), pt.logsumexp(log_ranges, axis=1, keepdims=True)
    )
    rule_post = pt.exp(log_rules - log_norm)
    range_post = pt.exp(log_ranges - log_norm)

    gen = rule_post @ pt.as_tensor(RULE_MEMBERSHIP) + range_post @ member_t.T  # (sets, 100)
    concept_p = pt.clip(gen.reshape((-1,))[cell_flat], 1e-6, 1 - 1e-6)

    belief_logit = pt.log(concept_p) - pt.log1p(-concept_p)
    cell_p = pt.clip(pm.math.sigmoid(pt.exp(log_gain) * belief_logit + bias), 1e-6, 1 - 1e-6)
    p_yes = pm.Deterministic("p_yes", cell_p[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    first = np.zeros((n_sets, 100))
    later = np.zeros((n_sets, 100))
    for i, s in enumerate(unique):
        first[i, s[0] - 1] = 1.0
        for e in s[1:]:
            later[i, e - 1] += 1.0
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    cells, row_cell = np.unique(set_idx * 100 + tgt, return_inverse=True)
    return {
        "first_onehot": first,
        "later_counts": later,
        "cell_flat": cells.astype("int64"),
        "row_cell": row_cell.astype("int64").reshape(-1),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
