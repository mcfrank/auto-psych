"""Bayesian concept learning with ranges sized on a compressed mental number line.

Refinement of `decade_aligned_range_concepts`: the same learner (42 rules, the
5,050 intervals of 1..100 with a Beta midpoint prior and a decade-aligned prior
bonus, size principle with 10% noise, evidence weight beta * n ** gamma), but
people judge the size of a number range on a compressed mental number line:
number i occupies psychological width (i + 0.5) ** a - (i - 0.5) ** a, with the
compression exponent a inferred (a = 1 is the incumbent's linear line, a -> 0
approaches a logarithmic line). An interval's size, and the probability it
produced each example, are measured in these compressed units, so
generalization is scale-invariant: a range around large examples extends
further in absolute terms than one around small examples.
"Yes" is the posterior probability that the concept contains the target.
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
    return rules, bounds


RULES, BOUNDS = concept_sets()
N_RULES, N_INTERVALS = len(RULES), len(BOUNDS)
LOWER = np.array([m for m, _ in BOUNDS], dtype="int64")
UPPER = np.array([n for _, n in BOUNDS], dtype="int64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LOWER[:, None]) & (NUMBERS[None, :] <= UPPER[:, None])).astype("float64")
MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, INTERVAL_MEMBERSHIP])
RULE_LOG_LIK = np.log(RULE_MEMBERSHIP * (1 - EPSILON) / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)

MIDPOINT_GRID = np.clip(np.arange(199) / 198, 1e-5, 1 - 1e-5)
INTERVAL_GRID_POINT = LOWER + UPPER - 2

DECADE_STARTS = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90}
DECADE_ENDS = {9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 100}
DECADE_ALIGNED = np.array([1.0 if m in DECADE_STARTS and n in DECADE_ENDS else 0.0 for m, n in BOUNDS])

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules): log P(D | rule)
    # Every (distinct example, interval containing it) pair, and how often each set shows each example.
    pair_example = pm.Data("pair_example", np.zeros(1, dtype="int64"))  # (pairs,): row in set_examples
    pair_number = pm.Data("pair_number", np.zeros(1, dtype="int64"))  # (pairs,): example - 1
    pair_interval = pm.Data("pair_interval", np.zeros(1, dtype="int64"))  # (pairs,)
    set_examples = pm.Data("set_examples", np.zeros((1, 1)))  # (sets, distinct examples): counts
    set_size = pm.Data("set_size", np.ones(1, dtype="float64"))  # (sets,)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    evidence_strength = pm.LogNormal("evidence_strength", mu=0.0, sigma=0.5)
    size_growth = pm.Normal("size_growth", mu=0.0, sigma=1.0)
    weight = evidence_strength * pt.exp(size_growth * pt.log(set_size))  # (sets,)
    decade_bonus = pm.Normal("decade_bonus", mu=0.0, sigma=3.0)
    # Compression of the mental number line (1 = linear, toward 0 = logarithmic).
    compression = pm.Beta("compression", 2.0, 1.5)

    # Psychological width of each number and size of each interval on the compressed line.
    edges = pt.power(pt.as_tensor_variable(np.arange(0.5, 101.0, 1.0)), compression)  # (101,)
    width = edges[1:] - edges[:-1]  # (100,)
    interval_size = edges[UPPER] - edges[LOWER - 1]  # (intervals,)
    # Log-likelihood gain of an example inside an interval over the noise floor log(EPSILON / 100).
    density = width[pair_number] / interval_size[pair_interval]  # (pairs,)
    gain = pt.log1p((1 - EPSILON) * 100 / EPSILON * density)
    example_gain = pt.zeros((set_examples.shape[1], N_INTERVALS))[pair_example, pair_interval].set(gain)
    interval_log_lik = (
        pt.dot(set_examples, example_gain) + set_size[:, None] * np.log(EPSILON / 100)
    )  # (sets, intervals)
    set_log_lik = pt.concatenate([rule_log_lik, interval_log_lik], axis=1)

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_interval = log_density[INTERVAL_GRID_POINT] + decade_bonus * DECADE_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval)

    log_prior = pt.concatenate([pt.log(p_mix) + pt.log(rule_weights), pt.log1p(-p_mix) + log_interval_prior])
    log_joint = log_prior[None, :] + weight[:, None] * set_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, hyps)
    in_concept = pt.dot(posterior, MEMBERSHIP)[cell_set, cell_target]  # (cells,)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    distinct = sorted({x for s in unique for x in s})
    position = {x: j for j, x in enumerate(distinct)}
    set_examples = np.zeros((len(unique), len(distinct)))
    for i, s in enumerate(unique):
        for x in s:
            set_examples[i, position[x]] += 1.0
    pair_j, pair_h = np.nonzero(INTERVAL_MEMBERSHIP[:, np.array(distinct) - 1].T)
    return {
        "rule_log_lik": np.stack([RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "pair_example": pair_j,
        "pair_number": np.array(distinct, dtype="int64")[pair_j] - 1,
        "pair_interval": pair_h,
        "set_examples": set_examples,
        "set_size": np.array([float(len(s)) for s in unique], dtype="float64"),
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
