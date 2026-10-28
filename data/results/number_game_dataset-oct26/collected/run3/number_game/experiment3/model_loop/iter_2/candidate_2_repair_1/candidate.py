"""Primality-blind concept learning.

People learn the program's concept by the same conservative Bayesian concept
learning as `outlier_tolerant_rule_concepts` (arithmetic rules, midpoint-weighted
decade-favoured intervals, evidence weight beta * n ** gamma, an inferred stray
rate, guessing that falls with set size), except that "prime numbers" is not a
concept they spontaneously consider: a set of primes is read as odd numbers or
as a range. The one change: the primes rule is removed from the 42 rules (41
remain), so far odd non-primes after a prime set are endorsed as readily as far
primes. For a well-identified fit, prior belief in the rules is inferred per
family of rules (powers, multiples, final digit, contained digit, parity,
squares, cubes) and split equally within a family, instead of one free weight
per rule.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)


def concept_sets():
    families = [0] * 9 + [1] * 10 + [2] * 9 + [3] * 9 + [4, 4, 5, 6]
    rules = [
        [2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36],
        [7, 49], [8, 64], [9, 81], [10, 100],
    ]
    rules += [[i for i in NUMBERS if i % n == 0] for n in range(3, 13)]
    rules += [[i for i in NUMBERS if i % 10 == d] for d in range(1, 10)]
    rules += [[i for i in NUMBERS if str(d) in str(i)] for d in range(1, 10)]
    # No "primes" rule: primality is not a concept people spontaneously entertain.
    rules.append([i for i in NUMBERS if i % 2 == 0])
    rules.append([i for i in NUMBERS if i % 2 == 1])
    rules.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    rules.append([1, 8, 27, 64])
    intervals = [list(range(m, n + 1)) for m in range(1, 101) for n in range(m, 101)]
    return rules, intervals, families


RULES, INTERVALS, FAMILIES = concept_sets()
N_RULES, N_INTERVALS = len(RULES), len(INTERVALS)
N_HYP = N_RULES + N_INTERVALS
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
N_FAMILIES = max(FAMILIES) + 1
FAMILY_ONEHOT = np.eye(N_FAMILIES)[FAMILIES]  # (rules, families)
LOG_FAMILY_COUNT = np.log(FAMILY_ONEHOT.sum(axis=0))
LOG_CONCEPT_SIZE = np.log(MEMBERSHIP.sum(axis=1))

MIDPOINT_GRID = np.clip(np.arange(199) / 198, 1e-5, 1 - 1e-5)
INTERVAL_GRID_POINT = np.array([c[0] + c[-1] - 2 for c in INTERVALS])

DECADE_STARTS = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90}
DECADE_ENDS = {9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 100}
DECADE_ALIGNED = np.array(
    [1.0 if c[0] in DECADE_STARTS and c[-1] in DECADE_ENDS else 0.0 for c in INTERVALS]
)

with pm.Model() as model:
    set_n_in = pm.Data("set_n_in", np.zeros((1, N_HYP)))  # (sets, hyps): examples h contains
    set_size = pm.Data("set_size", np.ones(1, dtype="float64"))  # (sets,)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    log_rule_prior = pt.dot(FAMILY_ONEHOT, pt.log(family_weights) - LOG_FAMILY_COUNT)  # (rules,)
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    evidence_strength = pm.LogNormal("evidence_strength", mu=0.0, sigma=0.5)
    size_growth = pm.Normal("size_growth", mu=0.0, sigma=1.0)
    weight = evidence_strength * pt.exp(size_growth * pt.log(set_size))  # (sets,)
    decade_bonus = pm.Normal("decade_bonus", mu=0.0, sigma=3.0)

    # Inferred belief that an example is a stray number from 1..100 (incumbent: 0.1).
    stray_rate = pm.Beta("stray_rate", 1.5, 6.0)
    log_member = pt.log((1 - stray_rate) * pt.exp(-LOG_CONCEPT_SIZE) + stray_rate / 100)  # (hyps,)
    # Examples outside h are strays under every concept, so only the examples h
    # contains change its posterior: log-likelihood gain per contained example.
    member_gain = log_member - pt.log(stray_rate / 100)  # (hyps,)

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_interval = log_density[INTERVAL_GRID_POINT] + decade_bonus * DECADE_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval)

    log_prior = pt.concatenate([pt.log(p_mix) + log_rule_prior, pt.log1p(-p_mix) + log_interval_prior])
    log_joint = log_prior[None, :] + weight[:, None] * set_n_in * member_gain[None, :]
    unnorm = pt.exp(log_joint - pt.max(log_joint, axis=1, keepdims=True))  # (sets, hyps)
    in_concept = (pt.dot(unnorm, MEMBERSHIP) / pt.sum(unnorm, axis=1, keepdims=True))[cell_set, cell_target]

    guess_single = pm.Normal("guess_single", mu=-2.0, sigma=1.5)
    guess_decline = pm.Normal("guess_decline", mu=0.0, sigma=2.0)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    guess_rate = pm.math.sigmoid(guess_single - guess_decline * pt.log(set_size))[cell_set]  # (cells,)
    q = guess_rate * guess_yes + (1 - guess_rate) * in_concept
    p_yes = pm.Deterministic("p_yes", pt.clip(q[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    return {
        "set_n_in": np.stack([MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_size": np.array([float(len(s)) for s in unique], dtype="float64"),
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
