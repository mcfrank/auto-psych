"""Pedagogical teacher sampling: people assume the examples were chosen by a
helpful teacher, not sampled at random from the program's concept. Concepts
are 42 arithmetic rules (primes excluding 1, kind-level prior) and every
interval of 1-100 (Erlang width prior, decade-boundary preference). A teacher
with concept h picks example x in h with probability proportional to
P(x)^-alpha, where P(x) is how likely x is under the learner's prior over
concepts: distinctive numbers, which few plausible concepts produce, point
most clearly to h. alpha (inferred, >= 0) is the strength of the teacher
assumption; alpha = 0 is ordinary strong sampling. 10% of examples are random;
"yes" = posterior probability the concept contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_EX = 4


def concept_sets():
    rules = [
        [2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36],
        [7, 49], [8, 64], [9, 81], [10, 100],
    ]
    rules += [[i for i in NUMBERS if i % n == 0] for n in range(3, 13)]
    rules += [[i for i in NUMBERS if i % 10 == d] for d in range(1, 10)]
    rules += [[i for i in NUMBERS if str(d) in str(i)] for d in range(1, 10)]
    rules.append([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
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
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
INTERVAL_MEMBERSHIP = np.array([(NUMBERS >= m) & (NUMBERS <= n) for m, n in BOUNDS], dtype="float64")
MEMBERSHIP = np.concatenate([RULE_MEMBERSHIP, INTERVAL_MEMBERSHIP])
SIZES = MEMBERSHIP.sum(axis=1)
UNIFORM_LIK = MEMBERSHIP / SIZES[:, None]
LOG_MASK = np.where(MEMBERSHIP > 0, 0.0, -1e4)
INTERVAL_WIDTH = np.array([n - m + 1 for m, n in BOUNDS], dtype="float64")
N_HYP = len(MEMBERSHIP)
DECADE_BOUNDED = np.array([(m - 1) % 10 == 0 and n % 10 == 0 for m, n in BOUNDS], dtype="float64")


with pm.Model() as model:
    examples = pm.Data("examples", np.zeros((1, MAX_EX), dtype="int64"))
    example_in = pm.Data("example_in", np.zeros((1, MAX_EX, N_HYP)))
    example_used = pm.Data("example_used", np.ones((1, MAX_EX)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    rule_share = pm.Beta("rule_share", alpha=2.0, beta=2.0)
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    width_scale = pm.LogNormal("width_scale", mu=np.log(15.0), sigma=1.0)
    decade_boost = pm.Normal("decade_boost", mu=1.0, sigma=1.5)
    # Strength of the assumption that a teacher chose distinctive examples.
    teach = pm.HalfNormal("teach", sigma=1.0)

    log_rw = pt.log(pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS))
    log_int = np.log(INTERVAL_WIDTH) - INTERVAL_WIDTH / width_scale + decade_boost * DECADE_BOUNDED
    log_int = log_int - pt.logsumexp(log_int)
    log_prior = pt.concatenate([pt.log(rule_share) + log_rw, pt.log(1 - rule_share) + log_int])

    # Learner's prior predictive of a single randomly drawn example.
    log_px = pt.log(pt.dot(pt.exp(log_prior), UNIFORM_LIK))
    # Teacher's preference for each number (distinctive = rare under the
    # prior), and its normalizer over the numbers each concept contains.
    appeal = pt.exp(-teach * (log_px - pt.min(log_px)))
    norm = pt.dot(MEMBERSHIP, appeal)
    lik = (1 - EPSILON) * example_in * appeal[examples][:, :, None] / norm[None, None, :] + EPSILON / 100.0
    set_ll = (pt.log(lik) * example_used[:, :, None]).sum(axis=1)  # (set, hyp)
    log_post = log_prior[None, :] + set_ll
    post = pt.exp(log_post - pt.logsumexp(log_post, axis=1, keepdims=True))
    pred = pt.dot(post, MEMBERSHIP)  # (set, number)

    p_yes = pm.Deterministic("p_yes", pt.clip(pred[cell_set, cell_target][row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    ex = np.zeros((len(unique), MAX_EX), dtype="int64")
    used = np.zeros((len(unique), MAX_EX))
    for i, s in enumerate(unique):
        ex[i, : len(s)] = np.array(s) - 1
        used[i, : len(s)] = 1.0
    return {
        "examples": ex,
        "example_in": MEMBERSHIP[:, ex].transpose(1, 2, 0) * used[:, :, None],
        "example_used": used,
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
