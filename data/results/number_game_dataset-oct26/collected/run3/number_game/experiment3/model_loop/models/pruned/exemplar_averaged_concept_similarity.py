"""Exemplar-averaged concept similarity.

People do not infer one concept explaining all the examples together; they store
each example separately and judge the target by its average similarity to the
stored examples (exemplar generalization). The similarity of the target to one
example is how readily that number alone generalizes to the target (Shepard /
Tenenbaum-Griffiths similarity): summed over familiar categories (42 arithmetic
rules and number ranges with a midpoint and decade-aligned prior) that contain
both, with smaller categories counting more (size weighting with an inferred
strength). Each example is compared on its own, so a target like one member of a
mixed set earns partial "yes", and extra consistent examples do not sharpen
generalization. Occasional blind guesses with an inferred "yes" leaning.
The 100 x 100 single-example similarity matrix is computed once per gradient
step; examples per trial are indices into it.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_SET = 4


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
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")  # (hyps, 100)
LOG_CONCEPT_SIZE = np.log(MEMBERSHIP.sum(axis=1))

MIDPOINT_GRID = np.clip(np.arange(199) / 198, 1e-5, 1 - 1e-5)
INTERVAL_GRID_POINT = np.array([c[0] + c[-1] - 2 for c in INTERVALS])
DECADE_STARTS = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90}
DECADE_ENDS = {9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 100}
DECADE_ALIGNED = np.array(
    [1.0 if c[0] in DECADE_STARTS and c[-1] in DECADE_ENDS else 0.0 for c in INTERVALS]
)
STRAY = 0.1

with pm.Model() as model:
    cell_examples = pm.Data("cell_examples", np.zeros((1, MAX_SET), dtype="int64"))  # example - 1
    cell_mask = pm.Data("cell_mask", np.ones((1, MAX_SET), dtype="float64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    decade_bonus = pm.Normal("decade_bonus", mu=0.0, sigma=3.0)
    size_weight = pm.LogNormal("size_weight", mu=0.0, sigma=0.5)

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_interval = log_density[INTERVAL_GRID_POINT] + decade_bonus * DECADE_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval)
    log_prior = pt.concatenate([pt.log(p_mix) + pt.log(rule_weights), pt.log1p(-p_mix) + log_interval_prior])

    # Single-example generalization: category h's weight for example x is
    # prior(h) * P(x | h) ** size_weight, which takes one value for members and one
    # for non-members; dividing both by the non-member value gives
    # prior(h) * (1 + (ratio_h - 1) [x in h]), so similarity is a weighted Gram matrix.
    log_ratio = pt.log((1 - STRAY) * pt.exp(-LOG_CONCEPT_SIZE) + STRAY / 100) - np.log(STRAY / 100)
    prior = pt.exp(log_prior)
    v = prior * pt.expm1(size_weight * log_ratio)  # (hyps,)
    shared = pt.dot(MEMBERSHIP.T * v[None, :], MEMBERSHIP)  # (example, target)
    base = pt.dot(prior, MEMBERSHIP)  # (target,)
    norm = 1.0 + pt.dot(v, MEMBERSHIP)  # (example,)
    similarity = (base[None, :] + shared) / norm[:, None]

    sims = similarity[cell_examples, cell_target[:, None]]  # (cells, MAX_SET)
    mean_sim = pt.sum(sims * cell_mask, axis=1) / pt.sum(cell_mask, axis=1)

    guess_rate = pm.Beta("guess_rate", 1.5, 8.0)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    q = guess_rate * guess_yes + (1 - guess_rate) * mean_sim
    p_yes = pm.Deterministic("p_yes", pt.clip(q[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    examples = np.zeros((len(cells), MAX_SET), dtype="int64")
    mask = np.zeros((len(cells), MAX_SET), dtype="float64")
    for i, (s, _) in enumerate(cells):
        examples[i, : len(s)] = np.array(s) - 1
        mask[i, : len(s)] = 1.0
    return {
        "cell_examples": examples,
        "cell_mask": mask,
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
