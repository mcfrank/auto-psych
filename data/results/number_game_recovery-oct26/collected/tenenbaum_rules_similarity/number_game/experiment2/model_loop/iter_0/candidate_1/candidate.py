"""People do Bayesian concept learning over 42 arithmetic rules (kind-level
prior: a Dirichlet share for powers, multiples, final digit, contained digit and
special sets, split equally within a kind) and numeric ranges, with the size
principle (10% of examples random) and "yes" = posterior probability that the
concept contains the target — but the only ranges they entertain are whole
blocks of consecutive decades (10a+1 .. 10b, 55 blocks), with an Erlang prior
over how many decades a block spans. Range generalization is therefore flat
within the enclosing decade block and stops abruptly at decade boundaries.
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
    kinds = [0] * 9 + [1] * 10 + [2] * 9 + [3] * 9 + [4] * 5
    # Decade blocks: decades a..b (0-indexed), i.e. numbers 10a+1 .. 10(b+1).
    blocks = [(a, b) for a in range(10) for b in range(a, 10)]
    return rules, kinds, blocks


RULES, RULE_KINDS, BLOCKS = concept_sets()
N_RULES = len(RULES)
N_KINDS = 5
N_BLOCKS = len(BLOCKS)
KIND_MATRIX = np.eye(N_KINDS)[RULE_KINDS]
KIND_COUNTS = KIND_MATRIX.sum(axis=0)
BLOCK_SETS = [list(range(10 * a + 1, 10 * (b + 1) + 1)) for a, b in BLOCKS]
BLOCK_DECADES = np.array([b - a + 1 for a, b in BLOCKS], dtype="float64")
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + BLOCK_SETS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)

with pm.Model() as model:
    set_log_lik = pm.Data("set_log_lik", np.zeros((1, N_RULES + N_BLOCKS)))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_in_hypothesis = pm.Data("cell_in_hypothesis", np.zeros((1, N_RULES + N_BLOCKS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    kind_weights = pm.Dirichlet("kind_weights", a=np.full(N_KINDS, 2.0))
    rule_weights = pt.dot(KIND_MATRIX, kind_weights / KIND_COUNTS)
    # Typical block length, in decades.
    block_scale = pm.LogNormal("block_scale", mu=np.log(2.0), sigma=1.0)

    log_block = np.log(BLOCK_DECADES) - BLOCK_DECADES / block_scale
    log_block_prior = log_block - pt.logsumexp(log_block)

    log_prior = pt.concatenate([pt.log(p_mix) + pt.log(rule_weights), pt.log1p(-p_mix) + log_block_prior])
    log_joint = log_prior[None, :] + set_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (posterior[cell_set] * cell_in_hypothesis).sum(axis=1)

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
    cell_set = np.array([index[s] for s, _ in cells], dtype="int64")
    cell_target = np.array([t - 1 for _, t in cells])
    return {
        "set_log_lik": set_log_lik,
        "cell_set": cell_set,
        "cell_in_hypothesis": MEMBERSHIP[:, cell_target].T,
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
