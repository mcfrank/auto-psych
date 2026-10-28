"""Win-stay, lose-sample sequential concept learning.

People do not keep a full belief over every concept; they read the examples
one at a time while holding a single working guess (win-stay, lose-sample,
Bonawitz et al. 2014). After the first example they sample a concept from the
posterior given that example; each later example the guess contains leaves it
in place, and an example it does not contain makes them resample a guess from
the posterior given all examples seen so far. "Yes" means the held guess
contains the target, apart from blind "yes"/"no" lapses.

The concept space and the posterior used for (re)sampling are those of
`lapsing_growing_evidence_concepts`: 42 rules (Dirichlet weights, share p_mix)
and the 5,050 intervals of 1..100 (Beta midpoint prior), size principle with
10% noise, examples' evidence weighted by one inferred strength (no extra
set-size growth term: the sequential guess-holding process itself is what
makes generalization depend on how many examples were read). The marginal
distribution of the held guess is computed exactly by the recursion
  Q_1 = Post_1,  Q_t = Q_{t-1} * [x_t in h] + (mass of Q_{t-1} on h not
  containing x_t) * Post_t.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

EPSILON = 0.1
NUMBERS = np.arange(1, 101)
MAX_STEPS = 4  # longest example set shown (one to four examples)


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
N_HYP = N_RULES + N_INTERVALS
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(MEMBERSHIP * (1 - EPSILON) / MEMBERSHIP.sum(axis=1, keepdims=True) + EPSILON / 100)

MIDPOINT_GRID = np.clip(np.arange(199) / 198, 1e-5, 1 - 1e-5)
INTERVAL_GRID_POINT = np.array([c[0] + c[-1] - 2 for c in INTERVALS])

with pm.Model() as model:
    # Sets are ordered by decreasing size, so the sets still reading at step t
    # are the first ones. Step 1: log P(x_1 | h) for every set. Step t > 1, for
    # the sets with at least t examples: cumulative log P(x_1..x_t | h) and
    # whether h contains x_t.
    first_log_lik = pm.Data("first_log_lik", np.zeros((1, N_HYP)))
    later_log_lik = [pm.Data(f"step{t}_log_lik", np.zeros((1, N_HYP))) for t in range(2, MAX_STEPS + 1)]
    later_contains = [pm.Data(f"step{t}_contains", np.zeros((1, N_HYP))) for t in range(2, MAX_STEPS + 1)]
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    evidence_strength = pm.LogNormal("evidence_strength", mu=0.0, sigma=0.5)
    lapse_yes = pm.Beta("lapse_yes", 1.0, 19.0)
    lapse_no = pm.Beta("lapse_no", 1.0, 19.0)

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_interval = log_density[INTERVAL_GRID_POINT]
    log_interval_prior = log_interval - pt.logsumexp(log_interval)
    log_prior = pt.concatenate([pt.log(p_mix) + pt.log(rule_weights), pt.log1p(-p_mix) + log_interval_prior])

    def posterior(log_lik):
        log_joint = log_prior[None, :] + evidence_strength * log_lik
        return pt.special.softmax(log_joint, axis=1)

    # Each set's final held guess is mapped to target membership as soon as
    # the set has been read in full (sets are ordered longest first).
    n_reading = [log_lik.shape[0] for log_lik in later_log_lik]
    held = posterior(first_log_lik)  # (sets, hyps): distribution of the held guess
    finished = [pt.dot(held[n_reading[0]:], MEMBERSHIP)]
    for i, (log_lik, contains) in enumerate(zip(later_log_lik, later_contains)):
        reading = held[: n_reading[i]]
        lost = (reading * (1.0 - contains)).sum(axis=1, keepdims=True)  # guess fails x_t
        held = reading * contains + lost * posterior(log_lik)
        n_next = n_reading[i + 1] if i + 1 < len(n_reading) else 0
        finished.append(pt.dot(held[n_next:], MEMBERSHIP))
    in_concept = pt.concatenate(finished[::-1], axis=0)[cell_set, cell_target]  # (cells,)
    p_concept = in_concept[row_cell]
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(lapse_yes + (1 - lapse_yes - lapse_no) * p_concept, 1e-6, 1 - 1e-6)
    )
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = sorted(dict.fromkeys(sets), key=len, reverse=True)
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    out = {"first_log_lik": np.stack([EXAMPLE_LOG_LIK[:, s[0] - 1] for s in unique])}
    for t in range(2, MAX_STEPS + 1):
        reading = [s for s in unique if len(s) >= t]
        out[f"step{t}_log_lik"] = np.array([EXAMPLE_LOG_LIK[:, np.array(s[:t]) - 1].sum(axis=1) for s in reading]).reshape(-1, N_HYP)
        out[f"step{t}_contains"] = np.array([MEMBERSHIP[:, s[t - 1] - 1] for s in reading]).reshape(-1, N_HYP)
    return {
        **out,
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
