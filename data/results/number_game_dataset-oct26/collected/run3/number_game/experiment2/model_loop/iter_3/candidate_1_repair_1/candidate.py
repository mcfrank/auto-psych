"""Exception tolerance learned over the session.

Refinement of `outlier_tolerant_rule_concepts`: the same learner (42 rules, the
5,050 intervals of 1..100 with a Beta midpoint prior and decade-aligned bonus,
size principle, evidence weight beta * n ** gamma, guessing that falls with set
size), but people's belief that an example is a stray number drawn uniformly
from 1..100 is learned from experience: in the second half of the session,
after many sets that fit no neat rule, they are more tolerant of exceptions, so
examples rule out concepts less firmly and generalization broadens.
The one change: logit eps = stray_base +/- stray_growth / 2 for the first and
second half of the session, with P(x | h) = (1 - eps) [x in h] / |h| + eps / 100.
The posterior is computed once per (distinct set, session half).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
SESSION_LENGTH = 449.0
HALF_POS = np.array([-0.5, 0.5])  # first / second half of the session


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
INV_CONCEPT_SIZE = 1.0 / MEMBERSHIP.sum(axis=1)

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
    cell_half = pm.Data("cell_half", np.zeros(1, dtype="int64"))
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

    # Belief in stray examples, learned with experience (one value per session half).
    stray_base = pm.Normal("stray_base", mu=-2.0, sigma=1.0)
    stray_growth = pm.Normal("stray_growth", mu=0.0, sigma=1.0)
    stray_rate = pm.math.sigmoid(stray_base + stray_growth * HALF_POS)  # (halves,)
    # Log-likelihood of a member example relative to a stray one; strays cancel in the posterior.
    odds = (1 - stray_rate) / stray_rate * 100.0
    log_ratio = pt.log1p(odds[:, None] * INV_CONCEPT_SIZE[None, :])  # (halves, hyps)
    set_log_lik = set_n_in[:, None, :] * log_ratio[None, :, :]  # (sets, halves, hyps)

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_interval = log_density[INTERVAL_GRID_POINT] + decade_bonus * DECADE_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval)

    log_prior = pt.concatenate([pt.log(p_mix) + pt.log(rule_weights), pt.log1p(-p_mix) + log_interval_prior])
    log_joint = log_prior[None, None, :] + weight[:, None, None] * set_log_lik
    posterior = pt.special.softmax(log_joint, axis=2)
    in_concept = pt.dot(posterior, MEMBERSHIP)[cell_set, cell_half, cell_target]  # (cells,)

    guess_single = pm.Normal("guess_single", mu=-2.0, sigma=1.5)
    guess_decline = pm.Normal("guess_decline", mu=0.0, sigma=2.0)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    guess_rate = pm.math.sigmoid(guess_single - guess_decline * pt.log(set_size))[cell_set]  # (cells,)
    q = guess_rate * guess_yes + (1 - guess_rate) * in_concept
    p_yes = pm.Deterministic("p_yes", pt.clip(q[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _half(r):
    try:
        t = float(r.get("trial_index", SESSION_LENGTH / 2))
    except (TypeError, ValueError):
        t = SESSION_LENGTH / 2
    return 1 if t >= SESSION_LENGTH / 2 else 0


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target`, `trial_index` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    keys = [(index[s], _half(r), int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    return {
        "set_n_in": np.stack([MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_size": np.array([float(len(s)) for s in unique], dtype="float64"),
        "cell_set": np.array([c[0] for c in cells], dtype="int64"),
        "cell_half": np.array([c[1] for c in cells], dtype="int64"),
        "cell_target": np.array([c[2] for c in cells], dtype="int64"),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r.get("rating", 0) or 0) for r in rows], dtype="int64"),
    }
