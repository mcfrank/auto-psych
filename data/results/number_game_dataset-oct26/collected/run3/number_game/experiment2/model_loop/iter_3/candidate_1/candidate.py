"""Exception tolerance learned over the session.

Refinement of `outlier_tolerant_rule_concepts`: the same learner (42 rules, the
5,050 intervals of 1..100 with a Beta midpoint prior and decade-aligned bonus,
size principle, evidence weight beta * n ** gamma, guessing that falls with set
size), but people's belief that an example is a stray number drawn uniformly
from 1..100 grows with experience over the session: after many sets that fit no
neat rule they become more tolerant of exceptions, so examples rule out
concepts less firmly and generalization broadens late in the session.
The one change: eps(t) = sigmoid(stray_base + stray_growth * session_pos), with
P(x | h) = (1 - eps) [x in h] / |h| + eps / 100. For speed the session is split
into a few phases and the posterior is computed once per (distinct set, phase).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
SESSION_LENGTH = 449.0
N_PHASES = 3  # thirds of the session
PHASE_POS = (np.arange(N_PHASES) + 0.5) / N_PHASES - 0.5  # phase centres in [-0.5, 0.5]


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
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_phase = pm.Data("row_phase", np.zeros(1, dtype="int64"))
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))  # target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    evidence_strength = pm.LogNormal("evidence_strength", mu=0.0, sigma=0.5)
    size_growth = pm.Normal("size_growth", mu=0.0, sigma=1.0)
    weight = evidence_strength * pt.exp(size_growth * pt.log(set_size))  # (sets,)
    decade_bonus = pm.Normal("decade_bonus", mu=0.0, sigma=3.0)

    # Belief in stray examples, learned over the session (one value per phase).
    stray_base = pm.Normal("stray_base", mu=-2.0, sigma=1.0)
    stray_growth = pm.Normal("stray_growth", mu=0.0, sigma=1.5)
    stray_rate = pm.math.sigmoid(stray_base + stray_growth * PHASE_POS)  # (phases,)
    log_member = pt.log(
        (1 - stray_rate[:, None]) * np.exp(-LOG_CONCEPT_SIZE)[None, :] + stray_rate[:, None] / 100
    )  # (phases, hyps)
    # Stray examples add the same term to every concept, which cancels in the posterior.
    log_ratio = log_member - pt.log(stray_rate / 100)[:, None]  # (phases, hyps)
    set_log_lik = set_n_in[:, None, :] * log_ratio[None, :, :]  # (sets, phases, hyps)

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_interval = log_density[INTERVAL_GRID_POINT] + decade_bonus * DECADE_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval)

    log_prior = pt.concatenate([pt.log(p_mix) + pt.log(rule_weights), pt.log1p(-p_mix) + log_interval_prior])
    log_joint = log_prior[None, None, :] + weight[:, None, None] * set_log_lik
    posterior = pt.special.softmax(log_joint, axis=2)
    in_concept = pt.dot(posterior, MEMBERSHIP)  # (sets, phases, 100)
    in_row = in_concept[row_set, row_phase, row_target]  # (rows,)

    guess_single = pm.Normal("guess_single", mu=-2.0, sigma=1.5)
    guess_decline = pm.Normal("guess_decline", mu=0.0, sigma=2.0)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    guess_rate = pm.math.sigmoid(guess_single - guess_decline * pt.log(set_size))[row_set]
    q = guess_rate * guess_yes + (1 - guess_rate) * in_row
    p_yes = pm.Deterministic("p_yes", pt.clip(q, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target`, `trial_index` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    phases = []
    for r in rows:
        try:
            t = float(r.get("trial_index", SESSION_LENGTH / 2))
        except (TypeError, ValueError):
            t = SESSION_LENGTH / 2
        frac = min(max(t / SESSION_LENGTH, 0.0), 1.0)
        phases.append(min(int(frac * N_PHASES), N_PHASES - 1))
    return {
        "set_n_in": np.stack([MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_size": np.array([float(len(s)) for s in unique], dtype="float64"),
        "row_set": np.array([index[s] for s in sets], dtype="int64"),
        "row_phase": np.array(phases, dtype="int64"),
        "row_target": np.array([int(r["target"]) - 1 for r in rows], dtype="int64"),
        "rating": np.array([int(r.get("rating", 0) or 0) for r in rows], dtype="int64"),
    }
