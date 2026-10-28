"""Outlier-tolerant rule concepts plus final-digit rules restricted to a range.

Refinement of `outlier_tolerant_rule_concepts`: the same learner (42 rules, the
5,050 intervals of 1..100 with a Beta midpoint prior and a decade-aligned bonus,
size principle with an inferred stray-example rate, evidence weight
beta * n ** gamma, guessing that falls with set size), but the hypothesis space
also holds digit-banded ranges: the numbers ending in d (d = 0..9) between two
of its members, e.g. "ending in 2 from 22 to 62" (proper sub-ranges of at least
two members). The one change: this family gets its own inferred prior share,
split equally, so sets whose examples share a last digit favour a band near the
examples over the full digit rule (rejecting far same-digit targets) and over a
plain interval (rejecting in-range targets with another last digit).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

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
    bands = []
    for d in range(10):
        members = [i for i in NUMBERS if i % 10 == d]
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                if not (a == 0 and b == len(members) - 1):
                    bands.append(members[a:b + 1])
    return rules, intervals, bands


RULES, INTERVALS, BANDS = concept_sets()
N_RULES, N_INTERVALS, N_BANDS = len(RULES), len(INTERVALS), len(BANDS)
N_HYP = N_RULES + N_INTERVALS + N_BANDS
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS + BANDS], dtype="float64")
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
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    evidence_strength = pm.LogNormal("evidence_strength", mu=0.0, sigma=0.5)
    size_growth = pm.Normal("size_growth", mu=0.0, sigma=1.0)
    weight = evidence_strength * pt.exp(size_growth * pt.log(set_size))  # (sets,)
    decade_bonus = pm.Normal("decade_bonus", mu=0.0, sigma=3.0)

    stray_rate = pm.Beta("stray_rate", 1.5, 6.0)
    log_member = pt.log((1 - stray_rate) * pt.exp(-LOG_CONCEPT_SIZE) + stray_rate / 100)  # (hyps,)
    log_stray = pt.log(stray_rate / 100)
    set_log_lik = set_n_in * log_member[None, :] + (set_size[:, None] - set_n_in) * log_stray

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_interval = log_density[INTERVAL_GRID_POINT] + decade_bonus * DECADE_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval)

    # The one new component: prior share of the digit-banded ranges, split equally.
    band_share = pm.Beta("band_share", 1.0, 9.0)
    log_rest = pt.log1p(-band_share)
    log_prior = pt.concatenate([
        log_rest + pt.log(p_mix) + pt.log(rule_weights),
        log_rest + pt.log1p(-p_mix) + log_interval_prior,
        pt.log(band_share) - np.log(N_BANDS) + pt.zeros(N_BANDS),
    ])
    log_joint = log_prior[None, :] + weight[:, None] * set_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, hyps)
    in_concept = pt.dot(posterior, MEMBERSHIP)[cell_set, cell_target]  # (cells,)

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
