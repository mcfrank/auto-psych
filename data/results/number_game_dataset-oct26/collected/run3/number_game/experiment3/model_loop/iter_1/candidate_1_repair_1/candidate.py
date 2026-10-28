"""Conservative Bayesian concept learning whose believed ranges are read in whole decades.

People learn the concept as the best model does (42 arithmetic rules, the 5,050
intervals of 1..100 with a Beta midpoint prior and a decade-aligned bonus, size
principle with an inferred stray-example rate, evidence weight beta * n ** gamma,
guessing that falls with set size), but when checking a target against a range
they believe in, with an inferred probability they round the range outward to
the whole decades its ends fall in (22-25 read as 20-29; 100 is its own decade),
so same-decade targets just outside the examples' span (28, 29 after 22, 25) are
endorsed far more than equally near targets across a decade boundary (19, 31).
Inference about the concept is unchanged; only the membership check of ranges
is decade-rounded. Counts of examples each concept contains, and the extra
membership a decade-rounded range adds, are computed once outside the graph.
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
    return rules, intervals


RULES, INTERVALS = concept_sets()
N_RULES, N_INTERVALS = len(RULES), len(INTERVALS)
N_HYP = N_RULES + N_INTERVALS
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
LOG_CONCEPT_SIZE = np.log(MEMBERSHIP.sum(axis=1))

# Decade-rounded reading of each interval: every number whose tens digit lies
# between the tens digits of the interval's ends (100 has tens digit 10).
TENS = NUMBERS // 10
ROUNDED_EXTRA = np.array(
    [((TENS >= c[0] // 10) & (TENS <= c[-1] // 10)).astype("float64") for c in INTERVALS]
) - MEMBERSHIP[N_RULES:]  # (intervals, 100): members the rounding adds

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
    cell_member = pm.Data("cell_member", np.zeros((1, N_HYP)))  # (cells, hyps): h contains target
    cell_rounded_extra = pm.Data("cell_rounded_extra", np.zeros((1, N_INTERVALS)))  # rounding adds target
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
    # Per-set constant size * log_stray cancels in each set's normalised posterior.
    set_log_lik = set_n_in * (log_member - log_stray)[None, :]

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_interval = log_density[INTERVAL_GRID_POINT] + decade_bonus * DECADE_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval)

    log_prior = pt.concatenate([pt.log(p_mix) + pt.log(rule_weights), pt.log1p(-p_mix) + log_interval_prior])
    log_joint = log_prior[None, :] + weight[:, None] * set_log_lik
    posterior = pt.special.softmax(log_joint, axis=1)  # (sets, hyps)

    # The one new component: chance a believed range is read in whole decades.
    decade_rounding = pm.Beta("decade_rounding", 1.0, 3.0)
    cell_post = posterior[cell_set]  # (cells, hyps)
    strict = pt.sum(cell_post * cell_member, axis=1)
    rounded_gain = pt.sum(cell_post[:, N_RULES:] * cell_rounded_extra, axis=1)
    in_concept = strict + decade_rounding * rounded_gain  # (cells,)

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
        "cell_member": MEMBERSHIP[:, np.array([t - 1 for _, t in cells])].T.copy(),
        "cell_rounded_extra": ROUNDED_EXTRA[:, np.array([t - 1 for _, t in cells])].T.copy(),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
