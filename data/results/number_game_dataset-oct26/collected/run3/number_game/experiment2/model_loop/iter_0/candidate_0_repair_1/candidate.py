"""Decade-aligned range concepts with fuzzy (graded) range edges at the decision stage.

Refinement of `session_growing_acquiescence_concepts`: the same learner (42 rules,
the 5,050 intervals of 1..100 with a Beta midpoint prior and a decade-aligned prior
bonus, size principle with 10% noise, evidence weight beta * n ** gamma, blind "yes"
rate growing over the session), but people treat a number range as a vague category:
when judging whether the target belongs to a range they believe in, a target d steps
outside the range's edges counts as inside with graded credit exp(-d / tolerance)
(tolerance inferred; tolerance -> 0 recovers the incumbent). Rules stay crisp and the
examples are still explained by crisp ranges, so only generalization is softened —
most visibly after a single example, whose posterior concentrates on narrow ranges.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
SESSION_LENGTH = 449.0


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
INV_SIZE = 1.0 / MEMBERSHIP.sum(axis=1)

MIDPOINT_GRID = np.clip(np.arange(199) / 198, 1e-5, 1 - 1e-5)
INTERVAL_GRID_POINT = np.array([c[0] + c[-1] - 2 for c in INTERVALS])

# Fuzzy membership of t in [lo, hi] factorises as a(lo, t) + b(hi, t) - 1, with
# a = 1 if lo <= t else exp(-(lo - t) / tol) and b = 1 if t <= hi else exp(-(t - hi) / tol),
# so it only needs the posterior mass on each lower and each upper edge.
LO_ONEHOT = np.array([[1.0 if c[0] == v else 0.0 for v in NUMBERS] for c in INTERVALS])  # (intervals, 100)
HI_ONEHOT = np.array([[1.0 if c[-1] == v else 0.0 for v in NUMBERS] for c in INTERVALS])
LO_BEYOND = np.maximum(NUMBERS[:, None] - NUMBERS[None, :], 0).astype("float64")  # [v, t]: v - t if lower edge v > t
HI_BEYOND = np.maximum(NUMBERS[None, :] - NUMBERS[:, None], 0).astype("float64")  # [v, t]: t - v if upper edge v < t
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]

DECADE_STARTS = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90}
DECADE_ENDS = {9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 100}
DECADE_ALIGNED = np.array(
    [1.0 if c[0] in DECADE_STARTS and c[-1] in DECADE_ENDS else 0.0 for c in INTERVALS]
)

with pm.Model() as model:
    set_k_in = pm.Data("set_k_in", np.zeros((1, N_HYP)))  # (sets, hyps): examples inside h
    set_size = pm.Data("set_size", np.ones(1, dtype="float64"))  # (sets,)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    session_pos = pm.Data("session_pos", np.zeros(1, dtype="float64"))  # per row, in [-0.5, 0.5]
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    evidence_strength = pm.LogNormal("evidence_strength", mu=0.0, sigma=0.5)
    size_growth = pm.Normal("size_growth", mu=0.0, sigma=1.0)
    weight = evidence_strength * pt.exp(size_growth * pt.log(set_size))  # (sets,)
    # Log prior bonus for intervals bounded at decade edges.
    decade_bonus = pm.Normal("decade_bonus", mu=0.0, sigma=3.0)

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_interval = log_density[INTERVAL_GRID_POINT] + decade_bonus * DECADE_ALIGNED
    log_interval_prior = log_interval - pt.logsumexp(log_interval)

    log_prior = pt.concatenate([pt.log(p_mix) + pt.log(rule_weights), pt.log1p(-p_mix) + log_interval_prior])
    log_in = np.log(0.9 * INV_SIZE + 0.1 / 100.0)  # (hyps,)
    log_out = np.log(0.1 / 100.0)
    set_log_lik = set_k_in * log_in[None, :] + (set_size[:, None] - set_k_in) * log_out
    log_joint = log_prior[None, :] + weight[:, None] * set_log_lik
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))  # (sets, hyps)
    # Fuzzy range edges: a target d steps outside a range counts as exp(-d / tolerance) inside.
    edge_tolerance = pm.LogNormal("edge_tolerance", mu=0.0, sigma=0.75)
    post_rule, post_int = posterior[:, :N_RULES], posterior[:, N_RULES:]
    lo_mass = pt.dot(post_int, LO_ONEHOT)  # (sets, 100)
    hi_mass = pt.dot(post_int, HI_ONEHOT)
    edge_lo = pt.exp(-LO_BEYOND / edge_tolerance)  # (100, 100)
    edge_hi = pt.exp(-HI_BEYOND / edge_tolerance)
    in_range = pt.dot(lo_mass, edge_lo) + pt.dot(hi_mass, edge_hi) - post_int.sum(axis=1, keepdims=True)
    in_concept = (pt.dot(post_rule, RULE_MEMBERSHIP) + in_range)[cell_set, cell_target]  # (cells,)

    # Session-growing acquiescence: blind "yes" rate rises with session position.
    lapse_base = pm.Normal("lapse_base", mu=-3.0, sigma=1.0)
    lapse_growth = pm.Normal("lapse_growth", mu=0.0, sigma=2.0)
    blind_yes = pm.math.sigmoid(lapse_base + lapse_growth * session_pos)  # (rows,)
    q = in_concept[row_cell]
    p_yes = pm.Deterministic("p_yes", pt.clip(blind_yes + (1 - blind_yes) * q, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target`, `trial_index` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}
    pos = []
    for r in rows:
        try:
            t = float(r.get("trial_index", SESSION_LENGTH / 2))
        except (TypeError, ValueError):
            t = SESSION_LENGTH / 2
        pos.append(min(max(t / SESSION_LENGTH, 0.0), 1.0) - 0.5)
    return {
        "set_k_in": np.stack([MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_size": np.array([float(len(s)) for s in unique], dtype="float64"),
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "session_pos": np.array(pos, dtype="float64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
