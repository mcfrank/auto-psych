"""Individual rule leaning, on individually loose range sampling.

Refinement of individual_range_sampling_looseness. People do the same
capacity-limited Bayesian learning over 45 rules (kind plausibilities, sparse
heavy-tailed familiarities within kinds) and all intervals of 1-100, reading the
examples more loosely for ranges than for rules, with each participant's range
sampling strength scaled by a personal factor (fixed small spread). The single
change, taken from loose_range_individual_rule_leaning: participants also
differ in how readily they reach for a rule rather than a range a priori, a
personal shift of the prior log odds of all rules against the ranges
(zero-sum across participants, fixed spread). Unlike range looseness, this
leaning does not grow with the evidence, so people differ in rule-vs-range
generalization even for small sets. Participants are recovered from where
trial_index restarts; a trial with no session information is predicted for an
unknown participant by averaging over both population spreads (a 7 x 7
Gauss-Hermite product grid). A participant's range belief is expanded around
the population's (range mass to second order, membership to first order).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


def rule_concepts():
    """(family, members) for each rule; families index FAMILIES."""
    primes = [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))]
    rules = [
        (0, [n for n in NUMBERS if n % 2 == 1]),
        (1, list(range(2, 101, 2))),
        (2, primes),
        (3, [n * n for n in range(1, 11)]),
        (3, [n**3 for n in range(1, 5)]),
        (4, [2, 4, 8, 16, 64]),
        (4, [2, 4, 8, 16, 32, 37, 64]),
    ]
    rules += [(8 if m == 5 else 5, list(range(m, 101, m))) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rules += [(4, [m**k for k in range(1, 8) if m**k <= 100]) for m in range(2, 11)]
    rules += [(6, [n for n in NUMBERS if n % 10 == d]) for d in range(10)]
    rules += [(7, [n for n in NUMBERS if str(d) in str(n)]) for d in range(10)]
    return rules


FAMILIES = ["odd", "even", "primes", "squares_cubes", "powers", "multiples", "final_digit", "contains_digit", "multiples_of_5"]
_RULES = rule_concepts()
RULE_FAMILY = np.array([f for f, _ in _RULES], dtype="int64")
N_RULES = len(_RULES)
N_FAM = len(FAMILIES)
FAMILY_COUNT = np.bincount(RULE_FAMILY, minlength=N_FAM)
FAMILY_LOG_COUNT = np.log(FAMILY_COUNT).astype("float64")
KIND_ONEHOT = (RULE_FAMILY[:, None] == np.arange(N_FAM)[None, :]).astype("float64")  # (rules, kinds)
# How much rules of the same kind differ in familiarity (log prior odds), fixed.
FAMILIARITY_SPREAD = 2.5
# Only rules sharing their kind with other rules have a familiarity deviation.
MULTI_RULES = np.flatnonzero(FAMILY_COUNT[RULE_FAMILY] > 1)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for _, c in _RULES], dtype="float64")
INTERVALS = np.array([(m, n) for m in NUMBERS for n in range(m, 101)])
N_INTERVALS = len(INTERVALS)
INTERVAL_MEMBERSHIP = (NUMBERS[None, :] >= INTERVALS[:, :1]) & (NUMBERS[None, :] <= INTERVALS[:, 1:])
INTERVAL_MEMBERSHIP = INTERVAL_MEMBERSHIP.astype("float64")


def _example_log_lik(membership):
    return np.log(ALPHA * membership / membership.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)


RULE_EX_LL = _example_log_lik(RULE_MEMBERSHIP)  # (rules, 100)
INTERVAL_EX_LL = _example_log_lik(INTERVAL_MEMBERSHIP)  # (intervals, 100)

# Spread (sd of the log factor) of participants' range sampling strengths, fixed.
RANGE_SPREAD = 0.06
N_PART = 15  # participant slots; any extra participants share the last slot
# Spread (sd) of participants' leanings towards rules over ranges (prior log odds), fixed.
LEANING_SPREAD = 0.7
_X, _W = np.polynomial.hermite_e.hermegauss(7)
# Quadrature nodes over an unknown participant's (range factor, leaning) pair.
GH_RANGE = np.repeat(_X, len(_X))
GH_LEAN = np.tile(_X, len(_X))
GH_W = np.outer(_W, _W).reshape(-1)
GH_W = GH_W / GH_W.sum()
K_SLOTS = len(GH_W)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    set_log_n = pm.Data("set_log_n", np.zeros(1))  # log(number of examples) per set
    cell_rule_in = pm.Data("cell_rule_in", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,) distinct (set, target)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, 1)))  # (sets, interval groups)
    group_log_size = pm.Data("group_log_size", np.zeros((1, 1)))  # (sets, interval groups)
    cell_frac = pm.Data("cell_frac", np.zeros((1, 1)))  # (cells, interval groups) fraction containing the target
    cell_frac_lik = pm.Data("cell_frac_lik", np.zeros((1, 1)))  # (cells, interval groups) cell_frac * log lik
    # Distinct (learner, cell) units: a participant slot, or a quadrature node for an unknown participant.
    unit_cell = pm.Data("unit_cell", np.zeros(1, dtype="int64"))  # (units,)
    unit_slot = pm.Data("unit_slot", np.zeros(1, dtype="int64"))  # (units,)
    # Cells asked of an unknown participant, averaged over the quadrature units.
    avg_unit = pm.Data("avg_unit", np.zeros((1, K_SLOTS), dtype="int64"))  # (unknown cells, K)
    avg_weight = pm.Data("avg_weight", np.zeros((1, K_SLOTS)))  # (unknown cells, K)
    row_value = pm.Data("row_value", np.zeros(1, dtype="int64"))  # (rows,) into [units, unknown cells]
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior log-odds of each kind of rule against the intervals as a whole.
    family_logit = pm.Normal("family_logit", mu=0.0, sigma=2.5, shape=N_FAM)
    # Sampling strength of a single example for rules, and the population's for ranges.
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)
    beta_range = pm.LogNormal("beta_range", mu=0.0, sigma=0.7)
    gamma = pm.Normal("gamma", mu=0.0, sigma=0.5)  # capacity limit: loss of evidence per (log) example

    familiarity_raw = pm.StudentT("familiarity_z", nu=5.0, mu=0.0, sigma=0.4, shape=len(MULTI_RULES))
    familiarity_z = pt.set_subtensor(pt.zeros(N_RULES)[MULTI_RULES], familiarity_raw)
    kind_mean_z = (pt.dot(KIND_ONEHOT.T, familiarity_z) / FAMILY_COUNT)[RULE_FAMILY]
    log_rule_raw = family_logit[RULE_FAMILY] - FAMILY_LOG_COUNT[RULE_FAMILY] + FAMILIARITY_SPREAD * (familiarity_z - kind_mean_z)
    log_norm = pt.logsumexp(pt.concatenate([log_rule_raw, pt.zeros(1)]))
    log_rule_prior = log_rule_raw - log_norm
    log_interval_prior = -log_norm - np.log(N_INTERVALS)

    capacity = pt.exp(-gamma * set_log_n)
    # Rules are read with the shared strength: belief within the rules per set.
    log_rules = log_rule_prior[None, :] + beta * capacity[:, None] * rule_log_lik
    log_rule_mass = pt.logsumexp(log_rules, axis=1)
    rule_in_cell = (pt.exp(log_rules - log_rule_mass[:, None])[cell_set] * cell_rule_in).sum(axis=1)

    # Belief within the ranges for a learner with the population's range strength.
    strength0 = beta_range * capacity  # (sets,)
    log_groups = log_interval_prior + group_log_size + strength0[:, None] * group_log_lik
    log_range_mass0 = pt.logsumexp(log_groups, axis=1)
    within0 = pt.exp(log_groups - log_range_mass0[:, None])
    mean_ll = (within0 * group_log_lik).sum(axis=1)  # mean example log lik of the believed ranges
    var_ll = pt.maximum((within0 * group_log_lik**2).sum(axis=1) - mean_ll**2, 0.0)
    within0_cell = within0[cell_set]
    range_in0 = (within0_cell * cell_frac).sum(axis=1)
    # d range_in / d strength: covariance of "contains the target" with the log lik.
    range_slope = (within0_cell * cell_frac_lik).sum(axis=1) - range_in0 * mean_ll[cell_set]

    # Each participant's log factor on the range sampling strength (zero-sum,
    # fixed spread); the quadrature nodes stand for an unknown participant. The
    # factors are small, so a participant's range mass and range membership are
    # expanded around the population's (to second and first order).
    range_offset_z = pm.ZeroSumNormal("range_offset", sigma=RANGE_SPREAD, shape=N_PART)
    log_factor = pt.concatenate([range_offset_z, RANGE_SPREAD * GH_RANGE])
    # Each participant's leaning towards rules over ranges (zero-sum, fixed spread).
    leaning_z = pm.ZeroSumNormal("leaning", sigma=LEANING_SPREAD, shape=N_PART)
    leaning = pt.concatenate([leaning_z, LEANING_SPREAD * GH_LEAN])
    unit_set = cell_set[unit_cell]
    d_strength = strength0[unit_set] * pt.expm1(log_factor[unit_slot])
    log_range_mass = log_range_mass0[unit_set] + d_strength * mean_ll[unit_set] + 0.5 * d_strength**2 * var_ll[unit_set]
    range_in = pt.clip(range_in0[unit_cell] + d_strength * range_slope[unit_cell], 0.0, 1.0)
    w_unit = pm.math.sigmoid(log_rule_mass[unit_set] - log_range_mass + leaning[unit_slot])
    in_unit = w_unit * rule_in_cell[unit_cell] + (1 - w_unit) * range_in
    in_unknown = (in_unit[avg_unit] * avg_weight).sum(axis=1)
    in_concept = pt.concatenate([in_unit, in_unknown])[row_value]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target`, `rating` and
    `trial_index` (a participant's session restarts where it fails to increase;
    rows without it are an unknown participant, averaged over quadrature nodes).

    Intervals with the same likelihood of a set share a posterior and are grouped
    per distinct set; padding groups get a log size of -1e4 (no posterior mass).
    """
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}

    rule_ll = np.stack([RULE_EX_LL[:, np.array(s) - 1].sum(axis=1) for s in unique])
    groupings = []
    for s in unique:
        ll = INTERVAL_EX_LL[:, np.array(s) - 1].sum(axis=1)
        keys, inverse = np.unique(np.round(ll, 9), return_inverse=True)
        groupings.append((inverse.reshape(-1), keys))
    n_groups = max(len(k) for _, k in groupings)
    log_lik = np.zeros((len(unique), n_groups))
    log_size = np.full((len(unique), n_groups), -1e4)
    sizes = np.ones((len(unique), n_groups))
    for i, (inverse, keys) in enumerate(groupings):
        counts = np.bincount(inverse, minlength=len(keys))
        log_lik[i, : len(keys)] = keys
        log_size[i, : len(keys)], sizes[i, : len(keys)] = np.log(counts), counts
    cells = list(dict.fromkeys((s, int(r["target"])) for s, r in zip(sets, rows)))
    cell_index = {c: k for k, c in enumerate(cells)}
    frac = np.zeros((len(cells), n_groups))
    rule_in = np.zeros((len(cells), N_RULES))
    for k, (s, t) in enumerate(cells):
        i = set_index[s]
        inverse, keys = groupings[i]
        frac[k, : len(keys)] = np.bincount(inverse, weights=INTERVAL_MEMBERSHIP[:, t - 1], minlength=len(keys))
        frac[k] /= sizes[i]
        rule_in[k] = RULE_MEMBERSHIP[:, t - 1]

    persons, prev, pid = [], None, 0
    for r in rows:
        t = r.get("trial_index")
        if t is None or str(t).strip() == "":
            persons.append(-1)
            continue
        t = int(float(t))
        if prev is not None and t <= prev:
            pid += 1
        prev = t
        persons.append(min(pid, N_PART - 1))

    unit_index, unit_cell, unit_slot = {}, [], []

    def unit(slot, c):
        if (slot, c) not in unit_index:
            unit_index[(slot, c)] = len(unit_index)
            unit_cell.append(c)
            unit_slot.append(slot)
        return unit_index[(slot, c)]

    row_ref = []
    for st, r, p in zip(sets, rows, persons):
        c = cell_index[(st, int(r["target"]))]
        row_ref.append(("u", unit(p, c)) if p >= 0 else ("a", c))
    avg_cells = list(dict.fromkeys(c for kind, c in row_ref if kind == "a"))
    avg_index = {c: k for k, c in enumerate(avg_cells)}
    avg_unit = np.zeros((max(len(avg_cells), 1), K_SLOTS), dtype="int64")
    avg_weight = np.zeros((max(len(avg_cells), 1), K_SLOTS))
    for k, c in enumerate(avg_cells):
        avg_unit[k] = [unit(N_PART + q, c) for q in range(K_SLOTS)]
        avg_weight[k] = GH_W
    n_units = len(unit_index)
    row_value = [v if kind == "u" else n_units + avg_index[v] for kind, v in row_ref]
    return {
        "rule_log_lik": rule_ll,
        "set_log_n": np.log(np.array([len(s) for s in unique], dtype="float64")),
        "cell_set": np.array([set_index[st] for st, _ in cells], dtype="int64"),
        "cell_rule_in": rule_in,
        "group_log_lik": log_lik,
        "group_log_size": log_size,
        "cell_frac": frac,
        "cell_frac_lik": frac * log_lik[[set_index[st] for st, _ in cells]],
        "unit_cell": np.array(unit_cell, dtype="int64"),
        "unit_slot": np.array(unit_slot, dtype="int64"),
        "avg_unit": avg_unit,
        "avg_weight": avg_weight,
        "row_value": np.array(row_value, dtype="int64"),
        "rating": np.array([int(float(r.get("rating", 0))) for r in rows], dtype="int64"),
    }
