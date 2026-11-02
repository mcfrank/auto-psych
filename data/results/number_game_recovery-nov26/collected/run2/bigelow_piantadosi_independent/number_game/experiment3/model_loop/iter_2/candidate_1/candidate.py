"""Individual working-memory capacity in capacity-limited concept learning.

People learn the hidden concept by Bayesian inference over 45 arithmetic rules
(a prior plausibility for each kind of rule, multiples of 5 its own base-ten
kind, sparse heavy-tailed familiarities within kinds) and all intervals of
1-100, reading the examples with one sampling strength for rules and a looser
one for ranges. Evidence integration is capacity limited: the summed example
log-likelihood is scaled by n ** (-gamma), so each extra example adds less.
The one distortion examined here is that this capacity limit is a stable
personal trait: each participant's gamma is the population value plus a
personal offset (zero-sum across the participants, fixed spread). A tighter
limit makes larger sets generalize more broadly for that person, with no
difference for single examples. An unknown person (rows without trial_index) is
averaged over the population spread by Gauss-Hermite quadrature.
"""


import numpy as np
import pymc as pm
import pytensor.sparse as ps
import pytensor.tensor as pt

ALPHA = 0.9
CAPACITY_SPREAD = 0.15  # how much participants differ in capacity limit, fixed
LL_GRID = 0.2  # intervals whose example log-lik agree within this share one group
N_PERSON_SLOTS = 15  # participants with their own capacity; later ones get the population average
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

# Gauss-Hermite nodes and weights for averaging over an unknown person's capacity;
# known persons use the extra node with zero shift.
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(7)
GH_SHIFT = np.concatenate([np.sqrt(2.0) * CAPACITY_SPREAD * _GH_X, np.zeros(1)])
GH_WEIGHT = _GH_W / np.sqrt(np.pi)

with pm.Model() as model:
    unit_rule_ll = pm.Data("unit_rule_ll", np.zeros((1, N_RULES)))  # (units, rules) example log-lik
    unit_group_ll = pm.Data("unit_group_ll", np.zeros((1, 1)))  # (units, interval groups)
    unit_group_log_size = pm.Data("unit_group_log_size", np.zeros((1, 1)))
    unit_log_n = pm.Data("unit_log_n", np.zeros(1))  # log(number of examples)
    unit_person = pm.Data("unit_person", np.zeros(1, dtype="int64"))  # person slot (last: unknown)
    unit_node = pm.Data("unit_node", np.zeros(1, dtype="int64"))  # quadrature node (last: none)
    pair_unit = pm.Data("pair_unit", np.zeros(1, dtype="int32"))  # (pairs,) unit of each pair
    pair_rule_in = pm.Data("pair_rule_in", np.zeros((1, N_RULES)))  # target in each rule
    pair_frac = pm.Data("pair_frac", np.zeros((1, 1)))  # fraction of each group containing the target
    row_pairs = pm.Data("row_pairs", np.zeros((1, 7), dtype="int64"))  # (rows, 7)
    row_weights = pm.Data("row_weights", np.zeros((1, 7)))  # (rows, 7)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_logit = pm.Normal("family_logit", mu=0.0, sigma=2.5, shape=N_FAM)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)
    beta_range = pm.LogNormal("beta_range", mu=0.0, sigma=0.7)
    gamma = pm.Normal("gamma", mu=0.0, sigma=0.5)  # population capacity limit per (log) example
    # Each participant's own capacity limit, a fixed spread around the population.
    capacity_z = pm.ZeroSumNormal("capacity_offset", sigma=CAPACITY_SPREAD, shape=N_PERSON_SLOTS)

    familiarity_raw = pm.StudentT("familiarity_z", nu=5.0, mu=0.0, sigma=0.4, shape=len(MULTI_RULES))
    familiarity_z = pt.set_subtensor(pt.zeros(N_RULES)[MULTI_RULES], familiarity_raw)
    kind_mean_z = (pt.dot(KIND_ONEHOT.T, familiarity_z) / FAMILY_COUNT)[RULE_FAMILY]
    log_rule_raw = family_logit[RULE_FAMILY] - FAMILY_LOG_COUNT[RULE_FAMILY] + FAMILIARITY_SPREAD * (familiarity_z - kind_mean_z)
    log_norm = pt.logsumexp(pt.concatenate([log_rule_raw, pt.zeros(1)]))
    log_rule_prior = log_rule_raw - log_norm
    log_interval_prior = -log_norm - np.log(N_INTERVALS)

    unit_gamma = gamma + pt.concatenate([capacity_z, pt.zeros(1)])[unit_person] + pt.as_tensor(GH_SHIFT)[unit_node]
    capacity = pt.exp(-unit_gamma * unit_log_n)[:, None]
    log_rules = log_rule_prior[None, :] + beta * capacity * unit_rule_ll
    log_groups = log_interval_prior + unit_group_log_size + beta_range * capacity * unit_group_ll
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    # Each pair's unit posterior, selected by a sparse one-hot (pairs x units) matrix.
    n_pairs = pair_unit.shape[0]
    select = ps.CSR(pt.ones(n_pairs), pair_unit, pt.arange(n_pairs + 1, dtype="int32"), pt.stack([n_pairs, posterior.shape[0]]))
    post_pair = ps.structured_dot(select, posterior)
    pair_in = (post_pair[:, :N_RULES] * pair_rule_in).sum(axis=1) + (post_pair[:, N_RULES:] * pair_frac).sum(axis=1)
    in_concept = (pair_in[row_pairs] * row_weights).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target`, `rating` and
    `trial_index` (a participant's session restarts where it fails to increase;
    rows without it are judged by an unknown, population-average learner).

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
        _, inverse = np.unique(np.round(ll / LL_GRID), return_inverse=True)
        inverse = inverse.reshape(-1)
        keys = np.bincount(inverse, weights=ll) / np.bincount(inverse)
        groupings.append((inverse, keys))
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
            persons.append(N_PERSON_SLOTS)
            continue
        t = int(float(t))
        if prev is not None and t <= prev:
            pid += 1
        prev = t
        persons.append(min(pid, N_PERSON_SLOTS))
    # Units: a learner (a known person, or one quadrature node of an unknown
    # one; known people use the zero-shift node 7) judging a set. Pairs: a unit
    # judging one target.
    units, pairs = {}, {}
    row_pairs = np.zeros((len(rows), 7), dtype="int64")
    row_weights = np.zeros((len(rows), 7))

    def pair_of(person, node, s, t):
        u = units.setdefault((person, node, set_index[s]), len(units))
        return pairs.setdefault((u, cell_index[(s, t)]), len(pairs))

    for j, (s, r, person) in enumerate(zip(sets, rows, persons)):
        t = int(r["target"])
        if person < N_PERSON_SLOTS:
            row_pairs[j, 0], row_weights[j, 0] = pair_of(person, 7, s, t), 1.0
        else:
            for node in range(7):
                row_pairs[j, node], row_weights[j, node] = pair_of(person, node, s, t), GH_WEIGHT[node]
    ukeys = np.array(list(units), dtype="int64").reshape(-1, 3)
    pkeys = np.array(list(pairs), dtype="int64").reshape(-1, 2)
    log_n = np.log(np.array([len(s) for s in unique], dtype="float64"))
    return {
        "unit_rule_ll": rule_ll[ukeys[:, 2]],
        "unit_group_ll": log_lik[ukeys[:, 2]],
        "unit_group_log_size": log_size[ukeys[:, 2]],
        "unit_log_n": log_n[ukeys[:, 2]],
        "unit_person": ukeys[:, 0],
        "unit_node": ukeys[:, 1],
        "pair_unit": pkeys[:, 0].astype("int32"),
        "pair_rule_in": rule_in[pkeys[:, 1]],
        "pair_frac": frac[pkeys[:, 1]],
        "row_pairs": row_pairs,
        "row_weights": row_weights,
        "rating": np.array([int(float(r.get("rating", 0))) for r in rows], dtype="int64"),
    }
