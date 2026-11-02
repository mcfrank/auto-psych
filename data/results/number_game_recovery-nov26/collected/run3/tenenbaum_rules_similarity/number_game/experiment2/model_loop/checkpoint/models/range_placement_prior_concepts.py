"""Refinement of beta_share_width_exception_concepts: the same Bayesian concept
learning over 45 rules (family-organised prior, exception rules as their own
family) and all intervals of 1..100, with the size principle (examples come from
the concept with probability 0.9) and a beta prior over the share of the line a
range covers. The single change is where people expect a range to sit: instead
of every placement of a range of given width being equally likely, they hold a
beta prior over the position of the range's centre on the 1..100 line, with its
mean position and concentration inferred. So ranges among small (or large, or
middling) numbers are a priori more plausible than equally wide ranges elsewhere,
and examples in a disfavoured region are explained by a rule more readily and
generalise to fewer neighbouring numbers.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


def rule_concepts():
    rules = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
        "powers of 2, not 32": [2, 4, 8, 16, 64],
        "powers of 2, or 37": [2, 4, 8, 16, 32, 37, 64],
        "even": list(range(2, 101, 2)),
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(10)})
    return rules


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
BOUNDS = [(m, n) for m in NUMBERS for n in range(m, 101)]
INTERVALS = [list(range(m, n + 1)) for m, n in BOUNDS]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
SIZES = np.arange(1, 101, dtype="float64")
SHARE = (SIZES - 0.5) / 100.0  # share of the 1..100 line covered, in (0, 1)
# A range's placement is its centre, one of 199 half-unit positions on the line.
CENTRE_2X = np.array([m + n for m, n in BOUNDS])  # 2..200
POSITION = (np.arange(2, 201) / 2.0 - 0.5) / 100.0  # centre position as a share, in (0, 1)
# Ranges are grouped by size and by which twentieth of the line their centre
# lies in; within a group the placement prior is spread evenly (all
# parameter-free aggregation then happens once per set in prepare_observed).
N_BINS = 20
SIZE_OF = np.array([n - m for m, n in BOUNDS])  # L - 1
BIN_OF = np.minimum((POSITION[CENTRE_2X - 2] * N_BINS).astype(int), N_BINS - 1)
_GROUP_KEYS = sorted(set(zip(SIZE_OF.tolist(), BIN_OF.tolist())))
_GROUP_INDEX = {k: i for i, k in enumerate(_GROUP_KEYS)}
GROUP_OF = np.array([_GROUP_INDEX[k] for k in zip(SIZE_OF.tolist(), BIN_OF.tolist())])
N_GROUPS = len(_GROUP_KEYS)
GROUP_SIZE_IDX = np.array([k[0] for k in _GROUP_KEYS])
GROUP_COUNT = np.bincount(GROUP_OF).astype("float64")
GROUP_POSITIONS = np.zeros((N_GROUPS, 199))  # how many ranges of the group sit at each centre
np.add.at(GROUP_POSITIONS, (GROUP_OF, CENTRE_2X - 2), 1.0)
SIZE_GROUPS = (GROUP_SIZE_IDX[None, :] == np.arange(100)[:, None]).astype("float64")  # (sizes, groups)


def rule_family(name):
    if name in ("odd", "even"):
        return 0
    if name == "primes":
        return 1
    if name in ("squares", "cubes"):
        return 2
    if name.startswith("powers of 2,"):
        return 7
    if name.startswith("powers"):
        return 3
    if name.startswith("multiples"):
        return 4
    if name.startswith("ends in"):
        return 5
    return 6


FAMILY = np.array([rule_family(n) for n in RULE_NAMES])
N_FAMILIES = FAMILY.max() + 1
FAMILY_SIZE = np.bincount(FAMILY)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    group_log_evidence = pm.Data("group_log_evidence", np.zeros((1, N_GROUPS)))  # log mean P(D|j) in group
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_group_member = pm.Data("cell_group_member", np.zeros((1, N_GROUPS)))  # (cells, groups)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    mean_share = pm.Beta("mean_share", 2, 3)  # mean share of 1..100 a range covers
    concentration = pm.LogNormal("concentration", mu=np.log(5.0), sigma=1.0)
    mean_position = pm.Beta("mean_position", 2, 2)  # where on the line a range is expected to sit
    position_concentration = pm.LogNormal("position_concentration", mu=np.log(3.0), sigma=1.0)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    rule_log_prior = pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])

    log_size_prior = ((mean_share * concentration - 1.0) * np.log(SHARE)
                      + ((1.0 - mean_share) * concentration - 1.0) * np.log1p(-SHARE))
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior)

    # Placement of a range of a given size: beta density over its centre,
    # normalised among the placements available to that size.
    log_place = ((mean_position * position_concentration - 1.0) * np.log(POSITION)
                 + ((1.0 - mean_position) * position_concentration - 1.0) * np.log1p(-POSITION))
    place = pt.exp(log_place - pt.max(log_place))
    group_mass = pt.dot(GROUP_POSITIONS, place)  # (groups,)
    size_mass = pt.dot(SIZE_GROUPS, group_mass)  # (sizes,)
    log_group_prior = (log_size_prior[GROUP_SIZE_IDX] + pt.log(group_mass)
                       - pt.log(size_mass)[GROUP_SIZE_IDX])

    log_rules = pt.log(lambda_0) + rule_log_prior[None, :] + rule_log_lik
    log_groups = pt.log1p(-lambda_0) + log_group_prior[None, :] + group_log_evidence
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_post = posterior[cell_set]
    in_concept = ((cell_post[:, :N_RULES] * cell_rule_member).sum(axis=1)
                  + (cell_post[:, N_RULES:] * cell_group_member).sum(axis=1))

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    n_sets = len(unique)
    group_ev = np.zeros((n_sets, N_GROUPS))
    group_member = np.zeros((n_sets, N_GROUPS, 100))
    for g in range(N_GROUPS):
        sel = GROUP_OF == g
        ll = set_log_lik[:, N_RULES:][:, sel]
        ev = logsumexp(ll, axis=1)
        group_ev[:, g] = ev - np.log(GROUP_COUNT[g])
        group_member[:, g] = np.exp(ll - ev[:, None]) @ INTERVAL_MEMBERSHIP[sel]
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_s = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "group_log_evidence": group_ev,
        "cell_set": cell_s,
        "cell_rule_member": MEMBERSHIP[:N_RULES, cell_t].T.copy(),
        "cell_group_member": np.clip(group_member[cell_s, :, cell_t], 0.0, 1.0),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
