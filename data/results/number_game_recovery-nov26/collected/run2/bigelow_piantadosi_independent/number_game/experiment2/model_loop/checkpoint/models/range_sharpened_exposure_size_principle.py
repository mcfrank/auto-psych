"""Range-sharpened, exposure-weakened size principle.

Refinement of `exposure_weakened_size_principle`: the same Bayesian concept
learning over 45 rules (prior by kind of rule, even and odd separate) and all
intervals of 1..100, with sampling strength decaying as beta * (1 + k) ** (-gamma)
over k earlier trials on the set, but intervals and rules get separate sampling
strengths: intervals use beta * exp(delta). delta > 0 means the examples' range
is read more literally (sharper inside/outside-range gap) than rules' extensions.
"Yes" is the posterior probability that the concept contains the target.
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
    rules += [(5, list(range(m, 101, m))) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]]
    rules += [(4, [m**k for k in range(1, 8) if m**k <= 100]) for m in range(2, 11)]
    rules += [(6, [n for n in NUMBERS if n % 10 == d]) for d in range(10)]
    rules += [(7, [n for n in NUMBERS if str(d) in str(n)]) for d in range(10)]
    return rules


FAMILIES = ["odd", "even", "primes", "squares_cubes", "powers", "multiples", "final_digit", "contains_digit"]
_RULES = rule_concepts()
RULE_FAMILY = np.array([f for f, _ in _RULES], dtype="int64")
N_RULES = len(_RULES)
N_FAM = len(FAMILIES)
FAMILY_LOG_COUNT = np.log(np.bincount(RULE_FAMILY, minlength=N_FAM)).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for _, c in _RULES], dtype="float64")
INTERVALS = np.array([(m, n) for m in NUMBERS for n in range(m, 101)])
N_INTERVALS = len(INTERVALS)
INTERVAL_MEMBERSHIP = (NUMBERS[None, :] >= INTERVALS[:, :1]) & (NUMBERS[None, :] <= INTERVALS[:, 1:])
INTERVAL_MEMBERSHIP = INTERVAL_MEMBERSHIP.astype("float64")


def _example_log_lik(membership):
    return np.log(ALPHA * membership / membership.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)


RULE_EX_LL = _example_log_lik(RULE_MEMBERSHIP)  # (rules, 100)
INTERVAL_EX_LL = _example_log_lik(INTERVAL_MEMBERSHIP)  # (intervals, 100)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, 1)))  # (sets, interval groups)
    group_log_size = pm.Data("group_log_size", np.zeros((1, 1)))  # (sets, interval groups)
    unit_set = pm.Data("unit_set", np.zeros(1, dtype="int64"))  # (set, exposure) units -> set
    unit_log_exposure = pm.Data("unit_log_exposure", np.zeros(1))  # log(1 + earlier trials on set)
    cell_rule_in = pm.Data("cell_rule_in", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_frac = pm.Data("cell_frac", np.zeros((1, 1)))  # (cells, interval groups)
    cell_unit = pm.Data("cell_unit", np.zeros(1, dtype="int64"))  # (cells,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (rows,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_logit = pm.Normal("family_logit", mu=0.0, sigma=2.5, shape=N_FAM)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)  # sampling strength at first exposure
    gamma = pm.Normal("gamma", mu=0.0, sigma=0.5)  # decay of sampling strength with exposure
    delta = pm.Normal("delta", mu=0.0, sigma=0.7)  # log ratio of interval to rule sampling strength

    log_norm = pt.logsumexp(pt.concatenate([family_logit, pt.zeros(1)]))
    log_rule_prior = family_logit[RULE_FAMILY] - FAMILY_LOG_COUNT[RULE_FAMILY] - log_norm
    log_interval_prior = -log_norm - np.log(N_INTERVALS)

    beta_unit = (beta * pt.exp(-gamma * unit_log_exposure))[:, None]
    log_rules = log_rule_prior[None, :] + beta_unit * rule_log_lik[unit_set]
    log_groups = log_interval_prior + group_log_size[unit_set] + pt.exp(delta) * beta_unit * group_log_lik[unit_set]
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    post_cell = posterior[cell_unit]
    in_concept = (post_cell[:, :N_RULES] * cell_rule_in).sum(axis=1) + (post_cell[:, N_RULES:] * cell_frac).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _exposures(rows):
    """Earlier trials on the same set in the same session (trial_index restarts per session)."""
    counts, prev, out = {}, None, []
    for r in rows:
        t = r.get("trial_index")
        t = int(float(t)) if t not in (None, "") else None
        if t is not None and prev is not None and t <= prev:
            counts = {}
        prev = t
        s = str(r["number_set"])
        out.append(counts.get(s, 0))
        counts[s] = counts.get(s, 0) + 1
    return out


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8"), `target`, `trial_index` and `rating`.

    Intervals with the same likelihood of a set are grouped per distinct set;
    padding groups get a log size of -1e4 (no posterior mass). Posteriors are
    computed per (set, exposure count) unit, memberships per (unit, target) cell.
    """
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    exposure = _exposures(rows)
    units = [(set_index[s], k) for s, k in zip(sets, exposure)]
    unique_units = list(dict.fromkeys(units))
    unit_index = {u: i for i, u in enumerate(unique_units)}
    cells = [(unit_index[u], int(r["target"])) for u, r in zip(units, rows)]
    unique_cells = list(dict.fromkeys(cells))
    cell_index = {c: i for i, c in enumerate(unique_cells)}

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
    frac = np.zeros((len(unique_cells), n_groups))
    rule_in = np.zeros((len(unique_cells), N_RULES))
    for j, (u, t) in enumerate(unique_cells):
        inverse, keys = groupings[unique_units[u][0]]
        frac[j, : len(keys)] = np.bincount(inverse, weights=INTERVAL_MEMBERSHIP[:, t - 1], minlength=len(keys))
        frac[j] /= sizes[unique_units[u][0]]
        rule_in[j] = RULE_MEMBERSHIP[:, t - 1]
    return {
        "rule_log_lik": rule_ll,
        "group_log_lik": log_lik,
        "group_log_size": log_size,
        "unit_set": np.array([u[0] for u in unique_units], dtype="int64"),
        "unit_log_exposure": np.log1p(np.array([u[1] for u in unique_units], dtype="float64")),
        "cell_rule_in": rule_in,
        "cell_frac": frac,
        "cell_unit": np.array([c[0] for c in unique_cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
