"""Familiarity-relaxed size principle.

People do the same recency-weighted Bayesian concept learning as
`recency_weighted_even_odd_family_prior` (45 rules with a prior over kinds of
rule, even and odd separate; all intervals of 1..100; tempered size principle;
later examples weigh more), but the strength of the size principle depends on
how familiar the set is: on the k-th prior encounter with the same set (within
a participant's session) the sampling strength is beta * (1 + k) ** (-gamma).
On a first encounter (k = 0) the examples are read as tightly sampled; with
gamma > 0 each repeat discounts that evidence, broadening generalization and
raising the yes rate on repeats. A bare stimulus (no session history) counts
as a first encounter. "Yes" is the posterior probability that the concept
contains the target.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
MAX_EX = 4
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
    rule_pos_ll = pm.Data("rule_pos_ll", np.zeros((1, MAX_EX, N_RULES)))  # (sets, positions, rules)
    group_pos_ll = pm.Data("group_pos_ll", np.zeros((1, MAX_EX, 1)))  # (sets, positions, groups)
    group_log_size = pm.Data("group_log_size", np.zeros((1, 1)))  # (sets, groups)
    pos_back = pm.Data("pos_back", np.zeros((1, MAX_EX)))  # examples shown after this one
    pos_mask = pm.Data("pos_mask", np.ones((1, MAX_EX)))  # 1 for real examples
    state_set = pm.Data("state_set", np.zeros(1, dtype="int64"))  # set of each (set, exposure) state
    state_expo = pm.Data("state_expo", np.zeros(1))  # log(1 + prior encounters)
    cell_state = pm.Data("cell_state", np.zeros(1, dtype="int64"))
    cell_rule_in = pm.Data("cell_rule_in", np.zeros((1, N_RULES)))
    cell_frac = pm.Data("cell_frac", np.zeros((1, 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_logit = pm.Normal("family_logit", mu=0.0, sigma=2.5, shape=N_FAM)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)  # sampling strength on first encounter
    log_rho = pm.Normal("log_rho", mu=0.0, sigma=1.0)  # order gradient (< 0: recency)
    gamma = pm.Normal("gamma", mu=0.0, sigma=0.5)  # familiarity discount (> 0: repeats broaden)

    raw_w = pt.exp(log_rho * pos_back) * pos_mask
    w = raw_w * pos_mask.sum(axis=1, keepdims=True) / raw_w.sum(axis=1, keepdims=True)  # mean 1
    rule_ll = (w[:, :, None] * rule_pos_ll).sum(axis=1)
    group_ll = (w[:, :, None] * group_pos_ll).sum(axis=1)

    log_norm = pt.logsumexp(pt.concatenate([family_logit, pt.zeros(1)]))
    log_rule_prior = family_logit[RULE_FAMILY] - FAMILY_LOG_COUNT[RULE_FAMILY] - log_norm
    log_interval_prior = -log_norm - np.log(N_INTERVALS)

    beta_state = (beta * pt.exp(-gamma * state_expo))[:, None]
    log_rules = log_rule_prior[None, :] + beta_state * rule_ll[state_set]
    log_groups = log_interval_prior + group_log_size[state_set] + beta_state * group_ll[state_set]
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    post_cell = posterior[cell_state]
    in_concept = (post_cell[:, :N_RULES] * cell_rule_in).sum(axis=1) + (post_cell[:, N_RULES:] * cell_frac).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _prior_encounters(rows, sets):
    """Per row, how many earlier trials of the same participant showed the same set.

    A participant's session restarts when trial_index does not increase; rows
    without a trial_index (bare stimuli) count as first encounters.
    """
    counts, out, prev = {}, [], None
    for r, s in zip(rows, sets):
        t = r.get("trial_index")
        if t is None or str(t) == "":
            out.append(0)
            continue
        t = int(float(t))
        if prev is None or t <= prev:
            counts = {}
        prev = t
        out.append(counts.get(s, 0))
        counts[s] = counts.get(s, 0) + 1
    return out


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target`, `trial_index`, `rating`.

    Intervals with the same per-position likelihoods of a set have the same
    posterior at every parameter value, so they are grouped per distinct set;
    padding groups get a log size of -1e4 (no posterior mass), padding
    positions get zero likelihood and zero weight. Posteriors are computed per
    (set, prior-encounter count) state.
    """
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    expo = _prior_encounters(rows, sets)
    unique = list(dict.fromkeys(sets))
    set_index = {s: i for i, s in enumerate(unique)}
    states = [(set_index[s], k) for s, k in zip(sets, expo)]
    unique_states = list(dict.fromkeys(states))
    state_index = {st: i for i, st in enumerate(unique_states)}
    cells = [(state_index[st], int(r["target"])) for st, r in zip(states, rows)]
    unique_cells = list(dict.fromkeys(cells))
    cell_index = {c: i for i, c in enumerate(unique_cells)}
    n_sets = len(unique)

    rule_pos = np.zeros((n_sets, MAX_EX, N_RULES))
    back = np.zeros((n_sets, MAX_EX))
    mask = np.zeros((n_sets, MAX_EX))
    groupings = []
    for i, s in enumerate(unique):
        idx = np.array(s) - 1
        n = len(s)
        rule_pos[i, :n] = RULE_EX_LL[:, idx].T
        back[i, :n] = np.arange(n - 1, -1, -1)
        mask[i, :n] = 1.0
        ll = INTERVAL_EX_LL[:, idx]  # (intervals, n)
        keys, inverse = np.unique(np.round(ll, 9), axis=0, return_inverse=True)
        groupings.append((inverse.reshape(-1), keys))
    n_groups = max(len(k) for _, k in groupings)
    group_pos = np.zeros((n_sets, MAX_EX, n_groups))
    log_size = np.full((n_sets, n_groups), -1e4)
    sizes = np.ones((n_sets, n_groups))
    for i, (inverse, keys) in enumerate(groupings):
        counts = np.bincount(inverse, minlength=len(keys))
        group_pos[i, : keys.shape[1], : len(keys)] = keys.T
        log_size[i, : len(keys)], sizes[i, : len(keys)] = np.log(counts), counts
    frac = np.zeros((len(unique_cells), n_groups))
    rule_in = np.zeros((len(unique_cells), N_RULES))
    for j, (si, t) in enumerate(unique_cells):
        i = unique_states[si][0]
        inverse, keys = groupings[i]
        frac[j, : len(keys)] = np.bincount(inverse, weights=INTERVAL_MEMBERSHIP[:, t - 1], minlength=len(keys))
        frac[j] /= sizes[i]
        rule_in[j] = RULE_MEMBERSHIP[:, t - 1]
    return {
        "rule_pos_ll": rule_pos,
        "group_pos_ll": group_pos,
        "group_log_size": log_size,
        "pos_back": back,
        "pos_mask": mask,
        "state_set": np.array([st[0] for st in unique_states], dtype="int64"),
        "state_expo": np.log1p(np.array([st[1] for st in unique_states], dtype="float64")),
        "cell_state": np.array([c[0] for c in unique_cells], dtype="int64"),
        "cell_rule_in": rule_in,
        "cell_frac": frac,
        "row_cell": np.array([cell_index[c] for c in cells], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
