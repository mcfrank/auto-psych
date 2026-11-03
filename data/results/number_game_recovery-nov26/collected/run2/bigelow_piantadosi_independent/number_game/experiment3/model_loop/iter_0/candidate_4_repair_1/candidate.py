"""Individual rule-versus-range leaning, on the parity-veto sparse-familiarity learner.

Refinement of parity_veto_sparse_familiar_rules. People infer the hidden concept
by capacity-limited Bayesian learning over 45 rules (kinds with their own
plausibility, sparse heavy-tailed rule familiarities within kinds, multiples of
5 its own base-ten kind) and all intervals of 1-100, and a target whose parity
matches none of the examples is vetoed with a "no" on a fraction of trials.
The single change: people differ in how readily they reach for an arithmetic
rule rather than a stretch of the number line. Each participant carries one
personal log-odds shift added to the prior of every rule against the ranges
(a fixed spread, summing to zero over participants), so the same set is generalized by
its rule (distant rule members get yeses) by some people and by its range
(numerically close non-members get yeses) by others.

Participants are not labelled in the data: a new participant starts wherever
trial_index fails to increase. Rows without a trial_index (bare stimuli) and
participants beyond N_PERSON_SLOTS get the population-average learner.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
LEANING_SPREAD = 1.0  # how much participants differ in rule-vs-range prior log odds, fixed
N_PERSON_SLOTS = 15  # participants with their own leaning; later ones get the population mean
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

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    group_log_lik = pm.Data("group_log_lik", np.zeros((1, 1)))  # (sets, interval groups)
    group_log_size = pm.Data("group_log_size", np.zeros((1, 1)))  # (sets, interval groups)
    set_log_n = pm.Data("set_log_n", np.zeros(1))  # log(number of examples) per set
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,) distinct (set, target)
    row_person = pm.Data("row_person", np.zeros(1, dtype="int64"))  # (rows,) participant slot
    cell_rule_in = pm.Data("cell_rule_in", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_frac = pm.Data("cell_frac", np.zeros((1, 1)))  # (cells, interval groups)
    cell_mismatch = pm.Data("cell_mismatch", np.zeros(1))  # (cells,) target parity matches no example
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (rows,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior log-odds of each kind of rule against the intervals as a whole.
    family_logit = pm.Normal("family_logit", mu=0.0, sigma=2.5, shape=N_FAM)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)  # sampling strength of a single example
    gamma = pm.Normal("gamma", mu=0.0, sigma=0.5)  # capacity limit: loss of evidence per (log) example

    # Familiarity of each rule relative to the other rules of its kind: sparse,
    # heavy-tailed deviations (most near zero, a few large), centred within the
    # kind so that the kind's overall plausibility stays with family_logit.
    familiarity_raw = pm.StudentT("familiarity_z", nu=5.0, mu=0.0, sigma=0.4, shape=len(MULTI_RULES))
    familiarity_z = pt.set_subtensor(pt.zeros(N_RULES)[MULTI_RULES], familiarity_raw)
    kind_mean_z = (pt.dot(KIND_ONEHOT.T, familiarity_z) / FAMILY_COUNT)[RULE_FAMILY]
    log_rule_raw = family_logit[RULE_FAMILY] - FAMILY_LOG_COUNT[RULE_FAMILY] + FAMILIARITY_SPREAD * (familiarity_z - kind_mean_z)
    log_norm = pt.logsumexp(pt.concatenate([log_rule_raw, pt.zeros(1)]))
    log_rule_prior = log_rule_raw - log_norm
    log_interval_prior = -log_norm - np.log(N_INTERVALS)

    beta_set = (beta * pt.exp(-gamma * set_log_n))[:, None]
    log_rules = log_rule_prior[None, :] + beta_set * rule_log_lik
    log_groups = log_interval_prior + group_log_size + beta_set * group_log_lik
    # Belief within the rules and within the ranges, and the population's
    # log odds of rules against ranges, per set.
    log_rule_mass = pt.logsumexp(log_rules, axis=1, keepdims=True)
    log_range_mass = pt.logsumexp(log_groups, axis=1, keepdims=True)
    within_rules = pt.exp(log_rules - log_rule_mass)[cell_set]
    within_ranges = pt.exp(log_groups - log_range_mass)[cell_set]
    rule_in = (within_rules * cell_rule_in).sum(axis=1)
    range_in = (within_ranges * cell_frac).sum(axis=1)
    rule_odds = (log_rule_mass - log_range_mass)[:, 0][cell_set]

    # Each participant's leaning towards rules over ranges (a shift of the prior
    # log odds of every rule against the ranges); the last slot is the
    # population-average learner. Shifting all rules together only moves the
    # split of belief between rules and ranges, so it acts on rule_odds.
    # Leanings sum to zero across participants (the population's level stays
    # with family_logit) and their spread is fixed, so they do not trade off.
    leaning_z = pm.ZeroSumNormal("leaning", sigma=LEANING_SPREAD, shape=N_PERSON_SLOTS)
    leaning = pt.concatenate([leaning_z, pt.zeros(1)])[row_person]
    w_rule = pm.math.sigmoid(rule_odds[row_cell] + leaning)
    in_concept = w_rule * rule_in[row_cell] + (1 - w_rule) * range_in[row_cell]

    # Fraction of parity-conflict trials on which the conflict vetoes a "yes".
    veto = pm.Beta("veto", 2.0, 6.0)
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept * (1 - veto * cell_mismatch[row_cell]), 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target`, `rating` and
    `trial_index` (a participant's session restarts where it fails to increase).

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
            persons.append(N_PERSON_SLOTS)
            continue
        t = int(float(t))
        if prev is not None and t <= prev:
            pid += 1
        prev = t
        persons.append(min(pid, N_PERSON_SLOTS))
    return {
        "row_person": np.array(persons, dtype="int64"),
        "rule_log_lik": rule_ll,
        "group_log_lik": log_lik,
        "group_log_size": log_size,
        "set_log_n": np.log(np.array([len(s) for s in unique], dtype="float64")),
        "cell_set": np.array([set_index[s] for s, _ in cells], dtype="int64"),
        "cell_rule_in": rule_in,
        "cell_frac": frac,
        "cell_mismatch": np.array([float(all((t - x) % 2 for x in s)) for s, t in cells]),
        "row_cell": np.array([cell_index[(s, int(r["target"]))] for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(float(r.get("rating", 0))) for r in rows], dtype="int64"),
    }
