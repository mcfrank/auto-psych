"""Bounded rules: arithmetic rules confined to a stretch of the number line.

People compose concepts: besides a bare arithmetic rule (e.g. even numbers) or
a bare stretch of the number line (e.g. 20 to 40), they also entertain a rule
that the program applies only within a stretch (e.g. even numbers between 20
and 40). These bounded rules form one more kind of concept with its own prior
plausibility, the bounds being any stretch that is equally likely a priori, and
they are weighed by the same Bayesian learning as the effortful_computed_rule_search
model (kind plausibilities, sparse familiarities, a looser sampling strength
for ranges, computed rules searched for only on a share of judgments). A set
that fits a rule but sits in a narrow part of the line (e.g. 22, 26, 30) is
then generalized to the rule's members near the examples, but neither to its
distant members nor to in-span non-members.
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
# Rules that must be computed rather than seen in the digits or on the line.
COMPUTED = np.isin(RULE_FAMILY, [2, 3, 4, 5])
COMPUTED_MASK = np.where(COMPUTED, -1e4, 0.0)
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
    cell_rule_in = pm.Data("cell_rule_in", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_frac = pm.Data("cell_frac", np.zeros((1, 1)))  # (cells, interval groups)
    conj_log_lik = pm.Data("conj_log_lik", np.zeros((1, 1)))  # (sets, bounded-rule groups)
    conj_log_size = pm.Data("conj_log_size", np.full((1, 1), -1e4))  # log count of bounds per group
    conj_rule = pm.Data("conj_rule", np.zeros((1, 1), dtype="int64"))  # rule index per group
    cell_conj_frac = pm.Data("cell_conj_frac", np.zeros((1, 1)))  # (cells, bounded-rule groups)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (rows,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior log-odds of each kind of rule against the intervals as a whole.
    family_logit = pm.Normal("family_logit", mu=0.0, sigma=2.5, shape=N_FAM)
    # Sampling strength of a single example, one for rules and one for ranges
    # (each identified by its own kind of evidence, so they do not trade off).
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)
    beta_range = pm.LogNormal("beta_range", mu=0.0, sigma=0.7)
    pm.Deterministic("range_sampling", beta_range / beta)
    gamma = pm.Normal("gamma", mu=0.0, sigma=0.5)  # capacity limit: loss of evidence per (log) example

    # Familiarity of each rule relative to the other rules of its kind: sparse,
    # heavy-tailed deviations (most near zero, a few large), centred within the
    # kind so that the kind's overall plausibility stays with family_logit.
    familiarity_raw = pm.StudentT("familiarity_z", nu=5.0, mu=0.0, sigma=0.4, shape=len(MULTI_RULES))
    familiarity_z = pt.set_subtensor(pt.zeros(N_RULES)[MULTI_RULES], familiarity_raw)
    kind_mean_z = (pt.dot(KIND_ONEHOT.T, familiarity_z) / FAMILY_COUNT)[RULE_FAMILY]
    log_rule_raw = family_logit[RULE_FAMILY] - FAMILY_LOG_COUNT[RULE_FAMILY] + FAMILIARITY_SPREAD * (familiarity_z - kind_mean_z)
    # Prior log-odds of a bounded rule (any rule, any stretch) against the bare ranges.
    bounded_logit = pm.Normal("bounded_logit", mu=-2.0, sigma=2.0)
    log_norm = pt.logsumexp(pt.concatenate([log_rule_raw, pt.zeros(1), bounded_logit[None]]))
    log_rule_prior = log_rule_raw - log_norm
    log_interval_prior = -log_norm - np.log(N_INTERVALS)

    capacity = pt.exp(-gamma * set_log_n)[:, None]
    log_rules = log_rule_prior[None, :] + beta * capacity * rule_log_lik
    # Ranges are read with their own (looser) sampling strength.
    log_groups = log_interval_prior + group_log_size + beta_range * capacity * group_log_lik
    # A bounded rule's prior: the kind's plausibility, shared among rules as the
    # bare rules share theirs, times a uniformly chosen stretch.
    log_rule_share = log_rule_raw - pt.logsumexp(log_rule_raw)
    log_conj = (bounded_logit - log_norm - np.log(N_INTERVALS) + log_rule_share[conj_rule]
                + conj_log_size + beta * capacity * conj_log_lik)
    log_joint = pt.concatenate([log_rules, log_groups, log_conj], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    # Without the effortful search, computed rules are simply not considered.
    log_joint_glance = pt.concatenate(
        [log_rules + COMPUTED_MASK[None, :], log_groups, log_conj + pt.as_tensor(COMPUTED_MASK)[conj_rule]], axis=1
    )
    posterior_glance = pt.exp(log_joint_glance - pt.logsumexp(log_joint_glance, axis=1, keepdims=True))
    search_prob = pm.Beta("search_prob", alpha=2.0, beta=2.0)

    def endorse(post):
        post_cell = post[cell_set]
        n_bare = N_RULES + cell_frac.shape[1]
        return (
            (post_cell[:, :N_RULES] * cell_rule_in).sum(axis=1)
            + (post_cell[:, N_RULES:n_bare] * cell_frac).sum(axis=1)
            + (post_cell[:, n_bare:] * cell_conj_frac).sum(axis=1)
        )

    in_concept = search_prob * endorse(posterior) + (1.0 - search_prob) * endorse(posterior_glance)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-6, 1 - 1e-6)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`.

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
    # Bounded rules: each rule holding every example, within each stretch that
    # covers them, grouped per set by (rule, number of members in the stretch).
    conj = []
    for s in unique:
        ex = np.array(s)
        lo, hi = ex.min(), ex.max()
        m, n = np.meshgrid(np.arange(1, lo + 1), np.arange(hi, 101), indexing="ij")
        m, n = m.ravel(), n.ravel()
        groups = []  # (rule, k, log_lik, starts, ends)
        for r in np.flatnonzero(RULE_MEMBERSHIP[:, ex - 1].min(axis=1) > 0):
            cum = np.concatenate([[0.0], np.cumsum(RULE_MEMBERSHIP[r])])
            k = (cum[n] - cum[m - 1]).astype(int)
            for kk in np.unique(k):
                sel = k == kk
                ll = len(ex) * np.log(ALPHA / kk + (1 - ALPHA) / 100)
                groups.append((r, ll, m[sel], n[sel]))
        conj.append(groups)
    n_conj = max(1, max(len(g) for g in conj))
    conj_ll = np.zeros((len(unique), n_conj))
    conj_size = np.full((len(unique), n_conj), -1e4)
    conj_rule = np.zeros((len(unique), n_conj), dtype="int64")
    for i, groups in enumerate(conj):
        for j, (r, ll, gm, gn) in enumerate(groups):
            conj_ll[i, j], conj_size[i, j], conj_rule[i, j] = ll, np.log(len(gm)), r
    conj_frac = np.zeros((len(cells), n_conj))
    for c, (s, t) in enumerate(cells):
        for j, (r, ll, gm, gn) in enumerate(conj[set_index[s]]):
            if RULE_MEMBERSHIP[r, t - 1]:
                conj_frac[c, j] = np.mean((gm <= t) & (gn >= t))
    return {
        "conj_log_lik": conj_ll,
        "conj_log_size": conj_size,
        "conj_rule": conj_rule,
        "cell_conj_frac": conj_frac,
        "rule_log_lik": rule_ll,
        "group_log_lik": log_lik,
        "group_log_size": log_size,
        "set_log_n": np.log(np.array([len(s) for s in unique], dtype="float64")),
        "cell_set": np.array([set_index[s] for s, _ in cells], dtype="int64"),
        "cell_rule_in": rule_in,
        "cell_frac": frac,
        "row_cell": np.array([cell_index[(s, int(r["target"]))] for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(float(r.get("rating", 0))) for r in rows], dtype="int64"),
    }
