"""Sticky running guess: examples revise a held concept only when they contradict it.

People read the examples in the order shown and hold a running guess about
the concept (one of 45 rules, with kind-level plausibilities and rule-specific
familiarities as in rule_familiarity_within_kinds, or an interval of 1-100).
The first example's guess is drawn from the capacity-limited Bayesian
posterior given that example. When the next example arrives, a guess that
contains it is kept, with a fitted probability kappa (stickiness); otherwise
(the example contradicts the guess, or the learner reconsiders) the guess is
redrawn from the Bayesian posterior given all examples seen so far. The
answer is the probability that the final guess contains the target. kappa = 0
is the order-blind Bayesian learner; kappa > 0 lets guesses formed from early
examples outlive the evidence of later ones, so the same examples in a
different order can be generalized differently.
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
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for _, c in _RULES], dtype="float64")
INTERVALS = np.array([(m, n) for m in NUMBERS for n in range(m, 101)])
N_INTERVALS = len(INTERVALS)
INTERVAL_MEMBERSHIP = (NUMBERS[None, :] >= INTERVALS[:, :1]) & (NUMBERS[None, :] <= INTERVALS[:, 1:])
INTERVAL_MEMBERSHIP = INTERVAL_MEMBERSHIP.astype("float64")


def _example_log_lik(membership):
    return np.log(ALPHA * membership / membership.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)


RULE_EX_LL = _example_log_lik(RULE_MEMBERSHIP)  # (rules, 100)
INTERVAL_EX_LL = _example_log_lik(INTERVAL_MEMBERSHIP)  # (intervals, 100)

SIZES = (1, 2, 3, 4)

with pm.Model() as model:
    # Per set size n: example log-likelihoods and memberships by position,
    # (sets of size n, n, hypotheses), and the log sizes of interval groups.
    data = {}
    for n in SIZES:
        data[n] = (
            pm.Data(f"rule_log_lik_{n}", np.zeros((1, n, N_RULES))),
            pm.Data(f"group_log_lik_{n}", np.zeros((1, n, 1))),
            pm.Data(f"rule_member_{n}", np.zeros((1, n, N_RULES))),
            pm.Data(f"group_member_{n}", np.zeros((1, n, 1))),
            pm.Data(f"group_log_size_{n}", np.zeros((1, 1))),
        )
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # index into sets ordered by size
    cell_rule_in = pm.Data("cell_rule_in", np.zeros((1, N_RULES)))
    cell_frac = pm.Data("cell_frac", np.zeros((1, 1)))  # (cells, max interval groups)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_logit = pm.Normal("family_logit", mu=0.0, sigma=2.5, shape=N_FAM)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)  # sampling strength of a single example
    gamma = pm.Normal("gamma", mu=0.0, sigma=0.5)  # capacity limit: loss of evidence per (log) example
    # Probability of keeping a running guess that the new example fits.
    kappa = pm.Beta("kappa", 2.0, 2.0)
    familiarity_z = pm.Normal("familiarity_z", mu=0.0, sigma=1.0, shape=N_RULES)
    kind_mean_z = (pt.dot(KIND_ONEHOT.T, familiarity_z) / FAMILY_COUNT)[RULE_FAMILY]
    log_rule_raw = family_logit[RULE_FAMILY] - FAMILY_LOG_COUNT[RULE_FAMILY] + FAMILIARITY_SPREAD * (familiarity_z - kind_mean_z)
    log_norm = pt.logsumexp(pt.concatenate([log_rule_raw, pt.zeros(1)]))
    log_rule_prior = log_rule_raw - log_norm
    log_interval_prior = -log_norm - np.log(N_INTERVALS)

    n_groups_max = cell_frac.shape[1]
    posteriors = []
    for n in SIZES:
        r_ll, g_ll, r_in, g_in, g_size = data[n]
        cum_r = pt.cumsum(r_ll, axis=1)  # evidence of each prefix of the examples
        cum_g = pt.cumsum(g_ll, axis=1)
        beta_k = (beta * np.arange(1.0, n + 1) ** (-gamma))[None, :, None]
        lj = pt.concatenate(
            [log_rule_prior[None, None, :] + beta_k * cum_r, log_interval_prior + g_size[:, None, :] + beta_k * cum_g], axis=2
        )
        post_k = pt.exp(lj - pt.logsumexp(lj, axis=2, keepdims=True))  # Bayesian posterior after each prefix
        member = pt.concatenate([r_in, g_in], axis=2)
        # Distribution of the running guess, revised example by example: a
        # guess the new example fits is kept with probability kappa, otherwise
        # it is redrawn from the posterior given all examples so far.
        q = post_k[:, 0, :]
        for j in range(1, n):
            kept = q * kappa * member[:, j, :]
            q = kept + (1.0 - kept.sum(axis=1, keepdims=True)) * post_k[:, j, :]
        posteriors.append(pt.concatenate([q, pt.zeros((q.shape[0], N_RULES + n_groups_max - q.shape[1]))], axis=1))
    posterior = pt.concatenate(posteriors, axis=0)
    post_cell = posterior[cell_set]
    in_concept = (post_cell[:, :N_RULES] * cell_rule_in).sum(axis=1) + (post_cell[:, N_RULES:] * cell_frac).sum(axis=1)
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-6, 1 - 1e-6)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`.

    Sets are split by size. Within a set, intervals with the same likelihood
    for every example share a running-guess probability and are grouped;
    padding groups get a log size of -1e4 (no posterior mass).
    """
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = sorted(dict.fromkeys(sets), key=len)  # stable: ordered by size
    set_index = {s: i for i, s in enumerate(unique)}
    stray = np.log((1 - ALPHA) / 100) + 1e-6
    groupings = []
    for s in unique:
        keys, inverse = np.unique(np.round(INTERVAL_EX_LL[:, np.array(s) - 1], 9), axis=0, return_inverse=True)
        groupings.append((inverse.reshape(-1), keys))
    out = {}
    for n in SIZES:
        idx = [i for i, s in enumerate(unique) if len(s) == n]
        g_max = max([len(groupings[i][1]) for i in idx], default=1)
        r_ll = np.zeros((len(idx), n, N_RULES))
        g_ll = np.zeros((len(idx), n, g_max))
        g_size = np.full((len(idx), g_max), -1e4)
        for a, i in enumerate(idx):
            inverse, keys = groupings[i]
            r_ll[a] = RULE_EX_LL[:, np.array(unique[i]) - 1].T
            g_ll[a, :, : len(keys)] = keys.T
            g_size[a, : len(keys)] = np.log(np.bincount(inverse, minlength=len(keys)))
        out.update({
            f"rule_log_lik_{n}": r_ll, f"group_log_lik_{n}": g_ll,
            f"rule_member_{n}": (r_ll > stray).astype("float64"), f"group_member_{n}": (g_ll > stray).astype("float64"),
            f"group_log_size_{n}": g_size,
        })
    n_groups = max(len(k) for _, k in groupings)
    cells = list(dict.fromkeys((s, int(r["target"])) for s, r in zip(sets, rows)))
    cell_index = {c: k for k, c in enumerate(cells)}
    frac = np.zeros((len(cells), n_groups))
    rule_in = np.zeros((len(cells), N_RULES))
    for k, (s, t) in enumerate(cells):
        inverse, keys = groupings[set_index[s]]
        counts = np.bincount(inverse, minlength=len(keys))
        frac[k, : len(keys)] = np.bincount(inverse, weights=INTERVAL_MEMBERSHIP[:, t - 1], minlength=len(keys)) / counts
        rule_in[k] = RULE_MEMBERSHIP[:, t - 1]
    out.update({
        "cell_set": np.array([set_index[s] for s, _ in cells], dtype="int64"),
        "cell_rule_in": rule_in,
        "cell_frac": frac,
        "row_cell": np.array([cell_index[(s, int(r["target"]))] for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(float(r.get("rating", 0))) for r in rows], dtype="int64"),
    })
    return out
