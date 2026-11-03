"""Decision-stage parity conflict on top of base-ten capacity-limited concept learning.

People infer the hidden concept by capacity-limited Bayesian learning over 45
rules (multiples of 5 as its own base-ten-salient kind of rule) and all
intervals of 1-100, with the summed example log-likelihood scaled by
beta * n ** (-gamma). Parity is registered automatically, so when the target's
parity matches none of the examples, the conflict intrudes at the decision and,
on a fraction `veto` of such trials, overrides the inferred concept with a
"no": p_yes = P(target in concept) * (1 - veto * conflict). In all-odd or
all-even sets, opposite-parity targets covered by the concept (e.g. odd numbers
inside the span of 20, 32, 92) get fewer yeses; same-parity targets are
unaffected.
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
    set_log_n = pm.Data("set_log_n", np.zeros(1))  # log(number of examples) per set
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))  # (rows,)
    row_rule_in = pm.Data("row_rule_in", np.zeros((1, N_RULES)))  # (rows, rules)
    row_frac = pm.Data("row_frac", np.zeros((1, 1)))  # (rows, interval groups)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    row_mismatch = pm.Data("row_mismatch", np.zeros(1))  # target's parity matches no example
    # Fraction of parity-conflict trials on which the conflict overrides the concept.
    veto = pm.Beta("veto", 1.0, 3.0)

    # Prior log-odds of each kind of rule against the intervals as a whole.
    family_logit = pm.Normal("family_logit", mu=0.0, sigma=2.5, shape=N_FAM)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.7)  # sampling strength of a single example
    gamma = pm.Normal("gamma", mu=0.0, sigma=0.5)  # capacity limit: loss of evidence per (log) example

    log_norm = pt.logsumexp(pt.concatenate([family_logit, pt.zeros(1)]))
    log_rule_prior = family_logit[RULE_FAMILY] - FAMILY_LOG_COUNT[RULE_FAMILY] - log_norm
    log_interval_prior = -log_norm - np.log(N_INTERVALS)

    beta_set = (beta * pt.exp(-gamma * set_log_n))[:, None]
    log_rules = log_rule_prior[None, :] + beta_set * rule_log_lik
    log_groups = log_interval_prior + group_log_size + beta_set * group_log_lik
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    post_row = posterior[row_set]
    in_concept = (post_row[:, :N_RULES] * row_rule_in).sum(axis=1) + (post_row[:, N_RULES:] * row_frac).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept * (1 - veto * row_mismatch), 1e-6, 1 - 1e-6))
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
    frac = np.zeros((len(rows), n_groups))
    rule_in = np.zeros((len(rows), N_RULES))
    for j, (s, r) in enumerate(zip(sets, rows)):
        i, t = set_index[s], int(r["target"])
        inverse, keys = groupings[i]
        frac[j, : len(keys)] = np.bincount(inverse, weights=INTERVAL_MEMBERSHIP[:, t - 1], minlength=len(keys))
        frac[j] /= sizes[i]
        rule_in[j] = RULE_MEMBERSHIP[:, t - 1]
    return {
        "rule_log_lik": rule_ll,
        "group_log_lik": log_lik,
        "group_log_size": log_size,
        "set_log_n": np.log(np.array([len(s) for s in unique], dtype="float64")),
        "row_set": np.array([set_index[s] for s in sets], dtype="int64"),
        "row_rule_in": rule_in,
        "row_frac": frac,
        "row_mismatch": np.array([float(all(int(r["target"]) % 2 != x % 2 for x in s)) for s, r in zip(sets, rows)]),
        "rating": np.array([int(float(r.get("rating", 0))) for r in rows], dtype="int64"),
    }
