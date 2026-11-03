"""Equal-prior familiar rules, graded digit sums, round-number range edges.

Refinement of `equal_prior_familiar_rules_digit_sums`, adding one component from
`round_number_range_boundaries`: people do the same Bayesian concept learning
over familiar number rules (odd, even, primes, squares, cubes, powers of m,
multiples of k, "ends in d", all equally plausible a priori, one shared mass),
digit-sum rules (prior share proportional to exp(-decay * s)) and the 5,050
ranges of 1..100 (Erlang width prior, the size principle, 10% of examples
uniform noise). The one change is that multiples of ten act as reference points
when judging how plausible a range is a priori: a range's prior is multiplied
by exp(round_bonus) for each of its two edges that falls exactly on a multiple
of ten, with sign and strength inferred. So "yes" answers driven by magnitude
stop just before, or exactly at, a nearby multiple of ten. (Range outputs are
uniform within the range: the parent's centre-clustering was inferred to be
essentially zero, and dropping it makes the model fast enough to fit.)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


def familiar_rules():
    rules = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "even": list(range(2, 101, 2)),
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    return list(rules.values())


DSUM_RULES = [[n for n in NUMBERS if sum(map(int, str(n))) == t] for t in range(1, 19)]
RULES = familiar_rules() + DSUM_RULES
N_FAMILIAR = len(RULES) - 18
N_RULES = len(RULES)
DSUM_VALUE = np.arange(1, 19, dtype="float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
WIDTH = HI - LO + 1
N_ROUND_EDGES = ((LO % 10) == 0).astype("float64") + ((HI % 10) == 0).astype("float64")
WIDTHS = np.arange(1, 101, dtype="float64")
LOG_IN = np.log(ALPHA / WIDTHS + (1 - ALPHA) / 100)  # per-example log lik inside a width-W range
LOG_OUT = np.log((1 - ALPHA) / 100)
# Ranges sharing a width and a number of round edges share a prior, so their
# likelihood sums are precomputed per set (group = 3 * (width - 1) + round edges).
GROUP = (3 * (WIDTH - 1) + N_ROUND_EDGES).astype("int64")
N_GROUPS = 300
GROUP_COUNT = np.bincount(GROUP, minlength=N_GROUPS).astype("float64")
GROUP_WIDTH = np.repeat(WIDTHS, 3)
GROUP_EDGES = np.tile(np.arange(3.0), 100)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    range_log_total = pm.Data("range_log_total", np.zeros((1, N_GROUPS)))  # (sets, groups)
    range_in_share = pm.Data("range_in_share", np.zeros((1, N_GROUPS, 100)))  # (sets, groups, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of familiar rules (all equally plausible), digit sums, ranges.
    kind_weight = pm.Dirichlet("kind_weight", a=np.ones(3))
    log_w = pt.log(kind_weight)
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    dsum_decay = pm.Normal("dsum_decay", mu=0.0, sigma=0.5)
    round_bonus = pm.Normal("round_bonus", mu=0.0, sigma=1.0)

    # Per-range log prior: Erlang in width plus the round-edge bonus, normalised.
    raw = np.log(GROUP_WIDTH) - GROUP_WIDTH / interval_scale + round_bonus * GROUP_EDGES
    log_per_range = raw - pt.logsumexp(raw + np.log(np.maximum(GROUP_COUNT, 1e-300)))

    dsum_log_share = -dsum_decay * DSUM_VALUE
    dsum_log_share = dsum_log_share - pt.logsumexp(dsum_log_share)
    log_prior_rules = pt.concatenate(
        [pt.zeros(N_FAMILIAR) + log_w[0] - np.log(N_FAMILIAR), log_w[1] + dsum_log_share]
    )
    log_rules = log_prior_rules[None, :] + rule_log_lik
    log_ranges = log_w[2] + log_per_range[None, :] + range_log_total
    log_joint = pt.concatenate([log_rules, log_ranges], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + (
        posterior[:, N_RULES:, None] * range_in_share
    ).sum(axis=1)  # (sets, targets)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = np.array([int(r["target"]) for r in rows])
    unique = list(dict.fromkeys(sets))
    index = np.array([{s: i for i, s in enumerate(unique)}[s] for s in sets])
    total = np.zeros((len(unique), N_GROUPS))
    inside = np.zeros((len(unique), N_GROUPS, 100))
    for i, s in enumerate(unique):
        ll = np.zeros(len(LO))
        for x in s:
            ll += np.where((LO <= x) & (x <= HI), LOG_IN[WIDTH - 1], LOG_OUT)
        top = ll.max()
        e = np.exp(ll - top)
        tot = np.bincount(GROUP, weights=e, minlength=N_GROUPS)
        total[i] = top + np.log(np.maximum(tot, 1e-300))
        for t in range(1, 101):
            hit = (LO <= t) & (t <= HI)
            inside[i, :, t - 1] = np.bincount(GROUP[hit], weights=e[hit], minlength=N_GROUPS) / np.maximum(tot, 1e-300)
    rule_ll = np.stack([RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    return {
        "rule_log_lik": rule_ll,
        "range_log_total": total,
        "range_in_share": inside,
        "row_cell": (index * 100 + targets - 1).astype("int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
