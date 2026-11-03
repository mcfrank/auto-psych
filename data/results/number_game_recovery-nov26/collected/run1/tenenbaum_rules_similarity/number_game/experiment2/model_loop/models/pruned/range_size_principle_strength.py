"""Refinement of `threshold_concepts_erlang_interval`: a learned strength of the
size principle for magnitude concepts only.

People do the same Bayesian concept learning over 45 rules, bounded ranges
[m, n] of 1..100 (Erlang prior s * exp(-s / sigma) on width) and one-sided
thresholds ([1, n], [m, 100]; own family, uniform prior), with a prior mass per
concept family and 10% of examples drawn at random from 1..100. The one change:
for ranges and thresholds an in-concept example has probability
alpha * width^-gamma * 100^(gamma - 1) + (1 - alpha) / 100, with gamma learned
(gamma = 1 is the incumbent's strong sampling, gamma = 0 weak sampling), while
rules keep the ordinary size principle.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
ALPHA = 0.9
MAX_K = 4


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


def family(name):
    if name in ("odd", "even"):
        return 0
    if name.startswith("multiples"):
        return 2
    if name.startswith("ends"):
        return 3
    if name.startswith("contains"):
        return 4
    if name.startswith("powers of 2,"):
        return 5  # gerrymandered exception rules
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
RULES = list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in _RULES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
# Bounded ranges [m, n] with 1 < m <= n < 100, and one-sided thresholds [1, n] or [m, 100].
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
THRESHOLD = (LO == 1) | (HI == 100)
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
INTERVAL_WIDTH = HI - LO + 1
WIDTHS = np.arange(1, 99)  # widths a bounded range can have
N_PER_WIDTH = np.array([np.sum(~THRESHOLD & (INTERVAL_WIDTH == w)) for w in WIDTHS], dtype="float64")
N_THRESHOLD = float(THRESHOLD.sum())
# Magnitude concepts are grouped by (bounded or threshold, width, examples inside):
# within a group every concept has the same prior and the same likelihood.
GROUP_THRESHOLD = np.repeat([0, 1], 100 * (MAX_K + 1))
GROUP_WIDTH = np.tile(np.repeat(np.arange(1, 101), MAX_K + 1), 2).astype("float64")
GROUP_N_IN = np.tile(np.arange(MAX_K + 1), 200).astype("float64")
N_GROUPS = len(GROUP_WIDTH)
LOG_OUT = np.log((1 - ALPHA) / 100)
EMPTY = -1e4  # log count of a group no range falls into

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    n_examples = pm.Data("n_examples", np.ones((1, 1)), shape=(None, 1))
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))
    # Per set, the targets asked about it (zero-padded): each target's membership of
    # every rule and the share of each magnitude group's ranges that contain it.
    cell_member = pm.Data("cell_member", np.zeros((1, N_RULES + N_GROUPS, 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the six rule families, bounded ranges and one-sided thresholds.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(8))
    log_w = pt.log(family_weight)
    # Characteristic width of the Erlang prior over bounded ranges.
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)
    # Strength of the size principle for ranges and thresholds (1 = incumbent).
    range_size_strength = pm.Normal("range_size_strength", mu=1.0, sigma=0.5)

    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    log_per_range = erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH))
    log_per_range = pt.concatenate([log_per_range, pt.zeros(2)])  # widths 99, 100 are thresholds only
    group_prior = pt.switch(
        GROUP_THRESHOLD == 1,
        log_w[7] - np.log(N_THRESHOLD),
        log_w[6] + log_per_range[(GROUP_WIDTH - 1).astype("int64")],
    )
    log_in = pt.log(
        ALPHA * pt.exp(-range_size_strength * np.log(GROUP_WIDTH) + (range_size_strength - 1) * np.log(100.0))
        + (1 - ALPHA) / 100
    )
    group_log_lik = GROUP_N_IN[None, :] * log_in[None, :] + (n_examples - GROUP_N_IN[None, :]) * LOG_OUT

    log_rules = (log_w[:6] - np.log(FAMILY_SIZE))[RULE_FAMILY][None, :] + rule_log_lik
    log_groups = group_prior[None, :] + group_log_count + group_log_lik
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = pt.matmul(posterior[:, None, :], cell_member)[:, 0, :]  # (sets, targets asked)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    group_of = THRESHOLD.astype("int64") * 100 * (MAX_K + 1) + (INTERVAL_WIDTH - 1) * (MAX_K + 1)
    log_count = np.full((len(unique), N_GROUPS), EMPTY)
    in_concept = np.zeros((len(unique), N_GROUPS, 100))
    for i, s in enumerate(unique):
        n_in = INTERVAL_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1).astype("int64")
        g = group_of + n_in
        count = np.bincount(g, minlength=N_GROUPS).astype("float64")
        sums = np.zeros((N_GROUPS, 100))
        np.add.at(sums, g, INTERVAL_MEMBERSHIP)
        used = count > 0
        log_count[i, used] = np.log(count[used])
        in_concept[i, used] = sums[used] / count[used, None]
    targets = [int(r["target"]) - 1 for r in rows]
    asked = [list(dict.fromkeys(t for s2, t in zip(sets, targets) if index[s2] == i)) for i in range(len(unique))]
    n_asked = max(len(a) for a in asked)
    member = np.zeros((len(unique), N_RULES + N_GROUPS, n_asked))
    for i, a in enumerate(asked):
        member[i, :N_RULES, : len(a)] = RULE_MEMBERSHIP[:, a]
        member[i, N_RULES:, : len(a)] = in_concept[i][:, a]
    slot = {(i, t): i * n_asked + j for i, a in enumerate(asked) for j, t in enumerate(a)}
    return {
        "rule_log_lik": np.stack([RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "n_examples": np.array([[len(s)] for s in unique], dtype="float64"),
        "group_log_count": log_count,
        "cell_member": member,
        "row_cell": np.array([slot[(index[s], t)] for s, t in zip(sets, targets)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
