"""Limited consideration set: people do not search the whole space of possible
programs on every trial. Each kind of program (parity, special sets such as
primes, squares and powers, multiples, final digit, contained digit, exception
rules, bounded ranges, one-sided thresholds) comes to mind on a given trial
only with its own inferred probability, and people do Bayesian concept learning
(size principle with draws without replacement, Erlang prior on range width,
10% of examples random) only over the kinds that came to mind, each considered
kind sharing the prior equally. So when the kind of program that would best fit
the examples is not thought of, people fall back on whatever kind they did
consider: "yes" is the average, over which kinds came to mind, of the posterior
probability that the program produces the target. Availability replaces the
fixed prior mass per family of the best model.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
ALPHA = 0.9


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
# Bounded ranges [m, n] with 1 < m <= n < 100, and one-sided thresholds [1, n] or [m, 100].
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
THRESHOLD = (LO == 1) | (HI == 100)
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
INTERVAL_WIDTH = HI - LO + 1
WIDTHS = np.arange(1, 99)  # widths a bounded range can have
N_PER_WIDTH = np.array([np.sum(~THRESHOLD & (INTERVAL_WIDTH == w)) for w in WIDTHS], dtype="float64")
N_THRESHOLD = float(THRESHOLD.sum())

FAMILY_RULES = [np.flatnonzero(RULE_FAMILY == f) for f in range(6)]
N_FAM = 8
# Every non-empty subset of the 8 kinds of program that can come to mind.
SUBSETS = np.array([[(s >> f) & 1 for f in range(N_FAM)] for s in range(1, 2**N_FAM)], dtype="float64")

with pm.Model() as model:
    # Per set: log mean P(D | rule) within each rule family and each family's
    # posterior membership of every target (constant), as for ranges by width.
    rule_family_log_evidence = pm.Data("rule_family_log_evidence", np.zeros((1, 6)))
    rule_family_in_concept = pm.Data("rule_family_in_concept", np.zeros((1, 6, 100)))
    width_log_evidence = pm.Data("width_log_evidence", np.zeros((1, len(WIDTHS))))
    width_in_concept = pm.Data("width_in_concept", np.zeros((1, len(WIDTHS), 100)))
    threshold_log_evidence = pm.Data("threshold_log_evidence", np.zeros((1, 1)), shape=(None, 1))
    threshold_in_concept = pm.Data("threshold_in_concept", np.zeros((1, 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Probability that each kind of program comes to mind on a trial.
    availability = pm.Beta("availability", alpha=2.0, beta=2.0, shape=N_FAM)
    # Characteristic width of the Erlang prior over bounded ranges.
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)

    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    log_per_range = erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH))
    range_joint = log_per_range[None, :] + width_log_evidence  # (sets, widths)
    range_log_evidence = pt.logsumexp(range_joint, axis=1, keepdims=True)
    range_in_concept = (pt.exp(range_joint - range_log_evidence)[:, :, None] * width_in_concept).sum(axis=1)

    log_evidence = pt.concatenate(
        [rule_family_log_evidence, range_log_evidence, threshold_log_evidence - np.log(N_THRESHOLD)], axis=1
    )
    in_family = pt.concatenate(
        [rule_family_in_concept, range_in_concept[:, None, :], threshold_in_concept[:, None, :]], axis=1
    )  # (sets, families, targets)

    # Probability of each consideration set, conditioned on at least one kind coming to mind.
    log_q, log_not_q = pt.log(availability), pt.log1p(-availability)
    log_subset = pt.as_tensor(SUBSETS) @ log_q + pt.as_tensor(1 - SUBSETS) @ log_not_q
    p_subset = pt.exp(log_subset - pt.logsumexp(log_subset))
    # Posterior share of each family within each consideration set (equal prior per family).
    masked = log_evidence[None, :, :] + pt.as_tensor(np.where(SUBSETS > 0, 0.0, -1e4))[:, None, :]
    share = pt.exp(masked - pt.logsumexp(masked, axis=2, keepdims=True))  # (subsets, sets, families)
    family_weight = pt.tensordot(p_subset, share, axes=[[0], [0]])  # (sets, families)
    in_concept = (family_weight[:, :, None] * in_family).sum(axis=1)  # (sets, targets)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _without_replacement_log_lik(examples, membership):
    """Log P(examples | concept) for every concept, the program drawing its outputs
    without replacement: the j-th example a concept contains is one of its members
    not yet shown, so it has probability 1 / (size - j) rather than 1 / size."""
    size = membership.sum(axis=1)
    shown = np.zeros(len(membership))
    total = np.zeros(len(membership))
    for x in examples:
        inside = membership[:, x - 1]
        left = np.maximum(size - shown, 1.0)
        total += np.log(ALPHA * inside / left + (1 - ALPHA) / 100)
        shown += inside
    return total


def _group(log_lik, membership):
    """Log evidence of a group of equally likely concepts and its membership of each target."""
    top = log_lik.max(axis=1, keepdims=True)
    weight = np.exp(log_lik - top)
    total = weight.sum(axis=1)
    return np.log(total) + top[:, 0], (weight / total[:, None]) @ membership


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    interval_log_lik = np.stack([_without_replacement_log_lik(s, INTERVAL_MEMBERSHIP) for s in unique])
    width_log_evidence = np.zeros((len(unique), len(WIDTHS)))
    width_in_concept = np.zeros((len(unique), len(WIDTHS), 100))
    for i, w in enumerate(WIDTHS):
        sel = ~THRESHOLD & (INTERVAL_WIDTH == w)
        width_log_evidence[:, i], width_in_concept[:, i] = _group(interval_log_lik[:, sel], INTERVAL_MEMBERSHIP[sel])
    threshold_log_evidence, threshold_in_concept = _group(
        interval_log_lik[:, THRESHOLD], INTERVAL_MEMBERSHIP[THRESHOLD]
    )
    rule_log_lik = np.stack([_without_replacement_log_lik(s, RULE_MEMBERSHIP) for s in unique])
    rule_family_log_evidence = np.zeros((len(unique), 6))
    rule_family_in_concept = np.zeros((len(unique), 6, 100))
    for f, sel in enumerate(FAMILY_RULES):
        log_sum, rule_family_in_concept[:, f] = _group(rule_log_lik[:, sel], RULE_MEMBERSHIP[sel])
        rule_family_log_evidence[:, f] = log_sum - np.log(len(sel))
    return {
        "rule_family_log_evidence": rule_family_log_evidence,
        "rule_family_in_concept": rule_family_in_concept,
        "width_log_evidence": width_log_evidence,
        "width_in_concept": width_in_concept,
        "threshold_log_evidence": threshold_log_evidence[:, None],
        "threshold_in_concept": threshold_in_concept,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
