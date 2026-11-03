"""Refinement of `without_replacement_size_principle`, adding one component from
`digit_sum_concepts_threshold_erlang`: digit-sum concepts as their own family.

People do the same Bayesian concept learning over number rules, bounded ranges
[m, n] (Erlang prior s * exp(-s / sigma) over width) and one-sided thresholds
([1, n], [m, 100], their own family), with a prior mass per concept family, 10%
of examples taken as random draws from 1..100, and examples drawn from the
concept without replacement (the j-th example a concept contains has
probability 1 / (size - j)). The one change: the hypothesis space also holds
"numbers whose digits add up to s" for s = 1..18, a new family with its own
inferred prior mass shared equally among the 18 sums. Digit-sum concepts are
small (at most 10 members), so under without-replacement sampling a set whose
examples share a digit sum is an especially suspicious coincidence, and people
endorse other numbers with that sum and reject neighbours that break it.
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
    rules.update({f"digit sum {t}": [n for n in NUMBERS if sum(map(int, str(n))) == t] for t in range(1, 19)})
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
    if name.startswith("digit sum"):
        return 6
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

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    # Per set, log sum of P(D | range) over the bounded ranges of each width and over
    # all thresholds, and each group's posterior membership of every target (the
    # prior is uniform within a group, so these do not depend on the parameters).
    width_log_evidence = pm.Data("width_log_evidence", np.zeros((1, len(WIDTHS))))
    width_in_concept = pm.Data("width_in_concept", np.zeros((1, len(WIDTHS), 100)))
    threshold_log_evidence = pm.Data("threshold_log_evidence", np.zeros((1, 1)), shape=(None, 1))
    threshold_in_concept = pm.Data("threshold_in_concept", np.zeros((1, 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the seven rule families (digit sums the new one), bounded ranges and one-sided thresholds.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(9))
    log_w = pt.log(family_weight)
    # Characteristic width of the Erlang prior over bounded ranges.
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)

    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    log_per_range = erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH))

    log_rules = (log_w[:7] - np.log(FAMILY_SIZE))[RULE_FAMILY][None, :] + rule_log_lik
    log_widths = log_w[7] + log_per_range[None, :] + width_log_evidence
    log_thresholds = log_w[8] - np.log(N_THRESHOLD) + threshold_log_evidence
    log_joint = pt.concatenate([log_rules, log_widths, log_thresholds], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (
        posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)
        + (posterior[:, N_RULES:-1, None] * width_in_concept).sum(axis=1)
        + posterior[:, -1, None] * threshold_in_concept
    )  # (sets, targets)

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
    return {
        "rule_log_lik": np.stack([_without_replacement_log_lik(s, RULE_MEMBERSHIP) for s in unique]),
        "width_log_evidence": width_log_evidence,
        "width_in_concept": width_in_concept,
        "threshold_log_evidence": threshold_log_evidence[:, None],
        "threshold_in_concept": threshold_in_concept,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
