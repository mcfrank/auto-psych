"""Sequence-direction extrapolation: people do Bayesian concept learning over
the same arithmetic and digit rules, digit-sum concepts and numeric ranges (with
a width-graded range prior) as digit_sum_concepts, reading the examples as
distinct draws with 10% strays and a guessing lapse. The one change is that they
read the order the examples were shown in as a direction of travel (rising when
the last example is above the first, falling when below) and take it as evidence
that the program's range runs on in that direction: a range is credited (or
discounted, by an inferred strength) by how far it extends beyond the examples
ahead of the sequence minus how far it extends behind. This predicts more "yes"
to out-of-span targets beyond the last-shown end of a set than beyond the
first-shown end (e.g. 70 after 52, 60, 66 but not after 66, 60, 52), an order
effect the incumbent, which ignores order for ranges, cannot give.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
N_TERMS = 20  # Taylor terms for exp(beta * a), |a| <= 1


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
    return list(rules.values())


RULES = rule_concepts()
N_RULES = len(RULES)
DIGIT_SUM = np.array([sum(int(c) for c in str(n)) for n in NUMBERS])
SUM_CONCEPTS = [list(NUMBERS[DIGIT_SUM == s]) for s in range(1, 19)]
N_SUMS = len(SUM_CONCEPTS)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + SUM_CONCEPTS + INTERVALS], dtype="float64")
SIZE = MEMBERSHIP.sum(axis=1)
N_FINITE = N_RULES + N_SUMS
INTERVAL_SIZE = SIZE[N_FINITE:].astype(int)
INTERVAL_LO = np.array([c[0] for c in INTERVALS])
INTERVAL_HI = np.array([c[-1] for c in INTERVALS])
WIDTHS = np.arange(1, 101, dtype="float64")
N_OF_WIDTH = 101.0 - WIDTHS
FACTORIALS = np.array([math.factorial(k) for k in range(N_TERMS)], dtype="float64")


def set_log_lik(examples):
    """log P(examples in order | h) for every hypothesis, sampling members without replacement."""
    total = np.zeros(len(MEMBERSHIP))
    shown = np.zeros(len(MEMBERSHIP))
    for x in examples:
        member = MEMBERSHIP[:, x - 1]
        remaining = np.maximum(SIZE - shown, 1.0)
        total += np.log(ALPHA * member / remaining + (1 - ALPHA) / 100)
        shown += member
    return total


def forward_extension(examples):
    """Per range: (extent ahead of the sequence - extent behind it) / 99, signed by its direction."""
    direction = np.sign(examples[-1] - examples[0])
    above = np.maximum(INTERVAL_HI - max(examples), 0)
    below = np.maximum(min(examples) - INTERVAL_LO, 0)
    return direction * (above - below) / 99.0


# prepare_observed collapses rules and digit-sum concepts per set. Ranges are
# grouped per (set, width); within a group the direction credit exp(beta * a)
# is expanded as a Taylor series, so the data are the moments sum L * a^k / k!
# of the range likelihoods L (scaled by the group maximum), for the evidence
# and, per observed (set, target) cell, for the members containing the target.

with pm.Model() as model:
    rule_log_evidence = pm.Data("rule_log_evidence", np.zeros(1))  # (sets,)
    rule_in_concept = pm.Data("rule_in_concept", np.zeros((1, 100)))  # (sets, targets)
    sum_log_evidence = pm.Data("sum_log_evidence", np.zeros(1))
    sum_in_concept = pm.Data("sum_in_concept", np.zeros((1, 100)))
    width_log_scale = pm.Data("width_log_scale", np.zeros((1, 100)))  # (sets, widths)
    width_moments = pm.Data("width_moments", np.zeros((1, 100, N_TERMS)))  # (sets, widths, k)
    cell_moments = pm.Data("cell_moments", np.zeros((1, 100, N_TERMS)))  # (cells, widths, k)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 1.0)
    log_sum_share = pm.Normal("log_sum_share", -3.0, 1.5)
    # Strength of the credit for ranges running on in the sequence's direction.
    beta = pm.Normal("beta", 0.0, 1.0)

    powers = pt.stack([pt.ones(()), beta] + [beta**k for k in range(2, N_TERMS)])  # beta^k
    log_w = gamma * np.log(WIDTHS)
    log_prior = log_w - pt.logsumexp(log_w + np.log(N_OF_WIDTH))
    width_sum = pt.maximum(pt.dot(width_moments, powers), 1e-300)  # (sets, widths)
    weighted = width_log_scale + pt.log(width_sum) + log_prior[None, :]
    interval_log_evidence = pt.logsumexp(weighted, axis=1)
    # Interval posterior membership per observed cell.
    cell_weight = pt.exp(width_log_scale[cell_set] + log_prior[None, :] - interval_log_evidence[cell_set][:, None])
    interval_in_cell = (cell_weight * pt.dot(cell_moments, powers)).sum(axis=1)

    log_rules_only = rule_log_evidence - np.log(N_RULES)
    log_sums_only = sum_log_evidence - np.log(N_SUMS) + log_sum_share
    p_sum = pm.math.sigmoid(log_sums_only - log_rules_only)
    finite_in_cell = (1 - p_sum[cell_set]) * rule_in_concept[cell_set, cell_target] + p_sum[
        cell_set
    ] * sum_in_concept[cell_set, cell_target]
    log_finite = (
        pt.log(lambda_0) + pt.logaddexp(log_rules_only, log_sums_only) - pt.logaddexp(0.0, log_sum_share)
    )
    log_intervals = pt.log1p(-lambda_0) + interval_log_evidence
    p_rule = pm.math.sigmoid(log_finite - log_intervals)[cell_set]
    in_cell = p_rule * finite_in_cell + (1 - p_rule) * interval_in_cell

    p_concept = pt.clip(in_cell[row_cell], 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys((index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)))
    cell_index = {c: i for i, c in enumerate(cells)}
    log_lik_all = np.stack([set_log_lik(s) for s in unique])
    rule_ll = log_lik_all[:, :N_RULES]
    rule_ev = logsumexp(rule_ll, axis=1)
    sum_ll = log_lik_all[:, N_RULES:N_FINITE]
    sum_ev = logsumexp(sum_ll, axis=1)
    interval_ll = log_lik_all[:, N_FINITE:]
    interval_members = MEMBERSHIP[N_FINITE:]
    ks = np.arange(N_TERMS)
    scale = np.zeros((len(unique), 100))
    width_mom = np.zeros((len(unique), 100, N_TERMS))
    cell_mom = np.zeros((len(cells), 100, N_TERMS))
    cells_of_set = {}
    for ci, (si, t) in enumerate(cells):
        cells_of_set.setdefault(si, []).append((ci, t))
    for si, s in enumerate(unique):
        a = forward_extension(s)
        for w in range(1, 101):
            mask = INTERVAL_SIZE == w
            ll = interval_ll[si, mask]
            top = ll.max()
            lik = np.exp(ll - top)
            terms = lik[:, None] * a[mask][:, None] ** ks[None, :] / FACTORIALS[None, :]
            scale[si, w - 1] = top
            width_mom[si, w - 1] = terms.sum(axis=0)
            members = interval_members[mask]
            for ci, t in cells_of_set[si]:
                cell_mom[ci, w - 1] = members[:, t] @ terms
    return {
        "rule_log_evidence": rule_ev,
        "rule_in_concept": np.exp(rule_ll - rule_ev[:, None]) @ MEMBERSHIP[:N_RULES],
        "sum_log_evidence": sum_ev,
        "sum_in_concept": np.exp(sum_ll - sum_ev[:, None]) @ MEMBERSHIP[N_RULES:N_FINITE],
        "width_log_scale": scale,
        "width_moments": width_mom,
        "cell_moments": cell_mom,
        "cell_set": np.array([c[0] for c in cells], dtype="int64"),
        "cell_target": np.array([c[1] for c in cells], dtype="int64"),
        "row_cell": np.array([cell_index[(index[s], int(r["target"]) - 1)] for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


SAMPLER_SETTINGS = {"target_accept": 0.9}
