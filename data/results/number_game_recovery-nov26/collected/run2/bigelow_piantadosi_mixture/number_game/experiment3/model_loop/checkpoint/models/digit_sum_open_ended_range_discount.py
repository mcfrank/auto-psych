"""Digit-sum concepts with an open-ended range discount (refines
digit_sum_concepts). The one change: a range starting at 1 or ending at 100 is
a threshold rule ("anything up to 30") and takes its own inferred prior weight
(log_open) relative to bounded ranges, on top of the width-graded prior.

Bayesian concept learning over 45 arithmetic and digit
rules, every interval of 1..100 (with a width-graded range prior) and one more
concept family read off the written numerals, "numbers whose digits add up to
s" (s = 1..18). The 18 digit-sum concepts share equal prior; the family's
weight relative to a single rule (log_sum_share) is inferred. Examples are
distinct draws from the concept's members not yet shown (sampling without
replacement), 10% of them stray, with a guessing lapse. Examples sharing a
digit sum (e.g. 36, 45, 54, 81) then generalize to other numbers with that
digit sum (63, 72, 90, 27).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


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
_LO = np.array([c[0] for c in INTERVALS])
_HI = np.array([c[-1] for c in INTERVALS])
_OPEN = ((_LO == 1) | (_HI == 100)).astype(int)
# Intervals are grouped by (width, open-ended flag); only non-empty groups kept.
_GROUP_KEYS = sorted(set(zip(INTERVAL_SIZE.tolist(), _OPEN.tolist())))
_GROUP_OF = {g: i for i, g in enumerate(_GROUP_KEYS)}
INTERVAL_GROUP = np.array([_GROUP_OF[(s, o)] for s, o in zip(INTERVAL_SIZE, _OPEN)])
N_GROUPS = len(_GROUP_KEYS)
GROUP_WIDTH = np.array([g[0] for g in _GROUP_KEYS], dtype="float64")
GROUP_OPEN = np.array([g[1] for g in _GROUP_KEYS], dtype="float64")
GROUP_COUNT = np.bincount(INTERVAL_GROUP, minlength=N_GROUPS).astype("float64")


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


# The likelihoods are parameter-free. prepare_observed collapses the rules and
# the digit-sum concepts per set, and the intervals per (set, width, open-ended): their log
# evidence and the within-group posterior membership of every target.

with pm.Model() as model:
    rule_log_evidence = pm.Data("rule_log_evidence", np.zeros(1))  # (sets,)
    sum_log_evidence = pm.Data("sum_log_evidence", np.zeros(1))  # (sets,)
    width_log_evidence = pm.Data("width_log_evidence", np.zeros((1, N_GROUPS)))  # (sets, groups)
    # Only the (set, target) cells that were asked about: their within-group membership.
    rule_in_concept = pm.Data("rule_in_concept", np.zeros(1))  # (cells,)
    sum_in_concept = pm.Data("sum_in_concept", np.zeros(1))  # (cells,)
    width_in_concept = pm.Data("width_in_concept", np.zeros((1, N_GROUPS)))  # (cells, groups)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # set index of each cell
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # cell index of each trial
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 1.0)  # width bias of the range prior
    # Prior weight of the digit-sum family relative to the rule family.
    log_sum_share = pm.Normal("log_sum_share", -3.0, 1.5)
    # Log prior weight of an open-ended range (starting at 1 or ending at 100) vs a bounded one.
    log_open = pm.Normal("log_open", 0.0, 1.0)

    log_w = gamma * np.log(GROUP_WIDTH) + log_open * GROUP_OPEN
    log_prior = log_w - pt.logsumexp(log_w + np.log(GROUP_COUNT))
    weighted = width_log_evidence + log_prior[None, :]
    interval_log_evidence = pt.logsumexp(weighted, axis=1)
    width_post = pt.exp(weighted - interval_log_evidence[:, None])
    interval_in_concept = (width_post[cell_set] * width_in_concept).sum(axis=1)

    # Within the rule share lambda_0: rules and digit-sum family split 1 : exp(log_sum_share).
    log_rules_only = rule_log_evidence - np.log(N_RULES)
    log_sums_only = sum_log_evidence - np.log(N_SUMS) + log_sum_share
    p_sum = pm.math.sigmoid(log_sums_only - log_rules_only)[cell_set]
    finite_in_concept = (1 - p_sum) * rule_in_concept + p_sum * sum_in_concept
    log_finite = (
        pt.log(lambda_0) + pt.logaddexp(log_rules_only, log_sums_only) - pt.logaddexp(0.0, log_sum_share)
    )
    log_intervals = pt.log1p(-lambda_0) + interval_log_evidence
    p_rule = pm.math.sigmoid(log_finite - log_intervals)[cell_set]
    in_concept = p_rule * finite_in_concept + (1 - p_rule) * interval_in_concept

    p_concept = pt.clip(in_concept[row_cell], 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    log_lik_all = np.stack([set_log_lik(s) for s in unique])
    rule_ll = log_lik_all[:, :N_RULES]
    rule_ev = logsumexp(rule_ll, axis=1)
    sum_ll = log_lik_all[:, N_RULES:N_FINITE]
    sum_ev = logsumexp(sum_ll, axis=1)
    interval_ll = log_lik_all[:, N_FINITE:]
    interval_members = MEMBERSHIP[N_FINITE:]
    width_ev = np.zeros((len(unique), N_GROUPS))
    width_in = np.zeros((len(unique), N_GROUPS, 100))
    for g in range(N_GROUPS):
        mask = INTERVAL_GROUP == g
        ev = logsumexp(interval_ll[:, mask], axis=1)
        width_ev[:, g] = ev
        width_in[:, g] = np.exp(interval_ll[:, mask] - ev[:, None]) @ interval_members[mask]
    flat = [index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)]
    cells = sorted(set(flat))
    cell_of = {c: i for i, c in enumerate(cells)}
    c_set = np.array([c // 100 for c in cells], dtype="int64")
    c_tgt = np.array([c % 100 for c in cells], dtype="int64")
    rule_in = np.exp(rule_ll - rule_ev[:, None]) @ MEMBERSHIP[:N_RULES]
    sum_in = np.exp(sum_ll - sum_ev[:, None]) @ MEMBERSHIP[N_RULES:N_FINITE]
    return {
        "rule_log_evidence": rule_ev,
        "sum_log_evidence": sum_ev,
        "width_log_evidence": width_ev,
        "rule_in_concept": rule_in[c_set, c_tgt],
        "sum_in_concept": sum_in[c_set, c_tgt],
        "width_in_concept": width_in[c_set, :, c_tgt],
        "cell_set": c_set,
        "row_cell": np.array([cell_of[c] for c in flat], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


SAMPLER_SETTINGS = {"target_accept": 0.9}
