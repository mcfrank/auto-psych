"""Refinement of digit_sum_concepts: Bayesian concept learning over 45 arithmetic
and digit rules, the 18 digit-sum concepts ("digits add up to s", family share
inferred) and every interval of 1..100 with a width-graded range prior; examples
are distinct draws (sampling without replacement), 10% stray, with a guessing
lapse. The one change (from open_ended_range_discount): ranges that run to an
end of the number line (starting at 1 or ending at 100) are open-ended threshold
rules, held at their own prior weight exp(a) relative to bounded ranges of the
same width, with a inferred.
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
BOUNDS = [(m, n) for m in NUMBERS for n in range(m, 101)]
INTERVALS = [list(range(m, n + 1)) for m, n in BOUNDS]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + SUM_CONCEPTS + INTERVALS], dtype="float64")
SIZE = MEMBERSHIP.sum(axis=1)
N_FINITE = N_RULES + N_SUMS

# Ranges grouped by (width, open-ended flag); the prior is constant within a group.
_width = np.array([n - m + 1 for m, n in BOUNDS])
_open = np.array([1 if (m == 1 or n == 100) else 0 for m, n in BOUNDS])
_group_all = (_width - 1) * 2 + _open
_used = np.unique(_group_all)
_remap = {g: i for i, g in enumerate(_used)}
GROUP = np.array([_remap[g] for g in _group_all])
N_GROUPS = len(_used)
GROUP_WIDTH = (_used // 2 + 1).astype("float64")
GROUP_OPEN = (_used % 2).astype("float64")
GROUP_COUNT = np.bincount(GROUP, minlength=N_GROUPS).astype("float64")


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


with pm.Model() as model:
    rule_log_evidence = pm.Data("rule_log_evidence", np.zeros(1))  # (sets,)
    rule_in_concept = pm.Data("rule_in_concept", np.zeros((1, 100)))  # (sets, targets)
    sum_log_evidence = pm.Data("sum_log_evidence", np.zeros(1))  # (sets,)
    sum_in_concept = pm.Data("sum_in_concept", np.zeros((1, 100)))  # (sets, targets)
    group_log_evidence = pm.Data("group_log_evidence", np.zeros((1, N_GROUPS)))  # (sets, groups)
    group_in_concept = pm.Data("group_in_concept", np.zeros((1, N_GROUPS, 100)))  # (sets, groups, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 1.0)  # width bias of the range prior
    a = pm.Normal("a", 0.0, 1.0)  # log prior weight of open-ended ranges vs bounded ones
    log_sum_share = pm.Normal("log_sum_share", -3.0, 1.5)

    log_w = gamma * np.log(GROUP_WIDTH) + a * GROUP_OPEN
    log_prior = log_w - pt.logsumexp(log_w + np.log(GROUP_COUNT))
    weighted = group_log_evidence + log_prior[None, :]
    interval_log_evidence = pt.logsumexp(weighted, axis=1)
    group_post = pt.exp(weighted - interval_log_evidence[:, None])
    interval_in_concept = (group_post[:, :, None] * group_in_concept).sum(axis=1)

    log_rules_only = rule_log_evidence - np.log(N_RULES)
    log_sums_only = sum_log_evidence - np.log(N_SUMS) + log_sum_share
    p_sum = pm.math.sigmoid(log_sums_only - log_rules_only)[:, None]
    finite_in_concept = (1 - p_sum) * rule_in_concept + p_sum * sum_in_concept
    log_finite = (
        pt.log(lambda_0) + pt.logaddexp(log_rules_only, log_sums_only) - pt.logaddexp(0.0, log_sum_share)
    )
    log_intervals = pt.log1p(-lambda_0) + interval_log_evidence
    p_rule = pm.math.sigmoid(log_finite - log_intervals)[:, None]
    in_concept = p_rule * finite_in_concept + (1 - p_rule) * interval_in_concept

    p_concept = pt.clip(in_concept.reshape((-1,))[row_cell], 0.0, 1.0)
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
    group_ev = np.zeros((len(unique), N_GROUPS))
    group_in = np.zeros((len(unique), N_GROUPS, 100))
    for g in range(N_GROUPS):
        mask = GROUP == g
        ev = logsumexp(interval_ll[:, mask], axis=1)
        group_ev[:, g] = ev
        group_in[:, g] = np.exp(interval_ll[:, mask] - ev[:, None]) @ interval_members[mask]
    return {
        "rule_log_evidence": rule_ev,
        "rule_in_concept": np.exp(rule_ll - rule_ev[:, None]) @ MEMBERSHIP[:N_RULES],
        "sum_log_evidence": sum_ev,
        "sum_in_concept": np.exp(sum_ll - sum_ev[:, None]) @ MEMBERSHIP[N_RULES:N_FINITE],
        "group_log_evidence": group_ev,
        "group_in_concept": group_in,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


SAMPLER_SETTINGS = {"target_accept": 0.9}
