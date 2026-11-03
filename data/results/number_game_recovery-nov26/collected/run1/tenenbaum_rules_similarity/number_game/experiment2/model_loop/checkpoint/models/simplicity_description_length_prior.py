"""Simplicity (description-length) prior over programs.

People infer the hidden program as a Bayesian concept learner would: number
rules, bounded ranges [m, n] and one-sided thresholds ([1, n], [m, 100]) are
weighed by how well they explain the examples (the size principle, examples
drawn without repetition, 10% of examples random draws from 1..100). Their
prior plausibility of a program comes from one simplicity bias, not from
learned preferences per kind of concept: each program loses the same amount of
log prior for every symbol needed to state it ("odd" 1, "multiples of 7" 2,
"above 60" 2, "between 23 and 41" 3, "powers of 2 except 32" 4). One inferred
cost per symbol fixes how readily examples are read as a rule, a range or a
threshold. Bounded ranges are equally likely whatever their width.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
ALPHA = 0.9


def rule_concepts():
    """Each rule with its member list and its description length in symbols."""
    rules = {
        "odd": ([n for n in NUMBERS if n % 2 == 1], 1),
        "even": (list(range(2, 101, 2)), 1),
        "primes": ([n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))], 1),
        "squares": ([n * n for n in range(1, 11)], 1),
        "cubes": ([n**3 for n in range(1, 5)], 1),
        "powers of 2, not 32": ([2, 4, 8, 16, 64], 4),
        "powers of 2, or 37": ([2, 4, 8, 16, 32, 37, 64], 4),
    }
    rules.update({f"multiples of {m}": (list(range(m, 101, m)), 2) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": ([m**k for k in range(1, 8) if m**k <= 100], 2) for m in range(2, 11)})
    rules.update({f"ends in {d}": ([n for n in NUMBERS if n % 10 == d], 2) for d in range(10)})
    rules.update({f"contains digit {d}": ([n for n in NUMBERS if str(d) in str(n)], 2) for d in range(10)})
    return rules


_RULES = rule_concepts()
RULES = [v[0] for v in _RULES.values()]
N_RULES = len(RULES)
RULE_LENGTH = np.array([v[1] for v in _RULES.values()], dtype="float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
THRESHOLD = (LO == 1) | (HI == 100)
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
N_BOUNDED = float((~THRESHOLD).sum())
N_THRESHOLD = float(THRESHOLD.sum())
RANGE_LENGTH, THRESHOLD_LENGTH = 3.0, 2.0

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    # Per set: log sum of P(D | concept) over all bounded ranges / all thresholds
    # (each group shares one description length, so its prior is uniform within)
    # and the group's posterior membership of every target.
    range_log_evidence = pm.Data("range_log_evidence", np.zeros((1, 1)), shape=(None, 1))
    range_in_concept = pm.Data("range_in_concept", np.zeros((1, 100)))
    threshold_log_evidence = pm.Data("threshold_log_evidence", np.zeros((1, 1)), shape=(None, 1))
    threshold_in_concept = pm.Data("threshold_in_concept", np.zeros((1, 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Log-prior cost of each symbol in a program's description.
    symbol_cost = pm.HalfNormal("symbol_cost", sigma=5.0)

    log_rules = -symbol_cost * RULE_LENGTH[None, :] + rule_log_lik
    log_ranges = -symbol_cost * RANGE_LENGTH + range_log_evidence
    log_thresholds = -symbol_cost * THRESHOLD_LENGTH + threshold_log_evidence
    log_joint = pt.concatenate([log_rules, log_ranges, log_thresholds], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (
        posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)
        + posterior[:, N_RULES, None] * range_in_concept
        + posterior[:, N_RULES + 1, None] * threshold_in_concept
    )

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _without_replacement_log_lik(examples, membership):
    """Log P(examples | concept), the j-th in-concept example drawn from members not yet shown."""
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
    """Log summed likelihood of a group of concepts and its posterior membership of each target."""
    top = log_lik.max(axis=1, keepdims=True)
    weight = np.exp(log_lik - top)
    total = weight.sum(axis=1)
    return np.log(total) + top[:, 0], (weight / total[:, None]) @ membership


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    interval_log_lik = np.stack([_without_replacement_log_lik(s, INTERVAL_MEMBERSHIP) for s in unique])
    range_ev, range_in = _group(interval_log_lik[:, ~THRESHOLD], INTERVAL_MEMBERSHIP[~THRESHOLD])
    thr_ev, thr_in = _group(interval_log_lik[:, THRESHOLD], INTERVAL_MEMBERSHIP[THRESHOLD])
    return {
        "rule_log_lik": np.stack([_without_replacement_log_lik(s, RULE_MEMBERSHIP) for s in unique]),
        "range_log_evidence": range_ev[:, None],
        "range_in_concept": range_in,
        "threshold_log_evidence": thr_ev[:, None],
        "threshold_in_concept": thr_in,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
