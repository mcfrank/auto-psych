"""Bayesian concept learning from a leaky memory of the examples.

People generalize by Bayesian concept learning over 45 rules (odd, even,
primes, squares, cubes, powers of 2-10, powers of 2 without 32 or with 37,
multiples of 3-9, 11 and 12, ending in 0-9, containing digit 0-9) and every
interval of 1-100, with a prior weight per rule and the size principle (10%
of examples uniform noise), exactly as in bigelow_piantadosi_independent --
but each example is independently retained in memory with probability
`retain`, and only the retained examples enter the inference. "Yes" is the
posterior probability that the concept contains the target, averaged over
which subset of the examples was remembered (if none, the prior predictive).
"""

import itertools

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
    return rules


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]

# "ends in 0" and "contains digit 0" are the same set; sample their summed
# weight and a Beta split (Dirichlet aggregation), as in the incumbent.
END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    # Per remembered subset of examples (all subsets of every set shown).
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))
    interval_log_evidence = pm.Data("interval_log_evidence", np.zeros(1))
    interval_in_concept = pm.Data("interval_in_concept", np.zeros((1, 100)))
    # Per (set, subset) pair: examples kept and forgotten (mask 0 if not a subset).
    n_kept = pm.Data("n_kept", np.zeros((1, 1)))
    n_lost = pm.Data("n_lost", np.zeros((1, 1)))
    is_subset = pm.Data("is_subset", np.ones((1, 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    retain = pm.Beta("retain", 3, 1)
    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + interval_log_evidence
    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    subset_in_concept = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)
                         + posterior[:, N_RULES][:, None] * interval_in_concept)  # (subsets, 100)

    memory_weight = is_subset * pt.exp(n_kept * pt.log(retain) + n_lost * pt.log1p(-retain))  # (sets, subsets)
    in_concept = memory_weight @ subset_in_concept  # (sets, 100)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    subsets = {}
    for s in unique:
        for k in range(len(s) + 1):
            for c in itertools.combinations(sorted(s), k):
                subsets.setdefault(c, len(subsets))
    sub_list = list(subsets)
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(c, dtype=int) - 1].sum(axis=1) for c in sub_list])
    interval_log_lik = set_log_lik[:, N_RULES:]
    log_evidence = logsumexp(interval_log_lik, axis=1)
    kept = np.zeros((len(unique), len(sub_list)))
    lost = np.zeros_like(kept)
    mask = np.zeros_like(kept)
    for i, s in enumerate(unique):
        seen = set()
        for k in range(len(s) + 1):
            for c in itertools.combinations(sorted(s), k):
                if c in seen:  # repeated example values give the same subset twice
                    continue
                seen.add(c)
                j = subsets[c]
                kept[i, j], lost[i, j], mask[i, j] = k, len(s) - k, 1.0
    # Repeated values in a set: count each position's subset multiplicity instead.
    for i, s in enumerate(unique):
        if len(set(s)) < len(s):
            kept[i], lost[i], mask[i] = 0, 0, 0
            mult = {}
            for k in range(len(s) + 1):
                for c in itertools.combinations(sorted(s), k):
                    mult[c] = mult.get(c, 0) + 1
            for c, m in mult.items():
                j = subsets[c]
                kept[i, j], lost[i, j], mask[i, j] = len(c), len(s) - len(c), float(m)
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "interval_log_evidence": log_evidence,
        "interval_in_concept": np.exp(interval_log_lik - log_evidence[:, None]) @ MEMBERSHIP[N_RULES:],
        "n_kept": kept,
        "n_lost": lost,
        "is_subset": mask,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
