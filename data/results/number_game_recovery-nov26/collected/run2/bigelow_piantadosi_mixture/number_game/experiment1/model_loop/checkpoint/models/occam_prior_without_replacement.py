"""Refinement of occam_specificity_prior_all_concepts (itself a refinement of
equal_prior_concepts_with_guess_lapse): Bayesian concept learning
over the same 45 rules and every interval of 1..100 (size principle, 10% of
examples drawn uniformly), with a response lapse -- on a fraction `lapse` of
trials the participant ignores the concept and answers "yes" with probability
`guess_yes`. The one change: one shared Occam-style specificity bias over all
concepts. Within the rules (share lambda_0) and within the intervals (share
1 - lambda_0) each concept's prior is proportional to |concept| ** -g, the same
g for both, so g > 0 favours narrow rules and short ranges alike (g = 0 is the
flat prior). The one change here: examples are taken to be distinct, so each
is drawn from the concept's members not yet shown (strong sampling without
replacement): the k-th example's likelihood under h is
0.9 * [x in h] / (|h| - members of h already shown) + 0.1 / 100.
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
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
CONCEPT_SIZE = MEMBERSHIP.sum(axis=1)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_SIZE = np.array([len(c) for c in INTERVALS])
SIZES = np.arange(1, 101)
# Centre log sizes so g does not trade off against the normalisations.
LOG_RULE_SIZE_C = np.log(RULE_MEMBERSHIP.sum(axis=1)) - np.log(RULE_MEMBERSHIP.sum(axis=1)).mean()
LOG_SIZE_C = np.log(SIZES) - np.log(SIZES).mean()
LOG_COUNT_PER_SIZE = np.log(101 - SIZES)  # number of intervals of each size
SAMPLER_SETTINGS = {"target_accept": 0.9}


def set_log_lik(examples):
    """log P(examples | h) for every hypothesis, sampling distinct members without replacement."""
    total = np.zeros(len(MEMBERSHIP))
    shown = np.zeros(len(MEMBERSHIP))
    seen = set()
    for x in examples:
        if x in seen:  # a repeated number can only be a stray draw
            total += np.log((1 - ALPHA) / 100)
            continue
        seen.add(x)
        in_h = MEMBERSHIP[:, x - 1]
        remaining = np.maximum(CONCEPT_SIZE - shown, 1.0)
        total += np.log(ALPHA * in_h / remaining + (1 - ALPHA) / 100)
        shown += in_h
    return total

# Intervals of one size share a prior (and, without replacement, the same
# member-likelihood pattern), so prepare_observed collapses them per
# example set and size: total evidence and evidence-weighted target membership.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    size_log_evidence = pm.Data("size_log_evidence", np.zeros((1, 100)))  # (sets, sizes): mean log-evidence
    size_in_concept = pm.Data("size_in_concept", np.zeros((1, 100, 100)))  # (sets, sizes, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    g = pm.Normal("g", mu=0.0, sigma=1.0)  # specificity bias shared by all concepts
    log_rule_prior = -g * LOG_RULE_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior) + pt.log(lambda_0)
    log_size_prior = -g * LOG_SIZE_C + LOG_COUNT_PER_SIZE  # per interval of size s, times count
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior) + pt.log1p(-lambda_0)
    log_rules = log_rule_prior[None, :] + rule_log_lik  # (sets, rules)
    log_sizes = log_size_prior[None, :] + size_log_evidence  # (sets, sizes)
    log_total = pt.logaddexp(pt.logsumexp(log_rules, axis=1, keepdims=True), pt.logsumexp(log_sizes, axis=1, keepdims=True))
    rule_post = pt.exp(log_rules - log_total)
    size_post = pt.exp(log_sizes - log_total)
    in_concept = pt.dot(rule_post, RULE_MEMBERSHIP) + (size_post[:, :, None] * size_in_concept).sum(axis=1)

    # Rounding can put a sure membership a few ulps above 1.
    p_concept = pt.clip(in_concept.reshape((-1,))[row_cell], 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)  # share of trials answered by guessing
    guess_yes = pm.Beta("guess_yes", 2, 2)  # P(yes) when guessing
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    all_log_lik = np.stack([set_log_lik(s) for s in unique])
    interval_log_lik = all_log_lik[:, N_RULES:]
    n_sets = len(unique)
    size_log_evidence = np.zeros((n_sets, 100))
    size_in_concept = np.zeros((n_sets, 100, 100))
    for k, s in enumerate(SIZES):
        cols = np.where(INTERVAL_SIZE == s)[0]
        ll = interval_log_lik[:, cols]
        lse = logsumexp(ll, axis=1)
        size_log_evidence[:, k] = lse - np.log(len(cols))  # mean evidence of size-s intervals
        size_in_concept[:, k, :] = np.exp(ll - lse[:, None]) @ MEMBERSHIP[N_RULES + cols]
    return {
        "rule_log_lik": all_log_lik[:, :N_RULES],
        "size_log_evidence": size_log_evidence,
        "size_in_concept": size_in_concept,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
