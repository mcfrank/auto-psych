"""Heterogeneous range specificity bias (refinement of narrow_range_specificity_bias).

Bayesian concept learning over 45 arithmetic and digit rules (equal prior) and
every interval of 1..100. Examples are distinct draws from the concept's
members not yet shown (sampling without replacement), 10% of them stray, with
a guessing lapse. Each person's prior over ranges is biased by width: an
interval of size s has prior weight proportional to s**gamma. The one change
from the incumbent is that gamma differs between people. It is distributed
Normal(mu_gamma, tau_gamma) across the population, and the population P(yes)
averages each person's posterior predictive over that distribution, using
Gauss-Hermite quadrature. tau_gamma = 0 recovers the incumbent.
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
SIZE = MEMBERSHIP.sum(axis=1)
INTERVAL_SIZE = SIZE[N_RULES:].astype(int)
WIDTHS = np.arange(1, 101, dtype="float64")
N_OF_WIDTH = 101.0 - WIDTHS  # how many intervals have each width
GH_NODES, GH_WEIGHTS = np.polynomial.hermite_e.hermegauss(7)
GH_WEIGHTS = GH_WEIGHTS / GH_WEIGHTS.sum()  # quadrature over a standard normal


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


# The likelihoods are parameter-free. prepare_observed collapses the rules per
# set, and the intervals per (set, width): their log evidence and the
# within-width posterior membership of every target.

with pm.Model() as model:
    rule_log_evidence = pm.Data("rule_log_evidence", np.zeros(1))  # (sets,)
    rule_in_concept = pm.Data("rule_in_concept", np.zeros((1, 100)))  # (sets, targets)
    width_log_evidence = pm.Data("width_log_evidence", np.zeros((1, 100)))  # (sets, widths)
    width_in_concept = pm.Data("width_in_concept", np.zeros((1, 100, 100)))  # (sets, widths, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # set index * 100 + target - 1
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    mu_gamma = pm.Normal("mu_gamma", 0.0, 1.0)  # population mean width bias
    tau_gamma = pm.HalfNormal("tau_gamma", 0.5)  # spread of the width bias across people

    # One learner per quadrature node: gamma_k = mu + tau * z_k, shape (nodes,).
    gamma = mu_gamma + tau_gamma * GH_NODES
    # Per-interval log prior over widths, normalised over all 5050 intervals.
    log_w = gamma[:, None] * np.log(WIDTHS)[None, :]
    log_prior = log_w - pt.logsumexp(log_w + np.log(N_OF_WIDTH)[None, :], axis=1, keepdims=True)
    weighted = width_log_evidence[None, :, :] + log_prior[:, None, :]  # (nodes, sets, widths)
    interval_log_evidence = pt.logsumexp(weighted, axis=2)  # (nodes, sets)
    width_post = pt.exp(weighted - interval_log_evidence[:, :, None])
    # (sets, nodes, widths) @ (sets, widths, targets) -> (sets, nodes, targets)
    interval_in_concept = pt.batched_dot(width_post.dimshuffle(1, 0, 2), width_in_concept)
    log_rules = pt.log(lambda_0) - np.log(N_RULES) + rule_log_evidence
    log_intervals = pt.log1p(-lambda_0) + interval_log_evidence.T  # (sets, nodes)
    p_rule = pm.math.sigmoid(log_rules[:, None] - log_intervals)[:, :, None]
    per_learner = p_rule * rule_in_concept[:, None, :] + (1 - p_rule) * interval_in_concept
    in_concept = (per_learner * GH_WEIGHTS[None, :, None]).sum(axis=1)  # population average

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
    interval_ll = log_lik_all[:, N_RULES:]
    interval_members = MEMBERSHIP[N_RULES:]
    width_ev = np.zeros((len(unique), 100))
    width_in = np.zeros((len(unique), 100, 100))
    for w in range(1, 101):
        mask = INTERVAL_SIZE == w
        ev = logsumexp(interval_ll[:, mask], axis=1)
        width_ev[:, w - 1] = ev
        width_in[:, w - 1] = np.exp(interval_ll[:, mask] - ev[:, None]) @ interval_members[mask]
    return {
        "rule_log_evidence": rule_ev,
        "rule_in_concept": np.exp(rule_ll - rule_ev[:, None]) @ MEMBERSHIP[:N_RULES],
        "width_log_evidence": width_ev,
        "width_in_concept": width_in,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


SAMPLER_SETTINGS = {"target_accept": 0.9}
