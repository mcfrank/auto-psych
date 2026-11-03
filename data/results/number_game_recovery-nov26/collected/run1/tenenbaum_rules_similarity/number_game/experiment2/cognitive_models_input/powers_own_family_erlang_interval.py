"""Bayesian concept learning in which repeated-multiplication rules ("powers of
m") form their own concept family: a refinement of
`erlang_interval_size_family_prior`.

The learner considers the same 45 rules and the 5,050 intervals of 1..100,
samples examples from the concept (size principle, 10% uniformly from 1..100),
gives each rule family an inferred prior mass shared equally by its members
(exception rules in their own family), puts an Erlang prior with a learned
characteristic size over interval width, and answers "yes" with the posterior
probability that the concept contains the target. The one change: the powers
of 2..10 are not lumped with primes, squares and cubes in one "special sets"
family; they are their own family with its own inferred prior mass. If people
find "powers of m" an especially natural program, examples such as 8, 64 are
explained by "powers of 8" and people reject 2, 4, 16 and 32, while sets such
as 16, 8, 64 favour powers of 2 over "even".
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
    if name.startswith("powers of"):
        return 6  # powers of m: their own family
    return 1  # primes, squares, cubes


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in RULE_NAMES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
N_INTERVALS = len(INTERVALS)
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_WIDTH = INTERVAL_MEMBERSHIP.sum(axis=1).astype("int64")
WIDTHS = np.arange(1, 101, dtype="float64")
N_PER_WIDTH = np.bincount(INTERVAL_WIDTH - 1, minlength=100).astype("float64")  # 101 - s

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    # log sum over intervals of width s of P(D | interval), and the membership
    # probability of each target given width s (precomputed; the prior is width-only).
    width_log_evidence = pm.Data("width_log_evidence", np.zeros((1, 100)))  # (sets, widths)
    width_in_concept = pm.Data("width_in_concept", np.zeros((1, 100, 100)))  # (sets, widths, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the seven rule families (incl. exception rules and powers of m) and of intervals.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(8))
    log_w = pt.log(family_weight)
    # Characteristic interval width of the Erlang size prior.
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)

    # Per-interval prior proportional to s * exp(-s / sigma), normalised over all intervals.
    erlang = np.log(WIDTHS) - WIDTHS / interval_scale
    log_per_interval = erlang - pt.logsumexp(erlang + np.log(N_PER_WIDTH))

    log_rules = (log_w[:7] - np.log(FAMILY_SIZE))[RULE_FAMILY][None, :] + rule_log_lik
    log_widths = log_w[7] + log_per_interval[None, :] + width_log_evidence
    log_joint = pt.concatenate([log_rules, log_widths], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + (
        posterior[:, N_RULES:, None] * width_in_concept
    ).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    interval_log_lik = set_log_lik[:, N_RULES:]
    log_evidence = np.zeros((len(unique), 100))
    in_concept = np.zeros((len(unique), 100, 100))
    for w in range(1, 101):
        sel = INTERVAL_WIDTH == w
        ll = interval_log_lik[:, sel]
        log_evidence[:, w - 1] = logsumexp(ll, axis=1)
        in_concept[:, w - 1] = np.exp(ll - log_evidence[:, w - 1, None]) @ INTERVAL_MEMBERSHIP[sel]
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "width_log_evidence": log_evidence,
        "width_in_concept": in_concept,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
