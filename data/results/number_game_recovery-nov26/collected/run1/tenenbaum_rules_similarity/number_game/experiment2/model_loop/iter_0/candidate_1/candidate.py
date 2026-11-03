"""Bayesian concept learning over rules and ranges on a logarithmic mental
number line.

People do Bayesian concept learning over number rules (a prior mass per concept
family, the exception rules in their own family) and numeric ranges with the
size principle, but ranges are bands on a compressed, logarithmic number line:
a range [a, b] is a priori as likely wherever it sits on the log line (prior
area d log a * d log b, i.e. proportional to 1 / (a * b)), and its plausibility
depends on its ratio width L = log((b + 0.5) / (a - 0.5)) through an Erlang prior
L * exp(-L / tau) with a learned characteristic log-width tau. So magnitude
generalisation scales with the examples' size. The size principle still counts
integers (examples are drawn uniformly from the members of the concept, 10%
uniformly from 1..100), and "yes" is the posterior probability that the
concept contains the target.
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
        return 5
    return 1


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in RULE_NAMES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
BOUNDS = [(m, n) for m in NUMBERS for n in range(m, 101)]
INTERVALS = [list(range(m, n + 1)) for m, n in BOUNDS]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
LO = np.array([m for m, _ in BOUNDS], dtype="float64")
HI = np.array([n for _, n in BOUNDS], dtype="float64")
LOG_WIDTH = np.log((HI + 0.5) / (LO - 0.5))  # ratio width on the log number line
LOG_AREA = -np.log(LO) - np.log(HI)  # uniform placement on the log line
# For speed, ratio widths are grouped into fine bins; the learned Erlang factor
# exp(-L / tau) is evaluated at each bin's centre, everything else exactly.
N_BINS = 150
EDGES = np.linspace(0.0, LOG_WIDTH.max() + 1e-9, N_BINS + 1)
BIN = np.digitize(LOG_WIDTH, EDGES) - 1
BIN_CENTRE = 0.5 * (EDGES[:-1] + EDGES[1:])
FIXED_LOG_PRIOR = LOG_AREA + np.log(LOG_WIDTH)
BIN_LOG_PRIOR = np.array([logsumexp(FIXED_LOG_PRIOR[BIN == b]) if np.any(BIN == b) else -np.inf for b in range(N_BINS)])
USED = np.isfinite(BIN_LOG_PRIOR)
BIN_CENTRE, BIN_LOG_PRIOR = BIN_CENTRE[USED], BIN_LOG_PRIOR[USED]
BIN_IDS = np.arange(N_BINS)[USED]

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    # Per set and ratio-width bin: log sum of fixed prior * P(D | interval), and the
    # posterior membership of each target within the bin (precomputed).
    bin_log_evidence = pm.Data("bin_log_evidence", np.zeros((1, len(BIN_IDS))))  # (sets, bins)
    bin_in_concept = pm.Data("bin_in_concept", np.zeros((1, len(BIN_IDS), 100)))  # (sets, bins, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_weight = pm.Dirichlet("family_weight", a=np.ones(7))
    log_w = pt.log(family_weight)
    # Characteristic ratio width of a range on the log number line.
    log_width_scale = pm.LogNormal("log_width_scale", mu=np.log(0.7), sigma=1.0)

    log_prior_bin = BIN_LOG_PRIOR - BIN_CENTRE / log_width_scale
    log_norm = pt.logsumexp(log_prior_bin)

    log_rules = (log_w[:6] - np.log(FAMILY_SIZE))[RULE_FAMILY][None, :] + rule_log_lik
    log_ints = log_w[6] - log_norm - BIN_CENTRE[None, :] / log_width_scale + bin_log_evidence
    log_joint = pt.concatenate([log_rules, log_ints], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP) + (
        posterior[:, N_RULES:, None] * bin_in_concept
    ).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    weighted = FIXED_LOG_PRIOR[None, :] + set_log_lik[:, N_RULES:]
    bin_log_evidence = np.zeros((len(unique), len(BIN_IDS)))
    bin_in_concept = np.zeros((len(unique), len(BIN_IDS), 100))
    for j, b in enumerate(BIN_IDS):
        sel = BIN == b
        w = weighted[:, sel]
        bin_log_evidence[:, j] = logsumexp(w, axis=1)
        bin_in_concept[:, j] = np.exp(w - bin_log_evidence[:, j, None]) @ INTERVAL_MEMBERSHIP[sel]
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "bin_log_evidence": bin_log_evidence,
        "bin_in_concept": bin_in_concept,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
