"""People do Bayesian concept learning over 45 number rules and one-sided
thresholds ([1, n], [m, 100], their own family), with a prior mass per concept
family, the size principle and 10% of examples taken as random draws from
1..100, but magnitude-based programs are graded prototypes rather than
sharp-edged ranges. A prototype has a centre mu (1..100) and a spread sd; it
produces number x with probability proportional to exp(-(x - mu)^2 / (2 sd^2))
and a target belongs to it to that same graded degree (1 at the centre).
The prior over the spread is Erlang-shaped, sd * exp(-sd / scale), with a
learned characteristic scale; centres are equally likely. So magnitude
generalisation falls off smoothly with distance from the examples instead of
dropping at a range boundary.
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
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
RULES = list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in _RULES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
# One-sided thresholds [1, n] or [m, 100].
LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
THRESHOLD = (LO == 1) | (HI == 100)
THRESHOLD_MEMBERSHIP = ((NUMBERS[None, :] >= LO[THRESHOLD, None]) & (NUMBERS[None, :] <= HI[THRESHOLD, None])).astype(
    "float64"
)
N_THRESHOLD = float(THRESHOLD.sum())
# Prototype spreads on a log grid; each spread has 100 centres (mu = 1..100).
SPREADS = np.exp(np.linspace(np.log(0.7), np.log(60.0), 40))
LOG_DSPREAD = np.log(np.gradient(SPREADS))
_OFFSET = NUMBERS[None, :] - NUMBERS[:, None]  # (centre, number)
PROTOTYPES = [np.exp(-0.5 * (_OFFSET / sd) ** 2) for sd in SPREADS]  # graded membership

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    # Per set, log sum of P(D | prototype) over the 100 centres of each spread and over
    # all thresholds, and each group's posterior membership of every target (the
    # prior is uniform within a group, so these do not depend on the parameters).
    spread_log_evidence = pm.Data("spread_log_evidence", np.zeros((1, len(SPREADS))))
    spread_in_concept = pm.Data("spread_in_concept", np.zeros((1, len(SPREADS), 100)))
    threshold_log_evidence = pm.Data("threshold_log_evidence", np.zeros((1, 1)), shape=(None, 1))
    threshold_in_concept = pm.Data("threshold_in_concept", np.zeros((1, 100)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the six rule families, magnitude prototypes and one-sided thresholds.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(8))
    log_w = pt.log(family_weight)
    # Characteristic spread of the Erlang prior over prototype spreads.
    spread_scale = pm.LogNormal("spread_scale", mu=np.log(10.0), sigma=1.0)

    erlang = np.log(SPREADS) - SPREADS / spread_scale + LOG_DSPREAD
    log_per_prototype = erlang - pt.logsumexp(erlang) - np.log(100.0)

    log_rules = (log_w[:6] - np.log(FAMILY_SIZE))[RULE_FAMILY][None, :] + rule_log_lik
    log_spreads = log_w[6] + log_per_prototype[None, :] + spread_log_evidence
    log_thresholds = log_w[7] - np.log(N_THRESHOLD) + threshold_log_evidence
    log_joint = pt.concatenate([log_rules, log_spreads, log_thresholds], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (
        posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)
        + (posterior[:, N_RULES:-1, None] * spread_in_concept).sum(axis=1)
        + posterior[:, -1, None] * threshold_in_concept
    )  # (sets, targets)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _log_lik(examples, production):
    """Log P(examples | concept) for every concept (rows of `production`, the
    probability that the program emits each number), 10% of examples random."""
    return np.log(ALPHA * production[:, np.array(examples) - 1] + (1 - ALPHA) / 100).sum(axis=1)


def _group(log_lik, membership):
    """Log summed evidence of a group of equally likely concepts and its membership of each target."""
    top = log_lik.max(axis=1, keepdims=True)
    weight = np.exp(log_lik - top)
    total = weight.sum(axis=1)
    return np.log(total) + top[:, 0], (weight / total[:, None]) @ membership


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    rule_production = RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True)
    spread_log_evidence = np.zeros((len(unique), len(SPREADS)))
    spread_in_concept = np.zeros((len(unique), len(SPREADS), 100))
    for i, proto in enumerate(PROTOTYPES):
        production = proto / proto.sum(axis=1, keepdims=True)
        log_lik = np.stack([_log_lik(s, production) for s in unique])
        spread_log_evidence[:, i], spread_in_concept[:, i] = _group(log_lik, proto)
    threshold_production = THRESHOLD_MEMBERSHIP / THRESHOLD_MEMBERSHIP.sum(axis=1, keepdims=True)
    threshold_log_lik = np.stack([_log_lik(s, threshold_production) for s in unique])
    threshold_log_evidence, threshold_in_concept = _group(threshold_log_lik, THRESHOLD_MEMBERSHIP)
    return {
        "rule_log_lik": np.stack([_log_lik(s, rule_production) for s in unique]),
        "spread_log_evidence": spread_log_evidence,
        "spread_in_concept": spread_in_concept,
        "threshold_log_evidence": threshold_log_evidence[:, None],
        "threshold_in_concept": threshold_in_concept,
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
