"""Bayesian concept learning with a family-weighted rule prior (exception rules
in their own family) and an Erlang prior over interval width: a refinement of
`exception_rule_family_prior_2`.

The learner considers 45 rules and the 5,050 intervals of 1..100, samples
examples from the concept (size principle, 10% uniformly from 1..100), and
answers "yes" with the posterior probability that the concept contains the
target. Rule families get inferred prior masses as in the incumbent. The one
change: intervals are not equally likely a priori — each interval's prior is
proportional to an Erlang density of its width, (L / s^2) exp(-L / s), with the
typical size s inferred, so people expect magnitude concepts of a characteristic
moderate span rather than any span at all.
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
    return 1  # primes, squares, cubes, powers


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_FAMILY = np.array([family(n) for n in RULE_NAMES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
INTERVALS = [(m, n) for m in NUMBERS for n in range(m, 101)]
INTERVAL_WIDTH = np.array([n - m + 1 for m, n in INTERVALS])
WIDTHS = np.arange(1, 101, dtype="float64")
N_PER_WIDTH = 101.0 - WIDTHS  # number of intervals of each width
MEMBERSHIP = np.array(
    [np.isin(NUMBERS, c) for c in RULES] + [(NUMBERS >= m) & (NUMBERS <= n) for m, n in INTERVALS],
    dtype="float64",
)
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    width_log_evidence = pm.Data("width_log_evidence", np.zeros((1, 100)))  # (sets, widths): log sum over intervals of that width
    width_in_concept = pm.Data("width_in_concept", np.zeros((1, 100)))  # (rows, widths): P(target in interval | data, width)
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    row_target = pm.Data("row_target", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    family_weight = pm.Dirichlet("family_weight", a=np.ones(7))
    log_w = pt.log(family_weight)
    # Typical interval width (Erlang scale), on the 1..100 number line.
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(10.0), sigma=1.0)

    log_erlang = pt.log(WIDTHS) - WIDTHS / interval_scale  # per-interval, unnormalised
    log_z = pt.logsumexp(np.log(N_PER_WIDTH) + log_erlang)
    log_width_prior = log_w[6] + log_erlang - log_z  # per interval of each width

    log_rules = (log_w[:6] - np.log(FAMILY_SIZE))[RULE_FAMILY][None, :] + rule_log_lik
    log_widths = log_width_prior[None, :] + width_log_evidence
    log_joint = pt.concatenate([log_rules, log_widths], axis=1)  # (sets, rules + widths)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))

    rule_in = (posterior[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP))[row_set, row_target]
    interval_in = pt.sum(posterior[row_set, N_RULES:] * width_in_concept, axis=1)
    p_yes = pm.Deterministic("p_yes", pt.clip(rule_in + interval_in, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    interval_ll = np.stack([EXAMPLE_LOG_LIK[N_RULES:, np.array(s) - 1].sum(axis=1) for s in unique])
    rule_ll = np.stack([EXAMPLE_LOG_LIK[:N_RULES, np.array(s) - 1].sum(axis=1) for s in unique])
    width_ev = np.zeros((len(unique), 100))
    width_in = np.zeros((len(unique), 100, 100))  # (sets, widths, targets)
    for w in range(1, 101):
        sel = INTERVAL_WIDTH == w
        ll = interval_ll[:, sel]
        ev = logsumexp(ll, axis=1)
        width_ev[:, w - 1] = ev
        width_in[:, w - 1, :] = np.exp(ll - ev[:, None]) @ INTERVAL_MEMBERSHIP[sel]
    set_idx = np.array([index[s] for s in sets], dtype="int64")
    tgt = np.array([int(r["target"]) - 1 for r in rows], dtype="int64")
    return {
        "rule_log_lik": rule_ll,
        "width_log_evidence": width_ev,
        "width_in_concept": width_in[set_idx, :, tgt],
        "row_set": set_idx,
        "row_target": tgt,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
