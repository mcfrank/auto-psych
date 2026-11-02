"""Two-rule disjunctions instead of noise.

People infer the hidden program as a Bayesian concept learner over number rules
(parity, primes, squares, cubes, powers, multiples, final and contained digits,
digit sums, exception rules) and numeric ranges with an Erlang prior on width,
with the size principle and a prior mass per concept family, but they take
every example as a genuine output of the program: only a negligible, fixed
0.1% of examples are allowed to be random slips (the established models use
10%). An example that breaks the rule the others share is instead explained by
disjunctive programs joining two simple rules ("ends in 2 or ends in 7",
"squares or multiples of 5"), a family with its own inferred prior mass shared
equally among the distinct unions, so people endorse targets satisfying either
rule rather than only the majority rule.
"""

import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.999  # examples are almost never random slips
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
    rules.update({f"digit sum {t}": [n for n in NUMBERS if sum(map(int, str(n))) == t] for t in range(1, 19)})
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
    if name.startswith("digit sum"):
        return 6
    if name.startswith("powers of 2,"):
        return 5
    return 1


def log_lik_table(membership):
    return np.log(ALPHA * membership / membership.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)


_RULES = rule_concepts()
NAMES = list(_RULES)
N_RULES = len(NAMES)
RULE_FAMILY = np.array([family(n) for n in NAMES])
FAMILY_SIZE = np.bincount(RULE_FAMILY).astype("float64")
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, _RULES[n]) for n in NAMES], dtype="float64")
RULE_LOG_LIK = log_lik_table(RULE_MEMBERSHIP)

# Disjunctions of two distinct simple rules (exception rules excluded); a union
# identical to one of its parts or to an earlier union is kept once.
_simple = [i for i in range(N_RULES) if RULE_FAMILY[i] != 5]
_seen = {tuple(RULE_MEMBERSHIP[i].astype(bool)) for i in range(N_RULES)}
_unions = []
for i, j in itertools.combinations(_simple, 2):
    u = np.maximum(RULE_MEMBERSHIP[i], RULE_MEMBERSHIP[j])
    key = tuple(u.astype(bool))
    if key not in _seen:
        _seen.add(key)
        _unions.append(u)
UNION_MEMBERSHIP = np.array(_unions)
UNION_LOG_LIK = log_lik_table(UNION_MEMBERSHIP)
N_UNIONS = len(_unions)

LO, HI = np.array([(m, n) for m in NUMBERS for n in range(m, 101)]).T
WIDTH = HI - LO + 1
INTERVAL_MEMBERSHIP = ((NUMBERS[None, :] >= LO[:, None]) & (NUMBERS[None, :] <= HI[:, None])).astype("float64")
INTERVAL_LOG_LIK = log_lik_table(INTERVAL_MEMBERSHIP)
WIDTHS = np.arange(1, 101, dtype="float64")
N_PER_WIDTH = 101.0 - WIDTHS


def set_summaries(s):
    idx = np.array(s) - 1
    rule_ll = RULE_LOG_LIK[:, idx].sum(axis=1)
    ull = UNION_LOG_LIK[:, idx].sum(axis=1)
    union_logz = logsumexp(ull) - np.log(N_UNIONS)
    union_mem = np.exp(ull - logsumexp(ull)) @ UNION_MEMBERSHIP
    ill = INTERVAL_LOG_LIK[:, idx].sum(axis=1)
    width_logz = np.full(100, -np.inf)
    width_mem = np.zeros((100, 100))
    for w in range(1, 101):
        sel = WIDTH == w
        lz = logsumexp(ill[sel])
        width_logz[w - 1] = lz - np.log(sel.sum())  # mean likelihood of a width-w range
        width_mem[w - 1] = np.exp(ill[sel] - lz) @ INTERVAL_MEMBERSHIP[sel]
    return rule_ll, union_logz, union_mem, width_logz, width_mem


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    parts = [set_summaries(s) for s in unique]
    return {
        "rule_log_lik": np.stack([p[0] for p in parts]),
        "union_log_z": np.array([p[1] for p in parts]),
        "union_member": np.stack([p[2] for p in parts]),
        "width_log_z": np.stack([p[3] for p in parts]),
        "width_member": np.stack([p[4] for p in parts]),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    union_log_z = pm.Data("union_log_z", np.zeros(1))  # (sets,) mean likelihood over unions
    union_member = pm.Data("union_member", np.zeros((1, 100)))  # (sets, targets)
    width_log_z = pm.Data("width_log_z", np.zeros((1, 100)))  # (sets, widths)
    width_member = pm.Data("width_member", np.zeros((1, 100, 100)))  # (sets, widths, targets)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Prior mass of the seven rule families, of two-rule disjunctions, and of ranges.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(9))
    log_w = pt.log(family_weight)
    interval_scale = pm.LogNormal("interval_scale", mu=np.log(20.0), sigma=1.0)

    # Erlang prior mass of each width (shared by the ranges of that width).
    erlang = np.log(WIDTHS) - WIDTHS / interval_scale + np.log(N_PER_WIDTH)
    log_width_prior = erlang - pt.logsumexp(erlang)

    log_rules = (log_w[:7][RULE_FAMILY] - np.log(FAMILY_SIZE)[RULE_FAMILY])[None, :] + rule_log_lik
    log_union = (log_w[7] + union_log_z)[:, None]
    log_widths = log_w[8] + log_width_prior[None, :] + width_log_z
    log_joint = pt.concatenate([log_rules, log_union, log_widths], axis=1)
    post = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    in_concept = (
        post[:, :N_RULES] @ pt.as_tensor(RULE_MEMBERSHIP)
        + post[:, N_RULES : N_RULES + 1] * union_member
        + pt.sum(post[:, N_RULES + 1 :, None] * width_member, axis=1)
    )

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept.reshape((-1,))[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
