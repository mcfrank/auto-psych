"""Refinement of parity_within_range_concepts: Bayesian concept learning over
45 rules (family-organised prior; "powers of 2 except 32"/"plus 37" are their
own kind), all intervals of 1..100 with an Erlang prior over width, "odd (or
even) numbers within a range", examples drawn from the concept with probability
0.9 (size principle) and from 1..100 otherwise, "yes" = posterior probability
that the concept contains the target.
The single change: the conjunctive "rule within a range" idea extends beyond
parity — people also entertain "multiples of m within a range" (m = 3..9, 11,
12) as a fourth kind of concept with its own inferred plausibility and the same
width preference, so after 39,60,12,45 multiples of 3 near the examples are
accepted but multiples far outside their span are rejected.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
MULTS = [3, 4, 5, 6, 7, 8, 9, 11, 12]


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
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in MULTS})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(10)})
    return rules


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
_LO, _HI = np.triu_indices(100)  # interval [lo+1, hi+1]
N_INT = len(_LO)
INTERVAL_MEMBERSHIP_BASE = (NUMBERS[None, :] >= _LO[:, None] + 1) & (NUMBERS[None, :] <= _HI[:, None] + 1)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
# range kinds: plain (1 block), parity (2 blocks), multiples (len(MULTS) blocks)
_FILTERS = ([np.ones(100, bool)] + [NUMBERS % 2 == p for p in (1, 0)]
            + [NUMBERS % m == 0 for m in MULTS])
_KIND_OF_BLOCK = np.array([0, 1, 1] + [2] * len(MULTS))
INTERVAL_MEMBERSHIP = np.concatenate([INTERVAL_MEMBERSHIP_BASE & f[None, :] for f in _FILTERS]).astype("float64")
N_RANGE_KINDS = 3
N_SIZES = 100 * N_RANGE_KINDS
_SPAN = np.tile(_HI - _LO, len(_FILTERS))
INTERVAL_SIZE_IDX = _SPAN + 100 * np.repeat(_KIND_OF_BLOCK, N_INT)


def _log_lik(membership):
    n = membership.sum(axis=1, keepdims=True)
    ll = np.log(ALPHA * membership / np.maximum(n, 1) + (1 - ALPHA) / 100)
    ll[n[:, 0] == 0] = -1e4  # empty concepts are impossible
    return ll


RULE_LOG_LIK = _log_lik(RULE_MEMBERSHIP)
INTERVAL_LOG_LIK = _log_lik(INTERVAL_MEMBERSHIP)
SIZE_GROUPS = [np.flatnonzero(INTERVAL_SIZE_IDX == L) for L in range(N_SIZES)]
SIZES = np.arange(1, 101, dtype="float64")
LOG_N_OF_SIZE = np.log(101.0 - SIZES)  # number of intervals of each size


def rule_family(name):
    if name.startswith("powers of 2,"):  # a rule with an arbitrary exception
        return 7
    if name in ("odd", "even"):
        return 0
    if name == "primes":
        return 1
    if name in ("squares", "cubes"):
        return 2
    if name.startswith("powers"):
        return 3
    if name.startswith("multiples"):
        return 4
    if name.startswith("ends in"):
        return 5
    return 6


FAMILY = np.array([rule_family(n) for n in RULE_NAMES])
N_FAMILIES = FAMILY.max() + 1
FAMILY_SIZE = np.bincount(FAMILY)

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    size_log_evidence = pm.Data("size_log_evidence", np.zeros((1, N_SIZES)))  # (sets, size classes)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_size_member = pm.Data("cell_size_member", np.zeros((1, N_SIZES)))  # (cells, size classes)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # rules, ranges, odd/even within a range, multiples of m within a range
    kind = pm.Dirichlet("kind_weights", a=np.full(4, 2.0))
    sigma = pm.LogNormal("sigma", mu=np.log(15.0), sigma=1.0)  # Erlang scale of interval width
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    rule_log_prior = pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])

    log_size_prior = np.log(SIZES) - SIZES / sigma
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior)
    per_interval = log_size_prior - LOG_N_OF_SIZE
    log_per_interval = pt.concatenate([pt.log(kind[1]) + per_interval,
                                       pt.log(kind[2]) + per_interval - np.log(2.0),
                                       pt.log(kind[3]) + per_interval - np.log(len(MULTS))])

    log_rules = pt.log(kind[0]) + rule_log_prior[None, :] + rule_log_lik
    log_sizes = log_per_interval[None, :] + size_log_evidence
    log_joint = pt.concatenate([log_rules, log_sizes], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_post = posterior[cell_set]
    in_concept = ((cell_post[:, :N_RULES] * cell_rule_member).sum(axis=1)
                  + (cell_post[:, N_RULES:] * cell_size_member).sum(axis=1))

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    rule_ll = np.stack([RULE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    size_ev = np.zeros((n_sets, N_SIZES))
    size_member = np.zeros((n_sets, N_SIZES, 100))
    for i, s in enumerate(unique):
        ll = INTERVAL_LOG_LIK[:, np.array(s) - 1].sum(axis=1)
        for L, sel in enumerate(SIZE_GROUPS):
            ev = logsumexp(ll[sel])
            size_ev[i, L] = ev
            size_member[i, L] = np.exp(ll[sel] - ev) @ INTERVAL_MEMBERSHIP[sel]
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_s = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    return {
        "rule_log_lik": rule_ll,
        "size_log_evidence": size_ev,
        "cell_set": cell_s,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_t].T.copy(),
        "cell_size_member": np.clip(size_member[cell_s, :, cell_t], 0.0, 1.0),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
