"""Refinement of uncertain_reliability_noise_concepts: Bayesian concept learning
over 45 rules (each with its own inferred prior weight, a share lambda_0 for
the rules as a group) and every interval of 1-100, with the size principle and
an uncertain belief (Beta, mean alpha, concentration kappa) about how reliable
the program is on each set, integrated out per set -- but a stray example is
not only a uniform draw from 1-100: with inferred probability eps it is a
slip, a member of the concept miswritten -- one digit changed, added or
dropped, or two digits swapped -- that lands outside the concept (uniform over such numbers).
So an example that is a slip of a tight concept's member (83 beside 38, 58, 68
"ends in 8"; 27 beside 92, 82, 42) is easily explained and the tight concept
is kept, while an example no slip of a member explains counts against it as
before.

For an example set and concept h, every example is genuine (1[x in h] / s),
a uniform stray (1/100) or a slip (N_h(x)). The product over examples is
a polynomial whose coefficients C[g, t] (g genuine, t slips) depend only on
the stimulus, so prepare_observed precomputes them (summed over intervals per
set and per (set, target)), and
p(X | h) = sum_{g,t} C[g,t] eps^t (1-eps)^(n-g-t) B(u+g, v+n-g) / B(u, v).
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4


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
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
INT_LO = np.array([m for m in NUMBERS for n in range(m, 101)])
INT_HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INT_MEMBERSHIP = ((NUMBERS[None, :] >= INT_LO[:, None]) & (NUMBERS[None, :] <= INT_HI[:, None])).astype("float64")

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0


def is_slip(a, b):
    """b is a miswriting of a: one digit changed, added or dropped, or two swapped."""
    sa, sb = str(a), str(b)
    if sa == sb:
        return False
    if len(sa) == len(sb):
        diff = [i for i in range(len(sa)) if sa[i] != sb[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1
                                  and sa[diff[0]] == sb[diff[1]] and sa[diff[1]] == sb[diff[0]])
    short, long_ = (sa, sb) if len(sa) < len(sb) else (sb, sa)
    return len(long_) == len(short) + 1 and any(long_[:i] + long_[i + 1:] == short for i in range(len(long_)))


SLIP = np.array([[1.0 if is_slip(a, b) else 0.0 for b in NUMBERS] for a in NUMBERS])


def slip_density(member):
    """Slip stray density over the numbers outside each concept (rows of member)."""
    kernel = SLIP
    mass = (member @ kernel) * (1.0 - member)
    z = mass.sum(axis=1, keepdims=True)
    return np.where(z > 0, mass / np.maximum(z, 1e-300), 0.0)


def example_coefficients(member, near, ex):
    """C[h, g, t]: summed product terms with g genuine and t slip examples."""
    size = member.sum(axis=1)
    coef = np.zeros((member.shape[0], MAX_N + 1, MAX_N + 1))
    coef[:, 0, 0] = 1.0
    for x in ex:
        genuine, nearby = member[:, x - 1] / size, near[:, x - 1]
        new = coef * 0.01
        new[:, 1:, :] += coef[:, :-1, :] * genuine[:, None, None]
        new[:, :, 1:] += coef[:, :, :-1] * nearby[:, None, None]
        coef = new
    return coef


RULE_NEAR = slip_density(RULE_MEMBERSHIP)
INT_NEAR = slip_density(INT_MEMBERSHIP)

# Valid (n, g, t) cells and their binomial-free exponent of (1 - eps).
_N = np.arange(1, MAX_N + 1)[:, None, None]
_G = np.arange(MAX_N + 1)[None, :, None]
_T = np.arange(MAX_N + 1)[None, None, :]
_VALID = (_G + _T <= _N).astype("float64")
_UNI = np.maximum(_N - _G - _T, 0).astype("float64")
_G_F = (_G + 0 * _N + 0 * _T).astype("float64")
_T_F = (_T + 0 * _N + 0 * _G).astype("float64")
_REST = np.maximum(_N - _G, 0) + 0 * _T


def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


with pm.Model() as model:
    rule_coef = pm.Data("rule_coef", np.zeros((1, N_RULES, MAX_N + 1, MAX_N + 1)))  # (sets, rules, g, t)
    set_n_idx = pm.Data("set_n_idx", np.zeros(1, dtype="int64"))  # (sets,): n - 1
    interval_coef = pm.Data("interval_coef", np.zeros((1, MAX_N + 1, MAX_N + 1)))  # (sets, g, t)
    cell_coef = pm.Data("cell_coef", np.zeros((1, MAX_N + 1, MAX_N + 1)))  # (cells, g, t): containing target
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa
    # Share of strays that are miswritten members of the concept rather than uniform.
    eps_logit = pm.Normal("eps_logit", mu=0.0, sigma=1.5)
    eps = pm.Deterministic("eps", pm.math.sigmoid(eps_logit))

    # Weight of each (n, g, t) coefficient.
    weight = _VALID * pt.exp(
        _T_F * pt.log(eps) + _UNI * pt.log1p(-eps)
        + _betaln(u + _G_F, v + _REST) - _betaln(u, v)
    )  # (n, g, t)
    set_w = weight[set_n_idx]  # (sets, g, t)

    rule_lik = (rule_coef * set_w[:, None, :, :]).sum(axis=(2, 3))  # (sets, rules)
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + pt.log(rule_lik)
    int_ev = (interval_coef * set_w).sum(axis=(1, 2))
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + pt.log(int_ev)

    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])
    interval_post = pt.exp(log_intervals - log_norm)

    cell_int_share = (cell_coef * set_w[cell_set]).sum(axis=(1, 2)) / int_ev[cell_set]
    in_concept = (rule_post[cell_set] * cell_rule_member).sum(axis=1) + interval_post[cell_set] * cell_int_share
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-9, 1 - 1e-9)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    S = len(unique)
    rule_coef = np.zeros((S, N_RULES, MAX_N + 1, MAX_N + 1))
    interval_coef = np.zeros((S, MAX_N + 1, MAX_N + 1))
    cell_coef = np.zeros((len(cells), MAX_N + 1, MAX_N + 1))
    set_n_idx = np.zeros(S, dtype="int64")
    cells_of = {}
    for j, (i, t) in enumerate(cells):
        cells_of.setdefault(i, []).append((j, t))
    for i, s in enumerate(unique):
        set_n_idx[i] = len(s) - 1
        rule_coef[i] = example_coefficients(RULE_MEMBERSHIP, RULE_NEAR, s)
        int_c = example_coefficients(INT_MEMBERSHIP, INT_NEAR, s)
        interval_coef[i] = int_c.sum(axis=0)
        for j, t in cells_of.get(i, []):
            cell_coef[j] = int_c[INT_MEMBERSHIP[:, t - 1] > 0].sum(axis=0)
    return {
        "rule_coef": rule_coef,
        "set_n_idx": set_n_idx,
        "interval_coef": interval_coef,
        "cell_coef": cell_coef,
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
