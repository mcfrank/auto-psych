"""Best-explanation contrast: people do not average over every concept that
could explain the examples. They compare the single most plausible concept
(rule or magnitude range, same family-organised rule prior, beta-share range
prior and size principle as the Bayesian learner) that includes the target with
the single most plausible concept that excludes it, and say "yes" with a
probability that grows (with an inferred gain) with how much better the best
target-including explanation is than the best target-excluding one. (The maximum is smoothed at a fixed, sharp temperature
so the posterior is differentiable; it is not averaging over concepts.)
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
NONE = -1e3  # log score when no concept of that group is available
SHARP = 4.0  # fixed sharpness of the (smoothed) maximum: only the best explanations count


def soft_max(x, axis):
    return pt.logsumexp(SHARP * x, axis=axis) / SHARP


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


def rule_family(name):
    if name in ("odd", "even"):
        return 0
    if name == "primes":
        return 1
    if name in ("squares", "cubes"):
        return 2
    if name.startswith("powers of 2,"):
        return 7
    if name.startswith("powers"):
        return 3
    if name.startswith("multiples"):
        return 4
    if name.startswith("ends in"):
        return 5
    return 6


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_EX_LL = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
FAMILY = np.array([rule_family(n) for n in RULE_NAMES])
N_FAMILIES = FAMILY.max() + 1
FAMILY_SIZE = np.bincount(FAMILY)

# Intervals sorted by size, so per-size maxima are reduceat over contiguous blocks.
_IV = [(a, a + L - 1) for L in range(1, 101) for a in range(1, 102 - L)]
IV_LO = np.array([a for a, _ in _IV])
IV_HI = np.array([b for _, b in _IV])
IV_SIZE = IV_HI - IV_LO + 1
SIZE_START = np.searchsorted(IV_SIZE, np.arange(1, 101))
IV_MEMBER = (NUMBERS[None, :] >= IV_LO[:, None]) & (NUMBERS[None, :] <= IV_HI[:, None])
IV_EX_LL = np.log(ALPHA * IV_MEMBER / IV_SIZE[:, None] + (1 - ALPHA) / 100)
SIZES = np.arange(1, 101, dtype="float64")
LOG_N_OF_SIZE = np.log(101.0 - SIZES)
SHARE = (SIZES - 0.5) / 100.0


with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    size_ll_in = pm.Data("size_ll_in", np.zeros((1, 100)))  # best interval of each size containing target
    size_ll_out = pm.Data("size_ll_out", np.zeros((1, 100)))  # best interval of each size excluding it
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    mean_share = pm.Beta("mean_share", 2, 3)
    concentration = pm.LogNormal("concentration", mu=np.log(5.0), sigma=1.0)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    gain = pm.LogNormal("gain", mu=np.log(0.5), sigma=1.0)

    rule_log_prior = pt.log(lambda_0) + pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])
    log_size_prior = ((mean_share * concentration - 1.0) * np.log(SHARE)
                      + ((1.0 - mean_share) * concentration - 1.0) * np.log1p(-SHARE))
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior)
    log_per_interval = pt.log1p(-lambda_0) + log_size_prior - LOG_N_OF_SIZE

    cell_rules = (rule_log_prior[None, :] + rule_log_lik)[cell_set]
    best_rule_in = soft_max(pt.where(cell_rule_member > 0.5, cell_rules, NONE), axis=1)
    best_rule_out = soft_max(pt.where(cell_rule_member > 0.5, NONE, cell_rules), axis=1)
    best_iv_in = soft_max(log_per_interval[None, :] + size_ll_in, axis=1)
    best_iv_out = soft_max(log_per_interval[None, :] + size_ll_out, axis=1)
    best_in = soft_max(pt.stack([best_rule_in, best_iv_in]), axis=0)
    best_out = soft_max(pt.stack([best_rule_out, best_iv_out]), axis=0)

    p_cell = pm.math.sigmoid(gain * (best_in - best_out))
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_s = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")

    set_rule_ll = np.stack([RULE_EX_LL[:, np.array(s) - 1].sum(axis=1) for s in unique])
    ll_in = np.full((len(unique), 100, 100), NONE)  # (set, target, size)
    ll_out = np.full((len(unique), 100, 100), NONE)
    for i, s in enumerate(unique):
        iv_ll = IV_EX_LL[:, np.array(s) - 1].sum(axis=1)
        m_in = np.where(IV_MEMBER, iv_ll[:, None], NONE)
        m_out = np.where(IV_MEMBER, NONE, iv_ll[:, None])
        ll_in[i] = np.maximum.reduceat(m_in, SIZE_START, axis=0).T
        ll_out[i] = np.maximum.reduceat(m_out, SIZE_START, axis=0).T
    return {
        "rule_log_lik": set_rule_ll,
        "cell_set": cell_s,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_t].T.copy(),
        "size_ll_in": ll_in[cell_s, cell_t],
        "size_ll_out": ll_out[cell_s, cell_t],
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
