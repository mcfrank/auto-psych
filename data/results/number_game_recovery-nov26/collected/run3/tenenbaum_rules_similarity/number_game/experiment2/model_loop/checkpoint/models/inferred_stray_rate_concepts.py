"""Refinement of gamma_width_exception_family_concepts: the same Bayesian concept
learning over 45 rules (family-organised prior, exception rules as their own
family) and all intervals of 1..100 (gamma prior over interval width with shape
and mean inferred), with the size principle. The single change: how much people
believe the program's outputs are random rather than drawn from the concept is
no longer fixed at 10% but inferred. Each example is taken to come from the
concept (uniformly) with probability alpha and from 1..100 at random otherwise,
so a lower alpha lets concepts that miss an example survive (and weakens the
size principle), a higher alpha makes generalisation stricter.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4  # most examples in a set


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
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
SIZES = np.arange(1, 101, dtype="float64")
LOG_N_OF_SIZE = np.log(101.0 - SIZES)  # number of intervals of each size
N_GROUPS = 100 * (MAX_N + 1)  # interval groups (size L, examples inside m)
GROUP_SIZE = np.repeat(SIZES, MAX_N + 1)
GROUP_M = np.tile(np.arange(MAX_N + 1, dtype="float64"), 100)


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


FAMILY = np.array([rule_family(n) for n in RULE_NAMES])
N_FAMILIES = FAMILY.max() + 1
FAMILY_SIZE = np.bincount(FAMILY)

# A concept's likelihood depends only on its size and on how many examples it
# contains, so prepare_observed groups each set's intervals by (size, examples
# inside) and records group counts and, per target, the share containing it.

with pm.Model() as model:
    set_n = pm.Data("set_n", np.zeros(1))  # (sets,) number of examples
    rule_m = pm.Data("rule_m", np.zeros((1, N_RULES)))  # (sets, rules) examples inside
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_group_member = pm.Data("cell_group_member", np.zeros((1, N_GROUPS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    alpha = pm.Beta("alpha", 9.0, 1.0)  # probability an example comes from the concept
    lambda_0 = pm.Beta("lambda_0", 2, 2)
    mean_width = pm.LogNormal("mean_width", mu=np.log(30.0), sigma=1.0)
    shape = pm.LogNormal("shape", mu=np.log(2.0), sigma=0.75)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    rule_log_prior = pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])

    log_out = pt.log1p(-alpha) - np.log(100.0)  # example from outside the concept

    def log_lik(size, m):
        log_in = pt.log(alpha / size + (1.0 - alpha) / 100.0)
        return m * log_in + (set_n[:, None] - m) * log_out

    log_size_prior = (shape - 1.0) * np.log(SIZES) - SIZES * shape / mean_width
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior)
    log_per_interval = log_size_prior - LOG_N_OF_SIZE
    log_group_prior = pt.repeat(log_per_interval, MAX_N + 1)

    log_rules = pt.log(lambda_0) + rule_log_prior[None, :] + log_lik(RULE_SIZE[None, :], rule_m)
    log_groups = (pt.log1p(-lambda_0) + log_group_prior[None, :] + group_log_count
                  + log_lik(GROUP_SIZE[None, :], GROUP_M[None, :]))
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    cell_post = posterior[cell_set]
    in_concept = ((cell_post[:, :N_RULES] * cell_rule_member).sum(axis=1)
                  + (cell_post[:, N_RULES:] * cell_group_member).sum(axis=1))

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    lo = NUMBERS[:, None]  # interval start
    hi = NUMBERS[None, :]  # interval end
    valid = hi >= lo
    size_idx = (hi - lo).clip(0)
    n_sets = len(unique)
    group_count = np.zeros((n_sets, N_GROUPS))
    group_member = np.zeros((n_sets, N_GROUPS, 100))
    rule_m = np.zeros((n_sets, N_RULES))
    for i, s in enumerate(unique):
        ex = np.array(s)
        rule_m[i] = RULE_MEMBERSHIP[:, ex - 1].sum(axis=1)
        m = ((lo[..., None] <= ex) & (hi[..., None] >= ex)).sum(axis=2)
        g = (size_idx * (MAX_N + 1) + m)[valid]
        st, en = np.broadcast_to(lo, valid.shape)[valid], np.broadcast_to(hi, valid.shape)[valid]
        group_count[i] = np.bincount(g, minlength=N_GROUPS)
        for t in range(1, 101):
            inside = (st <= t) & (en >= t)
            group_member[i, :, t - 1] = np.bincount(g[inside], minlength=N_GROUPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(group_count[..., None] > 0, group_member / np.maximum(group_count[..., None], 1), 0.0)
    log_count = np.where(group_count > 0, np.log(np.maximum(group_count, 1)), -1e4)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_s = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    return {
        "set_n": np.array([len(s) for s in unique], dtype="float64"),
        "rule_m": rule_m,
        "group_log_count": log_count,
        "cell_set": cell_s,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_t].T.copy(),
        "cell_group_member": share[cell_s, :, cell_t],
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
