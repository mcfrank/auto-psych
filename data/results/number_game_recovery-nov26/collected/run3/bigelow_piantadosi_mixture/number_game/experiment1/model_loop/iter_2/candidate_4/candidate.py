"""Refinement of inferred_example_noise_concepts: Bayesian concept learning over
the 45 Bigelow-Piantadosi rules and every interval of 1-100, with the size
principle, an inferred probability alpha that an example comes from the
concept (else uniformly from 1-100), and "yes" = the posterior probability that
the concept contains the target. The one change is the prior over rules: rather
than an independent weight for every rule, people's prior plausibility of a
rule is organised by rule family (parity, primes, squares and cubes, powers,
multiples, final digit, contained digit) -- each rule's log prior weight is drawn
around its family's level with an inferred spread tau (partial pooling), so
rules of the same kind are similarly plausible unless the data say otherwise.
A share lambda_0 goes to the rules as a group; each interval has prior
(1 - lambda_0) / 100^2.

Under a hypothesis of size s, a set of n examples of which k fall in the
concept has likelihood (alpha/s + (1-alpha)/100)^k ((1-alpha)/100)^(n-k), so
intervals are grouped by (k, s): prepare_observed counts, per set, how many
intervals have each (k, s) and, per response, how many of those contain the
target. The model then needs only these counts, exactly.
"""

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
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
INT_LO = np.array([m for m in NUMBERS for n in range(m, 101)])
INT_HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INT_SIZE_IDX = INT_HI - INT_LO  # size - 1
SIZES = np.arange(1, 101, dtype="float64")

FAMILIES = ["parity", "primes", "squares and cubes", "powers", "multiples", "ends in", "contains digit"]


def family_of(name):
    if name in ("odd", "even"):
        return "parity"
    if name in ("squares", "cubes"):
        return "squares and cubes"
    for f in ("primes", "powers", "multiples", "ends in", "contains digit"):
        if name.startswith(f):
            return f
    raise ValueError(name)


RULE_FAMILY = np.array([FAMILIES.index(family_of(r)) for r in RULE_NAMES])
FAMILY_SIZE = np.bincount(RULE_FAMILY, minlength=len(FAMILIES)).astype("float64")

with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples in rule
    set_n = pm.Data("set_n", np.ones(1))  # (sets,): number of examples
    interval_log_count = pm.Data("interval_log_count", np.zeros((1, MAX_N + 1, 100)))  # (sets, k, size)
    # Computed once per distinct (set, target) cell, then indexed per response.
    cell_count = pm.Data("cell_count", np.zeros((1, MAX_N + 1, 100)))  # (cells, k, size): containing target
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    # Each rule's prior log-weight is drawn around its family's level, with an
    # inferred spread (non-centred): tau = 0 gives equal weights within a family.
    family_level = pm.Normal("family_level", mu=0.0, sigma=1.5, shape=len(FAMILIES))
    tau = pm.HalfNormal("tau", sigma=1.0)
    rule_dev = pm.Normal("rule_dev", mu=0.0, sigma=1.0, shape=N_RULES)
    rule_weights = pm.Deterministic(
        "rule_weights", pt.special.softmax(family_level[RULE_FAMILY] + tau * rule_dev)
    )
    # Probability an example is drawn from the concept (incumbent fixes 0.9).
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_noise = pt.log1p(-alpha) - np.log(100)

    n = set_n[:, None]
    log_in_rule = pt.log(alpha / RULE_SIZE + (1 - alpha) / 100)[None, :]
    rule_ll = rule_k * log_in_rule + (n - rule_k) * log_noise
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_ll  # (sets, rules)

    ks = np.arange(MAX_N + 1, dtype="float64")
    log_in_int = pt.log(alpha / SIZES + (1 - alpha) / 100)  # (size,)
    int_ll = ks[None, :, None] * log_in_int[None, None, :] + (n[:, :, None] - ks[None, :, None]) * log_noise
    int_terms = interval_log_count + int_ll  # (sets, k, size); -inf-like where empty
    int_log_ev = pt.logsumexp(int_terms.reshape((int_terms.shape[0], -1)), axis=1)  # (sets,)
    log_intervals = pt.log1p(-lambda_0) - 2 * np.log(100) + int_log_ev

    log_joint = pt.concatenate([log_rules, log_intervals[:, None]], axis=1)
    log_norm = pt.logsumexp(log_joint, axis=1)
    rule_post = pt.exp(log_rules - log_norm[:, None])  # (sets, rules)
    interval_post = pt.exp(log_intervals - log_norm)  # (sets,)

    # Share of interval evidence from intervals containing the target, factored
    # as sum_k exp((n - k) log_noise - evidence) * sum_s count[k, s] exp(k log_in_int[s])
    # so only a (cells, k) table is gathered per cell.
    in_k = pt.exp(ks[:, None] * log_in_int[None, :])  # (k, size)
    block = (in_k[:, :, None] * np.eye(MAX_N + 1)[:, None, :]).reshape(((MAX_N + 1) * 100, MAX_N + 1))
    cell_k_sum = cell_count.reshape((cell_count.shape[0], -1)) @ block  # (cells, k)
    set_k_scale = pt.exp((n - ks[None, :]) * log_noise - int_log_ev[:, None])  # (sets, k)
    cell_int_share = (cell_k_sum * set_k_scale[cell_set]).sum(axis=1)
    in_concept = (rule_post[cell_set] * cell_rule_member).sum(axis=1) + interval_post[cell_set] * cell_int_share
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-9, 1 - 1e-9)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_k = np.zeros((S, N_RULES))
    set_n = np.zeros(S)
    int_k = np.zeros((S, len(INT_LO)), dtype="int64")
    log_count = np.full((S, MAX_N + 1, 100), -1e4)
    for i, s in enumerate(unique):
        ex = np.array(s)
        set_n[i] = len(ex)
        rule_k[i] = RULE_MEMBERSHIP[:, ex - 1].sum(axis=1)
        int_k[i] = ((ex[None, :] >= INT_LO[:, None]) & (ex[None, :] <= INT_HI[:, None])).sum(axis=1)
        counts = np.zeros((MAX_N + 1, 100))
        np.add.at(counts, (int_k[i], INT_SIZE_IDX), 1.0)
        log_count[i] = np.where(counts > 0, np.log(np.maximum(counts, 1)), -1e4)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_count = np.zeros((len(cells), MAX_N + 1, 100))
    cell_set = np.array([i for i, _ in cells], dtype="int64")
    cell_rule_member = np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells])
    for j, (i, t) in enumerate(cells):
        contains = (INT_LO <= t) & (t <= INT_HI)
        np.add.at(cell_count[j], (int_k[i][contains], INT_SIZE_IDX[contains]), 1.0)
    return {
        "rule_k": rule_k,
        "set_n": set_n,
        "interval_log_count": log_count,
        "cell_count": cell_count,
        "cell_set": cell_set,
        "cell_rule_member": cell_rule_member,
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
