"""People do the same Bayesian concept learning as extended_multiples_concepts
(45 independently weighted rules plus multiples of 13..50, intervals of 1..100
with an Erlang width prior, an inferred share of random examples, examples drawn
without replacement), but when they check whether the target lies inside a range
of magnitudes they use a noisy analog magnitude of the target whose noise grows
in proportion to its size (scalar variability: Gaussian noise of sd `weber` on a
log number line). Arithmetic and digit rules are still checked exactly on the
numeral. "Yes" is the posterior probability that the concept contains the target,
with range membership taken in expectation over the perceived magnitude, so "yes"
fades gradually at range edges, more widely around large numbers.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_EX = 4


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
N_BASE = len(_RULES)
EXT_M = list(range(13, 51))  # less-familiar multiples rules
N_EXT = len(EXT_M)
_RULES.update({f"multiples of {m}": list(range(m, 101, m)) for m in EXT_M})
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)

# Intervals are grouped by (size, number of examples inside): within a group
# every interval has the same prior and likelihood for any noise rate.
SIZES = np.arange(1, 101)
GROUP_SIZE = np.repeat(SIZES, MAX_EX + 1).astype("float64")  # (500,)
GROUP_NIN = np.tile(np.arange(MAX_EX + 1), 100).astype("float64")
GROUP_SIZE_IDX = np.repeat(np.arange(100), MAX_EX + 1)
SIZES_F = SIZES.astype("float64")
N_PER_SIZE = 101.0 - SIZES_F  # intervals of each size in 1..100
LOG_N = np.log(NUMBERS.astype("float64"))

# Same Dirichlet aggregation trick as the incumbent for the duplicate set {10,...,100}.
END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_BASE - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))  # (sets, rules): examples in rule
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    group_log_count = pm.Data("group_log_count", np.zeros((1, 500)))  # (sets, groups)
    # Only the (set, target) cells that were asked about are evaluated.
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # (cells,) target - 1
    set_group_frac = pm.Data("set_group_frac", np.zeros((1, 500, 100)))  # (sets, groups, numbers)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,) index into cells
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    noise = pm.Beta("noise", 1.5, 6.0)  # share of examples believed random
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))
    # Share of the rule prior given to the multiples-of-13..50 family, split evenly.
    ext_share = pm.Beta("ext_share", 1.0, 4.0)
    all_weights = pt.concatenate([
        (1 - ext_share) * rule_weights, pt.ones(N_EXT) * ext_share / N_EXT,
    ])

    # Preferred interval width: Erlang(2, sigma) over interval size.
    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    log_erlang = pt.log(SIZES_F) - SIZES_F / sigma
    log_per_interval = log_erlang - pt.logsumexp(log_erlang + np.log(N_PER_SIZE))  # (sizes,)

    log_out = pt.log(noise / 100.0)
    # Without replacement: the j-th in-concept example is one of size - j unseen members.
    rule_log_lik = (n_examples[:, None] - rule_n_in) * log_out
    group_log_lik = (n_examples[:, None] - GROUP_NIN[None, :]) * log_out
    for j in range(MAX_EX):
        rule_term = pt.log((1 - noise) / np.maximum(RULE_SIZE - j, 1.0) + noise / 100.0)
        rule_log_lik = rule_log_lik + pt.gt(rule_n_in, j) * rule_term[None, :]
        group_term = pt.log((1 - noise) / np.maximum(GROUP_SIZE - j, 1.0) + noise / 100.0)
        group_log_lik = group_log_lik + ((GROUP_NIN > j) * group_term)[None, :]
    log_groups = (pt.log1p(-lambda_0) + log_per_interval[GROUP_SIZE_IDX])[None, :] + group_log_count + group_log_lik
    log_rules = pt.log(lambda_0) + pt.log(all_weights)[None, :] + rule_log_lik
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)  # (sets, rules + groups)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    # Range membership of each perceived magnitude, per set: (sets, numbers).
    range_mem = pt.batched_dot(posterior[:, N_RULES:], set_group_frac)
    # Scalar-variability perception: perceived magnitude ~ N(log t, weber^2) on a log line.
    weber = pm.LogNormal("weber", mu=np.log(0.08), sigma=0.8)
    d2 = (LOG_N[:, None] - LOG_N[None, :]) ** 2  # (true, perceived)
    kernel = pt.special.softmax(-d2 / (2 * weber**2), axis=1)
    range_seen = pt.dot(range_mem, kernel.T)  # (sets, true target)
    cell_post = posterior[cell_set]
    in_concept = (cell_post[:, :N_RULES] * cell_rule_member).sum(axis=1) + range_seen[cell_set, cell_target]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set`, `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_n_in = np.zeros((S, N_RULES))
    n_ex = np.zeros(S)
    counts = np.zeros((S, 500))
    contain = np.zeros((S, 500, 100))
    for i, s in enumerate(unique):
        ex = np.array(s) - 1
        n_ex[i] = len(s)
        rule_n_in[i] = RULE_MEMBERSHIP[:, ex].sum(axis=1)
        for m in range(1, 101):
            for n in range(m, 101):
                k = int(sum(m <= x <= n for x in s))
                g = (n - m) * (MAX_EX + 1) + k
                counts[i, g] += 1
                contain[i, g, m - 1:n] += 1
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(counts[:, :, None] > 0, contain / np.maximum(counts, 1)[:, :, None], 0.0)
    log_count = np.where(counts > 0, np.log(np.maximum(counts, 1)), -1e3)
    keys = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {k: i for i, k in enumerate(cells)}
    cs = np.array([k[0] for k in cells], dtype="int64")
    ct = np.array([k[1] for k in cells], dtype="int64")
    return {
        "rule_n_in": rule_n_in,
        "n_examples": n_ex,
        "group_log_count": log_count,
        "cell_set": cs,
        "cell_rule_member": RULE_MEMBERSHIP[:, ct].T.copy(),
        "cell_target": ct,
        "set_group_frac": frac,
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
