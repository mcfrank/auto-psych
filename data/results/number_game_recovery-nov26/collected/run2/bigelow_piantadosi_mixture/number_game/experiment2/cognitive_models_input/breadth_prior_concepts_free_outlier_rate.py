"""Refinement of rule_breadth_prior_concepts: Bayesian concept learning over
the same 45 rules (prior share lambda_0 split in proportion to |rule|**gamma)
and every interval of 1..100 (each (1 - lambda_0) / 5050), size principle,
with a response lapse (guess "yes" at rate guess_yes on a fraction `lapse` of
trials). The one change: the probability that an example is a stray uniform
draw from 1..100 is a free parameter `outlier` (fixed at 0.1 before), so the
learner may discount a misfitting example rather than abandon a tight concept.
"""

import numpy as np
import pymc as pm
import pytensor.sparse as ps
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
    return list(rules.values())


RULES = rule_concepts()
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
LOG_RULE_SIZE_C = np.log(RULE_SIZE) - np.log(RULE_SIZE).mean()
N_INTERVALS = 5050
# Intervals are grouped by (size, number of examples inside): group = (size - 1) * 5 + k.
GROUP_SIZE = np.repeat(np.arange(1, 101), MAX_EX + 1).astype("float64")
GROUP_K = np.tile(np.arange(MAX_EX + 1), 100).astype("float64")
N_GROUPS = len(GROUP_SIZE)

with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # (sets, rules): examples inside the rule
    n_examples = pm.Data("n_examples", np.ones(1))  # (sets,)
    group_log_count = pm.Data("group_log_count", np.zeros((1, N_GROUPS)))  # (sets, groups), -1e9 if empty
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))
    # Sparse (rows, sets * columns) matrices: how many hypotheses of each column contain the row's target.
    rule_val = pm.Data("rule_val", np.ones(1))
    rule_col = pm.Data("rule_col", np.zeros(1, dtype="int32"))
    rule_ptr = pm.Data("rule_ptr", np.array([0, 1], dtype="int32"))
    group_val = pm.Data("group_val", np.ones(1))
    group_col = pm.Data("group_col", np.zeros(1, dtype="int32"))
    group_ptr = pm.Data("group_ptr", np.array([0, 1], dtype="int32"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 1.0)
    outlier = pm.Beta("outlier", 2, 10)  # belief that an example is a random stray

    log_stray = pt.log(outlier) - np.log(100.0)
    n = n_examples[:, None]
    # log P(D | h) from the counts of examples inside / outside h.
    rule_ll = rule_k * pt.log((1 - outlier) / RULE_SIZE + outlier / 100.0)[None, :] + (n - rule_k) * log_stray
    group_ll = GROUP_K[None, :] * pt.log((1 - outlier) / GROUP_SIZE + outlier / 100.0)[None, :] + (
        n - GROUP_K[None, :]
    ) * log_stray

    log_rule_prior = gamma * LOG_RULE_SIZE_C
    log_rule_prior = log_rule_prior - pt.logsumexp(log_rule_prior)
    rule_joint = rule_ll + log_rule_prior[None, :]
    rule_log_evidence = pt.logsumexp(rule_joint, axis=1)
    rule_post = pt.exp(rule_joint - rule_log_evidence[:, None])  # (sets, rules)

    group_joint = group_ll + group_log_count
    interval_log_evidence = pt.logsumexp(group_joint, axis=1)
    interval_each = pt.exp(group_ll - interval_log_evidence[:, None])  # posterior of one interval in group

    n_rows = row_set.shape[0]
    rule_hits = ps.CSR(rule_val, rule_col, rule_ptr, pt.stack([n_rows, rule_post.size]))
    group_hits = ps.CSR(group_val, group_col, group_ptr, pt.stack([n_rows, interval_each.size]))
    rule_in = ps.structured_dot(rule_hits, rule_post.reshape((-1, 1)))[:, 0]
    interval_in = ps.structured_dot(group_hits, interval_each.reshape((-1, 1)))[:, 0]

    log_rules = pt.log(lambda_0) + rule_log_evidence
    log_intervals = pt.log1p(-lambda_0) - np.log(N_INTERVALS) + interval_log_evidence
    p_rule = pm.math.sigmoid(log_rules - log_intervals)[row_set]
    p_concept = pt.clip(p_rule * rule_in + (1 - p_rule) * interval_in, 0.0, 1.0)

    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)

SAMPLER_SETTINGS = {"target_accept": 0.9}


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    lo, hi = np.meshgrid(NUMBERS, NUMBERS, indexing="ij")
    valid = hi >= lo
    size_idx = (hi - lo)[valid]
    lo_v, hi_v = lo[valid], hi[valid]
    rule_k = np.zeros((len(unique), N_RULES))
    n_ex = np.zeros(len(unique))
    group_count = np.zeros((len(unique), N_GROUPS))
    group_of = []
    for i, s in enumerate(unique):
        ex = np.array(s)
        rule_k[i] = RULE_MEMBERSHIP[:, ex - 1].sum(axis=1)
        n_ex[i] = len(ex)
        k = ((ex[None, :] >= lo_v[:, None]) & (ex[None, :] <= hi_v[:, None])).sum(axis=1)
        g = size_idx * (MAX_EX + 1) + k
        group_of.append(g)
        np.add.at(group_count[i], g, 1.0)
    row_group_hits = np.zeros((len(rows), N_GROUPS))
    row_rule_hit = np.zeros((len(rows), N_RULES))
    row_set = np.zeros(len(rows), dtype="int64")
    for j, (s, r) in enumerate(zip(sets, rows)):
        t = int(r["target"])
        i = index[s]
        row_set[j] = i
        row_rule_hit[j] = RULE_MEMBERSHIP[:, t - 1]
        contains = (lo_v <= t) & (hi_v >= t)
        np.add.at(row_group_hits[j], group_of[i][contains], 1.0)
    return {
        "rule_k": rule_k,
        "n_examples": n_ex,
        "group_log_count": np.where(group_count > 0, np.log(np.maximum(group_count, 1.0)), -1e9),
        "row_set": row_set,
        **csr("rule", row_rule_hit, row_set, N_RULES),
        **csr("group", row_group_hits, row_set, N_GROUPS),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


def csr(prefix, hits, row_set, width):
    """CSR parts of the (rows, sets * width) matrix placing each row's hits in its set's block."""
    rows, cols = np.nonzero(hits)
    ptr = np.searchsorted(rows, np.arange(len(hits) + 1)).astype("int32")
    return {
        f"{prefix}_val": hits[rows, cols].astype("float64"),
        f"{prefix}_col": (row_set[rows] * width + cols).astype("int32"),
        f"{prefix}_ptr": ptr,
    }
