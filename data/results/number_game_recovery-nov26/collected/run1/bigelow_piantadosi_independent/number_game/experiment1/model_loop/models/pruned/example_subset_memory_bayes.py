"""Bayesian concept learning from a memory-limited subset of the examples.

People do Bayesian concept learning over the 45 rules and all intervals of
1..100 (Bigelow & Piantadosi prior: lambda_0 split equally over rules,
(1 - lambda_0) / 100^2 per interval; strong-sampling likelihood with 10% noise),
but under a working-memory limit: each shown example enters the inference
independently with fitted probability q (conditioned on at least one being
retained). "Yes" is the posterior probability that the concept contains the
target, averaged over which subset of the examples was retained. Reasoning
from a subset lets rules shared by only some examples win and weakens the
tightening produced by more examples.
"""

import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

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


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_LOG_LIK = np.log(ALPHA * RULE_MEMBERSHIP / RULE_MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
MAX_EXAMPLES = 4
N_SLOTS = 2**MAX_EXAMPLES - 1  # non-empty subsets of up to 4 examples
SIZES = np.arange(1, 101)
N_GROUPS = 100 * (MAX_EXAMPLES + 1)

with pm.Model() as model:
    # Within rules and within intervals the posterior is parameter-free, so per
    # (cell, retained-subset slot) we precompute: the log evidence of the rule
    # class and the interval class, and each class's posterior probability that
    # the concept contains the target. Only lambda_0 and q are inferred.
    slot_log_z_rule = pm.Data("slot_log_z_rule", np.zeros((1, N_SLOTS)))
    slot_log_z_interval = pm.Data("slot_log_z_interval", np.zeros((1, N_SLOTS)))
    slot_rule_in = pm.Data("slot_rule_in", np.zeros((1, N_SLOTS)))
    slot_interval_in = pm.Data("slot_interval_in", np.zeros((1, N_SLOTS)))
    slot_k = pm.Data("slot_k", np.zeros((1, N_SLOTS)))  # examples retained
    slot_n = pm.Data("slot_n", np.zeros((1, N_SLOTS)))  # examples shown
    slot_valid = pm.Data("slot_valid", np.zeros((1, N_SLOTS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    q = pm.Beta("q", 2, 2)  # probability each example is retained in memory

    # Posterior probability of the rule class given the retained subset.
    w_rule = pm.math.sigmoid(pt.log(lambda_0) - pt.log1p(-lambda_0) + slot_log_z_rule - slot_log_z_interval)
    in_concept = w_rule * slot_rule_in + (1 - w_rule) * slot_interval_in  # (cells, slots)

    weight = slot_valid * q**slot_k * (1 - q) ** (slot_n - slot_k)
    weight = weight / weight.sum(axis=1, keepdims=True)
    p_cell = (weight * in_concept).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _interval_groups(examples):
    log_lik = np.zeros(N_GROUPS)
    count = np.zeros(N_GROUPS)
    in_concept = np.zeros((N_GROUPS, 100))
    ex = np.array(examples)
    n = len(ex)
    for s in SIZES:
        for k in range(MAX_EXAMPLES + 1):
            log_lik[(s - 1) * (MAX_EXAMPLES + 1) + k] = k * np.log(ALPHA / s + (1 - ALPHA) / 100) + (n - k) * np.log(
                (1 - ALPHA) / 100
            )
        for m in range(1, 102 - s):
            k = int(((ex >= m) & (ex <= m + s - 1)).sum())
            g = (s - 1) * (MAX_EXAMPLES + 1) + k
            count[g] += 1
            in_concept[g, m - 1 : m + s - 1] += 1
    nonzero = count > 0
    in_concept[nonzero] /= count[nonzero, None]
    log_count = np.where(nonzero, np.log(np.maximum(count, 1)), -1e4)
    return log_lik, log_count, in_concept


def _logsumexp(x):
    m = x.max()
    return m + np.log(np.exp(x - m).sum())


def _subset_summary(examples):
    """Class log evidences and per-target in-concept probabilities within each class."""
    rule_lj = RULE_LOG_LIK[:, np.array(examples) - 1].sum(axis=1) - np.log(N_RULES)
    log_z_rule = _logsumexp(rule_lj)
    rule_in = np.exp(rule_lj - log_z_rule) @ RULE_MEMBERSHIP  # (100,)
    log_lik, log_count, gic = _interval_groups(examples)
    group_lj = log_lik + log_count - 2 * np.log(100)
    log_z_interval = _logsumexp(group_lj)
    interval_in = np.exp(group_lj - log_z_interval) @ gic  # (100,)
    return log_z_rule, log_z_interval, rule_in, interval_in


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    trial_cells = [(s, int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(trial_cells))
    cell_index = {c: i for i, c in enumerate(cells)}

    set_subsets = {}
    for s in dict.fromkeys(sets):
        subs = []
        for k in range(1, len(s) + 1):
            subs.extend(tuple(sorted(c)) for c in itertools.combinations(s, k))
        set_subsets[s] = subs
    summaries = {x: _subset_summary(x) for subs in set_subsets.values() for x in subs}

    n_cells = len(cells)
    out = {name: np.zeros((n_cells, N_SLOTS)) for name in [
        "slot_log_z_rule", "slot_log_z_interval", "slot_rule_in", "slot_interval_in", "slot_k", "slot_n", "slot_valid"]}
    for i, (s, t) in enumerate(cells):
        for j, x in enumerate(set_subsets[s]):
            lzr, lzi, r_in, i_in = summaries[x]
            out["slot_log_z_rule"][i, j] = lzr
            out["slot_log_z_interval"][i, j] = lzi
            out["slot_rule_in"][i, j] = r_in[t - 1]
            out["slot_interval_in"][i, j] = i_in[t - 1]
            out["slot_k"][i, j] = len(x)
            out["slot_n"][i, j] = len(s)
            out["slot_valid"][i, j] = 1.0
    out["row_cell"] = np.array([cell_index[c] for c in trial_cells], dtype="int64")
    out["rating"] = np.array([int(r["rating"]) for r in rows], dtype="int64")
    return out
