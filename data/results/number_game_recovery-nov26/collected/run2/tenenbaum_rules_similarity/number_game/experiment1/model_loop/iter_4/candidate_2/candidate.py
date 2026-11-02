"""Stochastic example forgetting: people do the Bayesian concept learning of
noisy_examples_erlang_intervals (45 independently weighted rules, every interval
of 1..100 with an Erlang prior on width, size principle, inferred share `noise`
of examples believed random), but each shown example is independently lost
from working memory with inferred probability `forget` before the target is
judged. "Yes" is the posterior probability that the concept contains the
target given the remembered examples, averaged over which examples survive
(all 2^n subsets, weighted forget^(lost) * (1 - forget)^(kept)).
"""

import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_EX = 4
N_SUB = 2 ** MAX_EX


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

# Intervals grouped by (size, number of remembered examples inside).
SIZES = np.arange(1, 101)
GROUP_SIZE = np.repeat(SIZES, MAX_EX + 1).astype("float64")
GROUP_NIN = np.tile(np.arange(MAX_EX + 1), 100).astype("float64")
GROUP_SIZE_IDX = np.repeat(np.arange(100), MAX_EX + 1)
SIZES_F = SIZES.astype("float64")
N_PER_SIZE = 101.0 - SIZES_F

END_0, CONTAINS_0 = RULE_NAMES.index("ends in 0"), RULE_NAMES.index("contains digit 0")
SET_CONCENTRATION = np.full(N_RULES - 1, 2.0)
SET_CONCENTRATION[END_0] = 4.0

with pm.Model() as model:
    # Memory states: every distinct subset of examples that could be remembered.
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))  # (subsets, rules)
    n_examples = pm.Data("n_examples", np.ones(1))  # (subsets,)
    group_log_count = pm.Data("group_log_count", np.zeros((1, 500)))  # (subsets, groups)
    sub_set = pm.Data("sub_set", np.zeros(1, dtype="int64"))  # (subcells,) memory state
    sub_rule_member = pm.Data("sub_rule_member", np.zeros((1, N_RULES)))  # (subcells, rules)
    sub_group_frac = pm.Data("sub_group_frac", np.zeros((1, 500)))  # (subcells, groups)
    # Each asked (set, target) cell mixes up to 16 memory states.
    cell_sub = pm.Data("cell_sub", np.zeros((1, N_SUB), dtype="int64"))
    cell_lost = pm.Data("cell_lost", np.zeros((1, N_SUB)))
    cell_kept = pm.Data("cell_kept", np.zeros((1, N_SUB)))
    cell_mask = pm.Data("cell_mask", np.ones((1, N_SUB)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    noise = pm.Beta("noise", 1.5, 6.0)
    forget = pm.Beta("forget", 1.5, 6.0)  # chance an example is not retained
    set_weights = pm.Dirichlet("set_weights", a=SET_CONCENTRATION)
    zero_split = pm.Beta("zero_split", 2, 2)
    shared = set_weights[END_0]
    rule_weights = pm.Deterministic("rule_weights", pt.concatenate([
        set_weights[:END_0], (zero_split * shared)[None], set_weights[END_0 + 1:CONTAINS_0],
        ((1 - zero_split) * shared)[None], set_weights[CONTAINS_0:],
    ]))

    sigma = pm.LogNormal("sigma", mu=np.log(20.0), sigma=1.0)
    log_erlang = pt.log(SIZES_F) - SIZES_F / sigma
    log_per_interval = log_erlang - pt.logsumexp(log_erlang + np.log(N_PER_SIZE))

    log_out = pt.log(noise / 100.0)
    rule_log_in = pt.log((1 - noise) / RULE_SIZE + noise / 100.0)
    rule_log_lik = rule_n_in * rule_log_in[None, :] + (n_examples[:, None] - rule_n_in) * log_out
    group_log_in = pt.log((1 - noise) / GROUP_SIZE + noise / 100.0)
    group_log_lik = (GROUP_NIN * group_log_in)[None, :] + (n_examples[:, None] - GROUP_NIN[None, :]) * log_out
    log_groups = (pt.log1p(-lambda_0) + log_per_interval[GROUP_SIZE_IDX])[None, :] + group_log_count + group_log_lik
    log_rules = pt.log(lambda_0) + pt.log(rule_weights)[None, :] + rule_log_lik
    log_joint = pt.concatenate([log_rules, log_groups], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    sp = posterior[sub_set]
    sub_in = (sp[:, :N_RULES] * sub_rule_member).sum(axis=1) + (sp[:, N_RULES:] * sub_group_frac).sum(axis=1)

    mem_w = cell_mask * pt.exp(cell_lost * pt.log(forget) + cell_kept * pt.log1p(-forget))
    in_concept = (mem_w * sub_in[cell_sub]).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-9, 1 - 1e-9))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _interval_stats(s, targets):
    """Group counts for remembered examples s, and per-target containment fractions."""
    counts = np.zeros(500)
    contain = np.zeros((500, 100))
    for m in range(1, 101):
        for n in range(m, 101):
            k = int(sum(m <= x <= n for x in s))
            g = (n - m) * (MAX_EX + 1) + k
            counts[g] += 1
            contain[g, m - 1:n] += 1
    frac = np.where(counts[:, None] > 0, contain / np.maximum(counts, 1)[:, None], 0.0)
    log_count = np.where(counts > 0, np.log(np.maximum(counts, 1)), -1e3)
    return log_count, {t: frac[:, t] for t in targets}


def prepare_observed(rows):
    """Rows carry `number_set`, `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    keys = [(s, int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {k: i for i, k in enumerate(cells)}

    # Memory states (sorted remembered multisets) and the targets asked of each.
    cell_states = []
    state_targets = {}
    for s, t in cells:
        states = []
        for keep in itertools.product([0, 1], repeat=len(s)):
            sub = tuple(sorted(x for x, k in zip(s, keep) if k))
            states.append((sub, len(s) - len(sub), len(sub)))
            state_targets.setdefault(sub, set()).add(t)
        cell_states.append(states)
    state_list = list(state_targets)
    state_index = {s: i for i, s in enumerate(state_list)}
    M = len(state_list)
    rule_n_in = np.zeros((M, N_RULES))
    n_ex = np.zeros(M)
    log_count = np.zeros((M, 500))
    sub_keys, sub_rule, sub_frac = [], [], []
    for i, sub in enumerate(state_list):
        n_ex[i] = len(sub)
        if sub:
            rule_n_in[i] = RULE_MEMBERSHIP[:, np.array(sub) - 1].sum(axis=1)
        lc, fr = _interval_stats(sub, sorted(state_targets[sub]))
        log_count[i] = lc
        for t, f in fr.items():
            sub_keys.append((sub, t))
            sub_rule.append(RULE_MEMBERSHIP[:, t])
            sub_frac.append(f)
    subcell_index = {k: j for j, k in enumerate(sub_keys)}

    C = len(cells)
    cell_sub = np.zeros((C, N_SUB), dtype="int64")
    cell_lost = np.zeros((C, N_SUB))
    cell_kept = np.zeros((C, N_SUB))
    cell_mask = np.zeros((C, N_SUB))
    for c, ((s, t), states) in enumerate(zip(cells, cell_states)):
        for j, (sub, lost, kept) in enumerate(states):
            cell_sub[c, j] = subcell_index[(sub, t)]
            cell_lost[c, j] = lost
            cell_kept[c, j] = kept
            cell_mask[c, j] = 1.0
    return {
        "rule_n_in": rule_n_in,
        "n_examples": n_ex,
        "group_log_count": log_count,
        "sub_set": np.array([state_index[k[0]] for k in sub_keys], dtype="int64"),
        "sub_rule_member": np.array(sub_rule),
        "sub_group_frac": np.array(sub_frac),
        "cell_sub": cell_sub,
        "cell_lost": cell_lost,
        "cell_kept": cell_kept,
        "cell_mask": cell_mask,
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
