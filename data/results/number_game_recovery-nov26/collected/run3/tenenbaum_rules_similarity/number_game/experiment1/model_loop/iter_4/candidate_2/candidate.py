"""Refinement of parity_within_range_concepts with limited memory for the examples.

People do the same Bayesian concept learning as the best model (45 rules with a
family-organised prior, exception rules as their own kind, magnitude ranges and
odd/even-within-a-range concepts with an Erlang width prior, size principle with
10% random examples), but each example shown is retained in memory
independently with an inferred probability, and the judgment is the posterior
membership probability given only the retained examples (at least one is
retained). "Yes" is therefore a mixture over recalled subsets of the set.
"""

import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
MAX_SUBSETS = 15  # non-empty subsets of up to four examples


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
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
PARITY_RANGES = [[n for n in c if n % 2 == p] for p in (1, 0) for c in INTERVALS]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS + PARITY_RANGES], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / np.maximum(MEMBERSHIP.sum(axis=1, keepdims=True), 1) + (1 - ALPHA) / 100)
EXAMPLE_LOG_LIK[MEMBERSHIP.sum(axis=1) == 0] = -1e4  # empty concepts are impossible
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
_SPAN = np.array([len(c) for c in INTERVALS] * 3) - 1
INTERVAL_SIZE_IDX = _SPAN + 100 * np.repeat([0, 1, 1], len(INTERVALS))  # parity ranges share 100..199
SIZE_SELECT = [INTERVAL_SIZE_IDX == L for L in range(200)]
SIZES = np.arange(1, 101, dtype="float64")
LOG_N_OF_SIZE = np.log(101.0 - SIZES)


def rule_family(name):
    if name.startswith("powers of 2,"):
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

# Each (set, recalled subset) pair is a "memory state": prepare_observed lists
# every non-empty subset of every set (padded to 15 per set) with its rule
# likelihoods, per-size interval evidence and per-cell membership shares.

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (states, rules)
    size_log_evidence = pm.Data("size_log_evidence", np.zeros((1, 200)))  # (states, sizes)
    state_kept = pm.Data("state_kept", np.zeros((1, MAX_SUBSETS)))  # (sets, subsets): examples recalled
    state_lost = pm.Data("state_lost", np.zeros((1, MAX_SUBSETS)))  # (sets, subsets): examples forgotten
    state_pad = pm.Data("state_pad", np.zeros((1, MAX_SUBSETS)))  # 0 real, -1e4 padding
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_size_member = pm.Data("cell_size_member", np.zeros((1, MAX_SUBSETS, 200)))  # (cells, subsets, sizes)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    kind = pm.Dirichlet("kind_weights", a=np.full(3, 2.0))
    sigma = pm.LogNormal("sigma", mu=np.log(15.0), sigma=1.0)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    recall = pm.Beta("recall", alpha=4.0, beta=1.5)  # probability an example is retained
    rule_log_prior = pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])

    log_size_prior = np.log(SIZES) - SIZES / sigma
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior)
    log_per_interval = pt.concatenate([pt.log(kind[1]) + log_size_prior - LOG_N_OF_SIZE,
                                       pt.log(kind[2]) + log_size_prior - LOG_N_OF_SIZE - np.log(2.0)])

    log_rules = pt.log(kind[0]) + rule_log_prior[None, :] + rule_log_lik
    log_sizes = log_per_interval[None, :] + size_log_evidence
    log_joint = pt.concatenate([log_rules, log_sizes], axis=1)
    posterior = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    posterior = posterior.reshape((-1, MAX_SUBSETS, N_RULES + 200))  # (sets, subsets, concepts)

    # Probability of each recalled subset; normalising over the non-empty subsets
    # conditions on recalling at least one example.
    log_w = state_kept * pt.log(recall) + state_lost * pt.log1p(-recall) + state_pad
    weight = pt.exp(log_w - pt.logsumexp(log_w, axis=1, keepdims=True))  # (sets, subsets)

    cell_post = posterior[cell_set]  # (cells, subsets, concepts)
    in_state = ((cell_post[:, :, :N_RULES] * cell_rule_member[:, None, :]).sum(axis=2)
                + (cell_post[:, :, N_RULES:] * cell_size_member).sum(axis=2))
    in_concept = (weight[cell_set] * in_state).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n_sets = len(unique)
    n_states = n_sets * MAX_SUBSETS
    rule_ll = np.zeros((n_states, N_RULES))
    size_ev = np.zeros((n_states, 200))
    size_member = np.zeros((n_states, 200, 100))
    kept = np.zeros((n_sets, MAX_SUBSETS))
    lost = np.zeros((n_sets, MAX_SUBSETS))
    pad = np.full((n_sets, MAX_SUBSETS), -1e4)
    for i, s in enumerate(unique):
        n = len(s)
        subsets = [c for k in range(1, n + 1) for c in itertools.combinations(s, k)]
        for j in range(MAX_SUBSETS):
            sub = subsets[j] if j < len(subsets) else subsets[-1]  # padding repeats a real state
            if j < len(subsets):
                kept[i, j], lost[i, j], pad[i, j] = len(sub), n - len(sub), 0.0
            ll = EXAMPLE_LOG_LIK[:, np.array(sub) - 1].sum(axis=1)
            st = i * MAX_SUBSETS + j
            rule_ll[st] = ll[:N_RULES]
            lli = ll[N_RULES:]
            for L, sel in enumerate(SIZE_SELECT):
                ev = logsumexp(lli[sel])
                size_ev[st, L] = ev
                size_member[st, L] = np.exp(lli[sel] - ev) @ INTERVAL_MEMBERSHIP[sel]
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_s = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    state_idx = cell_s[:, None] * MAX_SUBSETS + np.arange(MAX_SUBSETS)[None, :]
    return {
        "rule_log_lik": rule_ll,
        "size_log_evidence": size_ev,
        "state_kept": kept,
        "state_lost": lost,
        "state_pad": pad,
        "cell_set": cell_s,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_t].T.copy(),
        "cell_size_member": np.clip(size_member[state_idx, :, cell_t[:, None]], 0.0, 1.0),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
