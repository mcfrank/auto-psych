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

SAMPLER_SETTINGS = {"draws": 1000, "tune": 1000, "target_accept": 0.95}
MAX_SUBSETS = 15  # non-empty subsets of up to four examples

# Each (set, recalled non-empty subset) is a "memory state". The posterior is
# computed in linear space: prepare_observed stores, per state, each concept's
# example likelihood (scaled by a per-state constant) summed within each
# interval size, and per (cell, state) slot the same sums restricted to concepts
# containing the target, so every sum over concepts is a matrix-vector product.

with pm.Model() as model:
    state_rule_lik = pm.Data("state_rule_lik", np.zeros((1, N_RULES)))  # (states, rules)
    state_size_lik = pm.Data("state_size_lik", np.zeros((1, 200)))  # (states, sizes)
    state_kept = pm.Data("state_kept", np.zeros(1))  # (states,): examples recalled
    state_lost = pm.Data("state_lost", np.zeros(1))  # (states,): examples forgotten
    state_n = pm.Data("state_n", np.ones(1))  # (states,): size of the full set
    slot_state = pm.Data("slot_state", np.zeros(1, dtype="int64"))  # (cells * 15,)
    slot_real = pm.Data("slot_real", np.zeros(1))  # (cells * 15,): 1 real, 0 padding
    slot_rule_lik = pm.Data("slot_rule_lik", np.zeros((1, N_RULES)))  # (cells * 15, rules), target-containing only
    slot_size_lik = pm.Data("slot_size_lik", np.zeros((1, 200)))  # (cells * 15, sizes), target-containing only
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    kind = pm.Dirichlet("kind_weights", a=np.full(3, 2.0))
    sigma = pm.LogNormal("sigma", mu=np.log(15.0), sigma=1.0)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    recall = pm.Beta("recall", alpha=4.0, beta=1.5)  # probability an example is retained

    rule_prior = kind[0] * family_weights[FAMILY] / FAMILY_SIZE[FAMILY]  # per rule
    log_size_prior = np.log(SIZES) - SIZES / sigma
    size_prior = pt.exp(log_size_prior - pt.logsumexp(log_size_prior)) / (101.0 - SIZES)  # per interval
    interval_prior = pt.concatenate([kind[1] * size_prior, kind[2] * size_prior / 2.0])

    evidence = pt.dot(state_rule_lik, rule_prior) + pt.dot(state_size_lik, interval_prior)  # (states,)
    contains = pt.dot(slot_rule_lik, rule_prior) + pt.dot(slot_size_lik, interval_prior)  # (slots,)

    # Probability of each recalled subset given at least one example is recalled.
    log_w = (state_kept * pt.log(recall) + state_lost * pt.log1p(-recall)
             - pt.log1p(-pt.exp(state_n * pt.log1p(-recall))))
    state_scale = pt.exp(log_w) / evidence  # (states,)
    in_concept = (slot_real * state_scale[slot_state] * contains).reshape((-1, MAX_SUBSETS)).sum(axis=1)

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    rule_lik, size_lik, member_lik, kept, lost, n_of, set_states = [], [], [], [], [], [], []
    for s in unique:
        n = len(s)
        first = len(kept)
        for k in range(1, n + 1):
            for sub in itertools.combinations(s, k):
                ll = EXAMPLE_LOG_LIK[:, np.array(sub) - 1].sum(axis=1)
                lik = np.exp(ll - ll.max())  # common per-state scale cancels in the posterior
                rule_lik.append(lik[:N_RULES])
                lik_i = lik[N_RULES:]
                size_lik.append(np.array([lik_i[sel].sum() for sel in SIZE_SELECT]))
                # (sizes, 100): likelihood summed over intervals of each size containing each number
                member_lik.append(np.stack([lik_i[sel] @ INTERVAL_MEMBERSHIP[sel] for sel in SIZE_SELECT]))
                kept.append(k)
                lost.append(n - k)
                n_of.append(n)
        set_states.append(list(range(first, len(kept))))
    rule_lik = np.array(rule_lik)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    n_slots = len(cells) * MAX_SUBSETS
    slot_state = np.zeros(n_slots, dtype="int64")
    slot_real = np.zeros(n_slots)
    slot_rule = np.zeros((n_slots, N_RULES))
    slot_size = np.zeros((n_slots, 200))
    for ci, (si, t) in enumerate(cells):
        for j, st in enumerate(set_states[si]):
            sl = ci * MAX_SUBSETS + j
            slot_state[sl], slot_real[sl] = st, 1.0
            slot_rule[sl] = rule_lik[st] * RULE_MEMBERSHIP[:, t - 1]
            slot_size[sl] = member_lik[st][:, t - 1]
    return {
        "state_rule_lik": rule_lik,
        "state_size_lik": np.array(size_lik),
        "state_kept": np.array(kept, dtype="float64"),
        "state_lost": np.array(lost, dtype="float64"),
        "state_n": np.array(n_of, dtype="float64"),
        "slot_state": slot_state,
        "slot_real": slot_real,
        "slot_rule_lik": slot_rule,
        "slot_size_lik": slot_size,
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
