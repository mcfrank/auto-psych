"""Surface-format screening before concept learning.

People first screen the target by its written form: if it differs from every
example in its number of digits (1, 2 or 3 digits) or in its roundness (ending
in zero or not), they reject it outright, each mismatching dimension with its own
inferred rejection rate, without consulting their concept. Targets passing the
screen are judged by the incumbent's learner (42 rules, the 5,050 intervals of
1..100 with a Beta midpoint prior and a decade-aligned prior bonus, size principle
with an inferred stray-example rate, evidence weight beta * n ** gamma, guessing
with an inferred yes-leaning that falls with set size). This predicts that
non-round targets after all-round sets and one-digit targets after high two-digit
sets are rejected more often than concept membership implies.

Computation: the incumbent's posterior, organised cheaply via (interval length,
examples contained) keys and constant sparse indicator matrices built in
prepare_observed.
"""

import numpy as np
import pymc as pm
import pytensor.sparse as ps
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_SET = 4
N_KEYS = 100 * (MAX_SET + 1)  # (interval length, examples contained)


def concept_sets():
    rules = [
        [2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36],
        [7, 49], [8, 64], [9, 81], [10, 100],
    ]
    rules += [[i for i in NUMBERS if i % n == 0] for n in range(3, 13)]
    rules += [[i for i in NUMBERS if i % 10 == d] for d in range(1, 10)]
    rules += [[i for i in NUMBERS if str(d) in str(i)] for d in range(1, 10)]
    rules.append([1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97])
    rules.append([i for i in NUMBERS if i % 2 == 0])
    rules.append([i for i in NUMBERS if i % 2 == 1])
    rules.append([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])
    rules.append([1, 8, 27, 64])
    intervals = [(m, n) for m in range(1, 101) for n in range(m, 101)]
    return rules, intervals


RULES, INTERVALS = concept_sets()
N_RULES, N_INTERVALS = len(RULES), len(INTERVALS)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")  # (rules, 100)
LOG_RULE_SIZE = np.log(RULE_MEMBERSHIP.sum(axis=1))
INT_LO = np.array([m for m, _ in INTERVALS])
INT_HI = np.array([n for _, n in INTERVALS])
INT_LEN = INT_HI - INT_LO + 1
LOG_LENGTHS = np.log(np.arange(1, 101, dtype="float64"))

MIDPOINT_GRID = np.clip(np.arange(199) / 198, 1e-5, 1 - 1e-5)
INTERVAL_GRID_POINT = INT_LO + INT_HI - 2

DECADE_STARTS = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90}
DECADE_ENDS = {9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 100}
DECADE_ALIGNED = np.array(
    [1.0 if m in DECADE_STARTS and n in DECADE_ENDS else 0.0 for m, n in INTERVALS]
)


with pm.Model() as model:
    set_rule_n = pm.Data("set_rule_n", np.zeros((1, N_RULES)))  # (sets, rules): examples r contains
    set_size = pm.Data("set_size", np.ones(1, dtype="float64"))  # (sets,)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_target = pm.Data("cell_target", np.zeros(1, dtype="int64"))  # target - 1
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    digit_mismatch = pm.Data("digit_mismatch", np.zeros(1, dtype="float64"))  # (cells,) 1 if no example has the target's digit count
    round_mismatch = pm.Data("round_mismatch", np.zeros(1, dtype="float64"))  # (cells,) 1 if no example shares its roundness
    # CSR indicators: row (set, key) / (cell, key) -> intervals with that key [containing the target].
    z_indices = pm.Data("z_indices", np.zeros(1, dtype="int32"))
    z_indptr = pm.Data("z_indptr", np.zeros(1 + N_KEYS, dtype="int32"))
    c_indices = pm.Data("c_indices", np.zeros(1, dtype="int32"))
    c_indptr = pm.Data("c_indptr", np.zeros(1 + N_KEYS, dtype="int32"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    p_mix = pm.Beta("p_mix", 2, 2)
    rule_weights = pm.Dirichlet("rule_weights", a=np.full(N_RULES, 2.0))
    midpoint_shape = pm.Gamma("midpoint_shape", alpha=2, beta=1, shape=2)
    evidence_strength = pm.LogNormal("evidence_strength", mu=0.0, sigma=0.5)
    size_growth = pm.Normal("size_growth", mu=0.0, sigma=1.0)
    weight = evidence_strength * pt.exp(size_growth * pt.log(set_size))  # (sets,)
    decade_bonus = pm.Normal("decade_bonus", mu=0.0, sigma=3.0)

    # Inferred belief that an example is a stray number from 1..100 (incumbent: 0.1).
    # Each example a concept contains gains log P(x | h) - log P(x | stray); the stray
    # term common to every concept cancels in the normalisation.
    stray_rate = pm.Beta("stray_rate", 1.5, 6.0)
    log_stray = pt.log(stray_rate / 100)
    rule_gain = pt.log((1 - stray_rate) * pt.exp(-LOG_RULE_SIZE) + stray_rate / 100) - log_stray
    length_gain = pt.log((1 - stray_rate) * pt.exp(-LOG_LENGTHS) + stray_rate / 100) - log_stray

    log_density = pm.logp(pm.Beta.dist(midpoint_shape[0], midpoint_shape[1]), MIDPOINT_GRID)
    log_interval = log_density[INTERVAL_GRID_POINT] + decade_bonus * DECADE_ALIGNED
    log_interval_prior = pt.log1p(-p_mix) + log_interval - pt.logsumexp(log_interval)  # (intervals,)

    # Unnormalised log posterior of rules, and of intervals by (length, count) key.
    log_rule = pt.log(p_mix) + pt.log(rule_weights)[None, :] + weight[:, None] * set_rule_n * rule_gain[None, :]
    counts = np.arange(MAX_SET + 1, dtype="float64")
    log_key = (weight[:, None, None] * counts[None, :, None] * length_gain[None, None, :]).reshape((-1, N_KEYS))
    bound = pt.maximum(
        pt.max(log_rule, axis=1),
        weight * set_size * pt.max(length_gain) + pt.max(log_interval_prior),
    )  # (sets,) >= every used log term
    rule_w = pt.exp(log_rule - bound[:, None])  # (sets, rules)
    key_w = pt.exp(pt.minimum(log_key - bound[:, None], 50.0))  # (sets, keys)

    interval_prior = pt.exp(log_interval_prior)[:, None]
    n_sets, n_cells = set_size.shape[0], cell_set.shape[0]
    z_ind = ps.CSR(pt.ones_like(z_indices, dtype="float64"), z_indices, z_indptr,
                   pt.stack([n_sets * N_KEYS, N_INTERVALS]))
    c_ind = ps.CSR(pt.ones_like(c_indices, dtype="float64"), c_indices, c_indptr,
                   pt.stack([n_cells * N_KEYS, N_INTERVALS]))
    key_mass = ps.structured_dot(z_ind, interval_prior).reshape((n_sets, N_KEYS))
    key_mass_in = ps.structured_dot(c_ind, interval_prior).reshape((n_cells, N_KEYS))

    total = pt.sum(rule_w, axis=1) + pt.sum(key_w * key_mass, axis=1)  # (sets,)
    in_mass = (
        pt.sum(rule_w[cell_set] * pt.as_tensor_variable(RULE_MEMBERSHIP.T)[cell_target], axis=1)
        + pt.sum(key_w[cell_set] * key_mass_in, axis=1)
    )
    in_concept = in_mass / total[cell_set]  # (cells,)

    guess_single = pm.Normal("guess_single", mu=-2.0, sigma=1.5)
    guess_decline = pm.Normal("guess_decline", mu=0.0, sigma=2.0)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    guess_rate = pm.math.sigmoid(guess_single - guess_decline * pt.log(set_size))[cell_set]  # (cells,)
    q = guess_rate * guess_yes + (1 - guess_rate) * in_concept  # (cells,)

    # Surface-format screen (the one mechanism): reject outright a target whose
    # written form matches no example, per mismatching dimension.
    digit_reject = pm.Beta("digit_reject", 1.0, 4.0)
    round_reject = pm.Beta("round_reject", 1.0, 4.0)
    pass_screen = (1 - digit_reject * digit_mismatch) * (1 - round_reject * round_mismatch)  # (cells,)
    p_yes = pm.Deterministic("p_yes", pt.clip((pass_screen * q)[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _csr(row_keys, n_rows):
    """CSR indices/indptr of a 0/1 matrix given, per row, the interval columns."""
    lengths = np.array([len(c) for c in row_keys], dtype="int64")
    indptr = np.concatenate([[0], np.cumsum(lengths)]).astype("int32")
    indices = np.concatenate([np.asarray(c, dtype="int32") for c in row_keys]) if n_rows else np.zeros(0, "int32")
    return indices, indptr


def _key_rows(keys, mask=None):
    """Group interval columns by key (0..N_KEYS-1): one list of columns per key."""
    cols = np.arange(N_INTERVALS) if mask is None else np.flatnonzero(mask)
    k = keys[cols]
    order = np.argsort(k, kind="stable")
    bounds = np.searchsorted(k[order], np.arange(N_KEYS + 1))
    return [cols[order[bounds[i]:bounds[i + 1]]] for i in range(N_KEYS)]


def _form(x):
    """Written form of a number: (digit count, ends in zero)."""
    return len(str(x)), x % 10 == 0


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target`, `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = [int(r["target"]) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    cells = list(dict.fromkeys(zip(sets, targets)))
    cell_index = {c: i for i, c in enumerate(cells)}

    set_keys = []
    for s in unique:
        x = np.array(s)
        n_in = ((x[None, :] >= INT_LO[:, None]) & (x[None, :] <= INT_HI[:, None])).sum(axis=1)
        set_keys.append(n_in * 100 + (INT_LEN - 1))
    z_rows = [cols for keys in set_keys for cols in _key_rows(keys)]
    c_rows = []
    for s, t in cells:
        contains = (INT_LO <= t) & (INT_HI >= t)
        c_rows += _key_rows(set_keys[index[s]], contains)
    z_indices, z_indptr = _csr(z_rows, len(z_rows))
    c_indices, c_indptr = _csr(c_rows, len(c_rows))

    return {
        "set_rule_n": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_size": np.array([float(len(s)) for s in unique], dtype="float64"),
        "cell_set": np.array([index[s] for s, _ in cells], dtype="int64"),
        "cell_target": np.array([t - 1 for _, t in cells], dtype="int64"),
        "row_cell": np.array([cell_index[c] for c in zip(sets, targets)], dtype="int64"),
        "digit_mismatch": np.array([float(all(_form(x)[0] != _form(t)[0] for x in s)) for s, t in cells]),
        "round_mismatch": np.array([float(all(_form(x)[1] != _form(t)[1] for x in s)) for s, t in cells]),
        "z_indices": z_indices,
        "z_indptr": z_indptr,
        "c_indices": c_indices,
        "c_indptr": c_indptr,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
