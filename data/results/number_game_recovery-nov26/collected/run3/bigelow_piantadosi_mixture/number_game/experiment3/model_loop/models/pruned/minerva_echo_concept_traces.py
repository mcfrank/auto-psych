"""Echo-based exemplar memory over stored number-concept traces (MINERVA-2 style).

People do not infer which concept generated the examples; they hold familiar
number concepts (45 mathematical rules and every magnitude range of 1-100) as
stored traces, and the examples together form a retrieval probe. Each trace is
activated by a steep, inferred power of its cosine similarity to the probe,
overlap / sqrt(|concept| * n_examples), weighted by how many traces of its kind
are stored (rules vs ranges share lambda / 1 - lambda). A target is endorsed in
proportion to the share of total activation (the echo) carried by traces that
contain it. Small concepts gain from extra examples only by a fixed power of
overlap, and a concept missing an example is weakened, not excluded.
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


RULES = list(rule_concepts().values())
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")  # (rules, 100)
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
LO = np.array([m for m in NUMBERS for n in range(m, 101)])
HI = np.array([n for m in NUMBERS for n in range(m, 101)])
RANGE_SIZE = HI - LO + 1
N_RANGES = len(LO)
LOG_K = np.log(np.maximum(np.arange(MAX_N + 1), 1e-300))  # log overlap 0..4 (0 -> ~ -690)
LOG_SIZE = np.log(np.arange(1, 101, dtype="float64"))

with pm.Model() as model:
    set_log_n = pm.Data("set_log_n", np.zeros(1))  # (sets,)
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))  # overlap of each rule with the probe
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_mem = pm.Data("cell_rule_mem", np.zeros((1, N_RULES)))  # target in rule
    set_range_count = pm.Data("set_range_count", np.zeros((1, 100, MAX_N + 1)))  # ranges by (size, k)
    cell_range_count = pm.Data("cell_range_count", np.zeros((1, 100, MAX_N + 1)))  # ... containing target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)  # share of stored traces that are rules
    log_power = pm.Normal("log_power", mu=np.log(3.0), sigma=0.7)
    power = pm.Deterministic("power", pt.exp(log_power))  # activation = similarity ** power

    # Rule traces: activation = prior * (k / sqrt(|C| n)) ** power.
    rule_logsim = (
        pt.log(pt.maximum(rule_k, 1e-300)) - 0.5 * np.log(RULE_SIZE)[None, :] - 0.5 * set_log_n[:, None]
    )
    rule_act = pt.where(rule_k > 0, pt.exp(power * rule_logsim), 0.0) * (lambda_0 / N_RULES)  # (sets, rules)

    # Range traces, grouped by (size, overlap).
    range_logsim = (
        LOG_K[None, None, :] - 0.5 * LOG_SIZE[None, :, None] - 0.5 * set_log_n[:, None, None]
    )  # (sets, 100, 5)
    range_w = pt.where(np.arange(MAX_N + 1)[None, None, :] > 0, pt.exp(power * range_logsim), 0.0)
    range_w = range_w * ((1 - lambda_0) / N_RANGES)
    range_total = (range_w * set_range_count).sum(axis=(1, 2))  # (sets,)
    range_cell = (range_w[cell_set] * cell_range_count).sum(axis=(1, 2))  # (cells,)

    total = rule_act.sum(axis=1) + range_total
    echo_target = (rule_act[cell_set] * cell_rule_mem).sum(axis=1) + range_cell
    p_cell = echo_target / total[cell_set]
    p_yes = pm.Deterministic("p_yes", pt.clip(p_cell, 1e-6, 1 - 1e-6)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    set_log_n = np.array([np.log(len(set(s))) for s in unique])
    rule_k = np.zeros((S, N_RULES))
    range_k = np.zeros((S, N_RANGES), dtype="int64")
    set_range_count = np.zeros((S, 100, MAX_N + 1))
    for i, s in enumerate(unique):
        ex = np.array(sorted(set(s)))
        rule_k[i] = RULE_MEMBERSHIP[:, ex - 1].sum(axis=1)
        range_k[i] = ((ex[None, :] >= LO[:, None]) & (ex[None, :] <= HI[:, None])).sum(axis=1)
        np.add.at(set_range_count[i], (RANGE_SIZE - 1, range_k[i]), 1.0)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    C = len(cells)
    cell_set = np.array([i for i, _ in cells], dtype="int64")
    cell_rule_mem = np.zeros((C, N_RULES))
    cell_range_count = np.zeros((C, 100, MAX_N + 1))
    for j, (i, t) in enumerate(cells):
        cell_rule_mem[j] = RULE_MEMBERSHIP[:, t - 1]
        has = (LO <= t) & (HI >= t)
        np.add.at(cell_range_count[j], (RANGE_SIZE[has] - 1, range_k[i][has]), 1.0)
    return {
        "set_log_n": set_log_n,
        "rule_k": rule_k,
        "cell_set": cell_set,
        "cell_rule_mem": cell_rule_mem,
        "set_range_count": set_range_count,
        "cell_range_count": cell_range_count,
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
