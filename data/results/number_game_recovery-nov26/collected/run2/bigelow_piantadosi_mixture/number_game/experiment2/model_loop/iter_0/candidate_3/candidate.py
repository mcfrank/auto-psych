"""Refinement of without_replacement_size_principle: Bayesian concept learning
over the same 45 rules (equal rule prior) and every interval of 1..100, with
examples drawn without replacement from the concept's not-yet-shown members
(10% uniform strays) and a guessing lapse. The one change: the prior over
numeric ranges is not flat but proportional to width ** (-gamma), a single
inferred preference for narrow (gamma > 0) or wide (gamma < 0) ranges; the
rules keep their flat prior.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

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
    return list(rules.values())


RULES = rule_concepts()
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
SIZE = MEMBERSHIP.sum(axis=1)
WIDTH = SIZE[N_RULES:].astype(int)  # interval widths 1..100
LOG_WIDTH = np.log(np.arange(1, 101, dtype="float64"))
N_OF_WIDTH = (101 - np.arange(1, 101)).astype("float64")  # intervals of each width


def set_log_lik(examples):
    """log P(examples in order | h) for every hypothesis, sampling members without replacement."""
    total = np.zeros(len(MEMBERSHIP))
    shown = np.zeros(len(MEMBERSHIP))
    for x in examples:
        member = MEMBERSHIP[:, x - 1]
        remaining = np.maximum(SIZE - shown, 1.0)
        total += np.log(ALPHA * member / remaining + (1 - ALPHA) / 100)
        shown += member
    return total


# Intervals are grouped by width: gamma only reweights the 100 width classes,
# so each set's per-width evidence and each (set, target) cell's per-width
# membership are tabulated once; rows index into the cells.

with pm.Model() as model:
    rule_log_evidence = pm.Data("rule_log_evidence", np.zeros(1))  # (sets,): log mean_r P(D | r)
    width_log_evidence = pm.Data("width_log_evidence", np.zeros((1, 100)))  # (sets, widths): log sum over width-w intervals
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,): set index of each (set, target) cell
    rule_in_concept = pm.Data("rule_in_concept", np.zeros(1))  # (cells,)
    width_in_concept = pm.Data("width_in_concept", np.zeros((1, 100)))  # (cells, widths): P(target in h | D, width w)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (rows,): cell index
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 1.0)  # preference for narrow ranges
    log_width_prior = -gamma * LOG_WIDTH
    log_norm = pt.logsumexp(log_width_prior + np.log(N_OF_WIDTH))
    weighted = width_log_evidence + log_width_prior[None, :]
    interval_log_evidence = pt.logsumexp(weighted, axis=1) - log_norm
    width_post = pt.special.softmax(weighted, axis=1)
    p_rule = pm.math.sigmoid(pt.log(lambda_0) + rule_log_evidence - pt.log1p(-lambda_0) - interval_log_evidence)

    interval_in_concept = (width_post[cell_set] * width_in_concept).sum(axis=1)
    p_cell = p_rule[cell_set]
    p_concept = pt.clip(p_cell * rule_in_concept + (1 - p_cell) * interval_in_concept, 0.0, 1.0)[row_cell]

    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    n = len(unique)
    rule_ev, rule_in = np.zeros(n), np.zeros((n, 100))
    w_ev, w_in = np.zeros((n, 100)), np.zeros((n, 100, 100))
    for i, s in enumerate(unique):
        ll = set_log_lik(s)
        lr = ll[:N_RULES]
        rule_ev[i] = logsumexp(lr) - np.log(N_RULES)
        rule_in[i] = np.exp(lr - logsumexp(lr)) @ MEMBERSHIP[:N_RULES]
        li = ll[N_RULES:]
        top = li.max()
        e = np.exp(li - top)
        mass = np.bincount(WIDTH - 1, weights=e, minlength=100)
        w_ev[i] = np.log(mass) + top
        for t in range(100):
            w_in[i, :, t] = np.bincount(WIDTH - 1, weights=e * MEMBERSHIP[N_RULES:, t], minlength=100) / mass
    keys = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {k: i for i, k in enumerate(cells)}
    cs = np.array([k[0] for k in cells], dtype="int64")
    ct = np.array([k[1] for k in cells], dtype="int64")
    return {
        "rule_log_evidence": rule_ev,
        "width_log_evidence": w_ev,
        "cell_set": cs,
        "rule_in_concept": rule_in[cs, ct],
        "width_in_concept": w_in[cs, :, ct],
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
