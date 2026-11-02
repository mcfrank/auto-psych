"""Refinement of without_replacement_size_principle: Bayesian concept learning
over the same 45 rules and every interval of 1..100, with examples read as
distinct draws (sampling without replacement) and 10% random. The one change:
the prior on numeric ranges falls as a power of their width (prior proportional
to size ** -k, with k inferred), so narrow ranges are more plausible a priori
than wide ones. On lapse trials (rate `lapse`) people answer a bare "yes"
(the incumbent's guessing rate tends to 1 on these data).

Speed: intervals are grouped by width. Because the width prior is constant
within a width, the per-set evidence and the per-cell membership mass are
precomputed for each of the 100 widths, and the model only takes a
logsumexp over widths.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
NEG = -700.0


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
INT_WIDTH = SIZE[N_RULES:].astype(int)  # 1..100
LOG_WIDTH = np.log(np.arange(1, 101, dtype="float64"))
LOG_N_WIDTH = np.log(101.0 - np.arange(1, 101))  # number of intervals of each width
WIDTH_ONEHOT = (INT_WIDTH[:, None] == np.arange(1, 101)[None, :]).astype("float64")  # (intervals, widths)


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


def _log_group(log_lik, weights):
    """log sum over intervals of each width of exp(log_lik) * weights -> (..., widths)."""
    m = log_lik.max()
    s = (np.exp(log_lik - m) * weights) @ WIDTH_ONEHOT
    with np.errstate(divide="ignore"):
        out = np.log(s) + m
    return np.where(np.isfinite(out), out, NEG)


with pm.Model() as model:
    rule_log_evidence = pm.Data("rule_log_evidence", np.zeros(1))  # (sets,)
    rule_in_concept = pm.Data("rule_in_concept", np.zeros(1))  # (cells,)
    width_log_evidence = pm.Data("width_log_evidence", np.zeros((1, 100)))  # (sets, widths)
    width_log_member = pm.Data("width_log_member", np.zeros((1, 100)))  # (cells, widths)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # set of each distinct (set, target) cell
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    k = pm.Normal("k", mu=0.0, sigma=1.0)  # width preference: >0 favours narrow ranges
    log_w = -k * LOG_WIDTH  # unnormalised log prior of one interval of each width
    log_norm = pt.logsumexp(log_w + LOG_N_WIDTH)
    log_int_ev = pt.logsumexp(width_log_evidence + log_w[None, :], axis=1) - log_norm  # (sets,)

    log_rules = pt.log(lambda_0) - np.log(N_RULES) + rule_log_evidence
    log_intervals = pt.log1p(-lambda_0) + log_int_ev
    p_rule = pm.math.sigmoid(log_rules - log_intervals)[cell_set]

    log_int_member = pt.logsumexp(width_log_member + log_w[None, :], axis=1) - log_norm
    int_in_concept = pt.exp(pt.minimum(log_int_member - log_int_ev[cell_set], 0.0))
    cell_p = p_rule * rule_in_concept + (1 - p_rule) * int_in_concept
    p_concept = pt.clip(cell_p[row_cell], 0.0, 1.0)

    lapse = pm.Beta("lapse", 1, 20)  # share of trials answered with a bare "yes"
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    log_lik_all = np.stack([set_log_lik(s) for s in unique])
    rl = log_lik_all[:, :N_RULES]
    rule_ev = logsumexp(rl, axis=1)
    rule_in = np.exp(rl - rule_ev[:, None]) @ MEMBERSHIP[:N_RULES]  # (sets, 100)
    il = log_lik_all[:, N_RULES:]
    int_mem = MEMBERSHIP[N_RULES:]
    width_ev = np.stack([_log_group(il[i], 1.0) for i in range(len(unique))])
    cells = {}
    cell_set, rule_cell, width_member, row_cell = [], [], [], []
    for s, r in zip(sets, rows):
        i, t = index[s], int(r["target"]) - 1
        if (i, t) not in cells:
            cells[(i, t)] = len(cells)
            cell_set.append(i)
            rule_cell.append(rule_in[i, t])
            width_member.append(_log_group(il[i], int_mem[:, t]))
        row_cell.append(cells[(i, t)])
    return {
        "rule_log_evidence": rule_ev,
        "rule_in_concept": np.array(rule_cell),
        "width_log_evidence": width_ev,
        "width_log_member": np.stack(width_member),
        "cell_set": np.array(cell_set, dtype="int64"),
        "row_cell": np.array(row_cell, dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
