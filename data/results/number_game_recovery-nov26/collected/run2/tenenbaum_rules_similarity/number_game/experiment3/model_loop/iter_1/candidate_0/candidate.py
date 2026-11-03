"""People do not weigh competing concepts against each other; they check each
familiar regularity (an arithmetic or final-digit rule, or a stretch of the
number line) on its own and treat it as a real constraint of the program when
the examples sharing it would be too much of a coincidence by chance -- a graded
significance judgement driven by how rare the regularity is among 1-100 and how
many examples show it, with at most one example allowed to break it. They then
say "yes" only to a target that satisfies every regularity they have accepted
(a conjunction of independently detected constraints).

Coincidence evidence of a regularity C for examples x: n_in * log(100/|C|) minus
a penalty per example outside C. Detection probability is
sigmoid(slope * (evidence - criterion)); the target passes C if it is in C or C
was not detected; p(pass all) is a product over regularities (no competition,
no prior over concepts). The answer is "yes" with a high rate when the target
passes everything and a low rate otherwise.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_EX = 4


def _rules():
    rules = [
        [n for n in NUMBERS if n % 2 == 1],
        list(range(2, 101, 2)),
        [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n ** 0.5) + 1))],
        [n * n for n in range(1, 11)],
        [n ** 3 for n in range(1, 5)],
    ]
    rules += [[m ** k for k in range(1, 8) if m ** k <= 100] for m in range(2, 11)]
    rules += [list(range(m, 101, m)) for m in list(range(3, 10)) + list(range(11, 51))]
    rules += [[n for n in NUMBERS if n % 10 == d] for d in range(10)]
    out = []
    for r in rules:
        if sorted(r) not in out:
            out.append(sorted(r))
    return out


RULES = _rules()
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
# Intervals grouped by (size, examples inside k).
N_GROUPS = 100 * (MAX_EX + 1)
GROUP_SIZE = np.repeat(np.arange(1, 101), MAX_EX + 1).astype("float64")
GROUP_NIN = np.tile(np.arange(MAX_EX + 1), 100).astype("float64")


def prepare_observed(rows):
    """Per set: example counts inside each regularity; per (set, target) cell: which
    regularities exclude the target (interval counts grouped by size and k), restricted
    to regularities covering all but at most one example (and at least one)."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    rule_n_in = np.zeros((S, N_RULES))
    n_ex = np.zeros(S)
    rule_ok = np.zeros((S, N_RULES))
    excl = np.zeros((S, N_GROUPS, 100))
    for i, s in enumerate(unique):
        n = len(s)
        n_ex[i] = n
        rule_n_in[i] = RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1)
        rule_ok[i] = (rule_n_in[i] >= max(n - 1, 1)).astype(float)
        for a in range(1, 101):
            for b in range(a, 101):
                k = int(sum(a <= x <= b for x in s))
                if k < max(n - 1, 1):
                    continue
                g = (b - a) * (MAX_EX + 1) + min(k, MAX_EX)
                excl[i, g, : a - 1] += 1
                excl[i, g, b:] += 1
    keys = [(index[s], int(r["target"]) - 1) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {k: j for j, k in enumerate(cells)}
    cs = np.array([k[0] for k in cells], dtype="int64")
    ct = np.array([k[1] for k in cells], dtype="int64")
    return {
        "rule_n_in": rule_n_in,
        "n_examples": n_ex,
        "cell_set": cs,
        "cell_rule_excl": (1.0 - RULE_MEMBERSHIP[:, ct].T) * rule_ok[cs],
        "cell_group_excl": excl[cs, :, ct],
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    rule_n_in = pm.Data("rule_n_in", np.zeros((1, N_RULES)))
    n_examples = pm.Data("n_examples", np.ones(1))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_excl = pm.Data("cell_rule_excl", np.zeros((1, N_RULES)))
    cell_group_excl = pm.Data("cell_group_excl", np.zeros((1, N_GROUPS)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Criterion (nats of coincidence) for accepting a regularity, slope of the
    # graded judgement, and the evidence cost of one example breaking it.
    criterion = pm.Normal("criterion", mu=4.0, sigma=2.0)
    slope = pm.LogNormal("slope", mu=0.0, sigma=0.75)
    exception_cost = pm.HalfNormal("exception_cost", sigma=3.0)
    p_hi = pm.Beta("p_hi", 4.0, 2.0)
    p_lo = pm.Beta("p_lo", 1.0, 8.0)

    rule_ev = rule_n_in * np.log(100.0 / RULE_SIZE)[None, :] - (n_examples[:, None] - rule_n_in) * exception_cost
    group_ev = GROUP_NIN[None, :] * np.log(100.0 / GROUP_SIZE)[None, :] - (n_examples[:, None] - GROUP_NIN[None, :]) * exception_cost
    # log(1 - sigmoid(z)) = -softplus(z)
    rule_log_pass = -pt.softplus(slope * (rule_ev - criterion))
    group_log_pass = -pt.softplus(slope * (group_ev - criterion))
    log_pass_all = (cell_rule_excl * rule_log_pass[cell_set]).sum(axis=1) + (
        cell_group_excl * group_log_pass[cell_set]
    ).sum(axis=1)
    pass_all = pt.exp(log_pass_all)[row_cell]

    p_yes = pm.Deterministic("p_yes", pt.clip(p_lo + (p_hi - p_lo) * pass_all, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)
