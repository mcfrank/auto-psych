"""Nearest-example radius concepts: similarity neighbourhoods replace ranges.

People's magnitude-based guesses about the program are similarity
neighbourhoods around the examples, not stretches of the number line: besides
the 45 arithmetic and digit rules (final-digit rules boosted by kappa) and
"rule plus an example-anchored neighbourhood" concepts emitted through a side
channel, the only numerical concept is "numbers within distance r of one of the
examples" (the union of radius-r neighbourhoods of the examples), with a
gamma-shaped prior on r + 1 (smaller radii likelier), members sampled along a
compressed number line (weight x^-gamma), and the same tempered (beta),
exception-tolerant (eps) size principle. Endorsement of a non-rule target thus
falls off with its distance to the nearest example; gaps between distant
examples are not filled in.
"""


import numpy as np
import pymc as pm
import pytensor.tensor as pt

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


RULE_NAMES = list(rule_concepts().keys())
RULES = list(rule_concepts().values())
RULE_SALIENT = np.array([n in ("odd", "even") or n.startswith("ends in") for n in RULE_NAMES], dtype="float64")
N_RULES = len(RULES)
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
LOG_NUMBERS = np.log(NUMBERS.astype("float64"))
N_RADII = 100
RADII = np.arange(N_RADII, dtype="float64")
MAX_EX = 4
H_MAX = 3
H_RADII = np.arange(H_MAX + 1, dtype="float64")


def _scatter_sum(n, index, values):
    return pt.inc_subtensor(pt.zeros(n)[index], values)


with pm.Model() as model:
    rule_k = pm.Data("rule_k", np.zeros((1, N_RULES)))
    set_n = pm.Data("set_n", np.ones(1))
    nb_member = pm.Data("nb_member", np.zeros((N_RADII, 100)))
    ex_log_x = pm.Data("ex_log_x", np.zeros((1, MAX_EX)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAX_EX)))
    cell_dmin = pm.Data("cell_dmin", np.zeros(1, dtype="int64"))
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    cls_set = pm.Data("cls_set", np.zeros(1, dtype="int64"))
    cls_rule = pm.Data("cls_rule", np.zeros(1, dtype="int64"))
    cls_h = pm.Data("cls_h", np.zeros(1, dtype="int64"))
    cls_extra_n = pm.Data("cls_extra_n", np.ones(1))
    cls_e = pm.Data("cls_e", np.zeros(1))
    cls_m = pm.Data("cls_m", np.ones(1))
    chit_cell = pm.Data("chit_cell", np.zeros(1, dtype="int64"))
    chit_cls = pm.Data("chit_cls", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    beta = pm.LogNormal("beta", mu=0.0, sigma=0.5)
    eps = pm.Beta("eps", 2, 5)
    gamma = pm.HalfNormal("gamma", sigma=1.0)
    mean_radius = pm.LogNormal("mean_radius", mu=np.log(5.0), sigma=1.0)  # typical similarity radius (+1)
    radius_shape = pm.LogNormal("radius_shape", mu=0.0, sigma=0.75)  # gamma shape of the radius prior
    kappa = pm.Normal("kappa", mu=0.0, sigma=1.5)
    alpha = pm.Beta("alpha", 1, 4)
    rho = pm.LogNormal("rho", mu=0.0, sigma=1.0)
    omega = pm.Beta("omega", 2, 6)

    n_sets = set_n.shape[0]
    n_cells = cell_set.shape[0]
    log_noise = pt.log(eps) - np.log(100)
    rule_in = pt.log((1 - eps) / RULE_SIZE + eps / 100)
    rule_log_lik = rule_k * rule_in[None, :] + (set_n[:, None] - rule_k) * log_noise
    rule_score = kappa * RULE_SALIENT
    log_rule_prior = rule_score - pm.math.logsumexp(rule_score)
    has_off = pt.gt(set_n[:, None] - rule_k, 0.5)
    log_rules = pt.log(lambda_0) + pt.log1p(-alpha * has_off) + log_rule_prior[None, :] + beta * rule_log_lik

    # Neighbourhood concepts: members within r of an example, drawn with weight x^-gamma.
    log_r = (radius_shape - 1) * pt.log(RADII + 1) - (RADII + 1) * radius_shape / mean_radius
    log_r_prior = pt.log1p(-lambda_0) + log_r - pm.math.logsumexp(log_r)
    x_w = pt.exp(-gamma * LOG_NUMBERS)
    w_tot = pt.dot(nb_member, x_w).reshape((n_sets, N_RADII))  # (sets, radii)
    ex_in = pt.log((1 - eps) * pt.exp(-gamma * ex_log_x)[:, None, :] / w_tot[:, :, None] + eps / 100)
    nb_lik = (ex_in * ex_mask[:, None, :]).sum(axis=2)
    log_nb = log_r_prior[None, :] + beta * nb_lik
    shift = pt.maximum(pt.max(log_rules, axis=1), pt.max(log_nb, axis=1))
    w_rules = pt.exp(log_rules - shift[:, None])

    # Rule plus anchored neighbourhood, two channels (rule w.p. 1 - omega, neighbourhood w.p. omega).
    log_h_prior = -rho * H_RADII - pm.math.logsumexp(-rho * H_RADII)
    side_rule_in = pt.log((1 - eps) * (1 - omega) / RULE_SIZE + eps / 100)
    cls_x_in = pt.log((1 - eps) * omega / cls_extra_n + eps / 100)
    k_rule = rule_k[cls_set, cls_rule]
    log_cls = (
        pt.log(lambda_0) + pt.log(alpha) + log_rule_prior[cls_rule] + log_h_prior[cls_h]
        - pt.log(cls_m)
        + beta * (
            k_rule * side_rule_in[cls_rule]
            + cls_e * cls_x_in
            + (set_n[cls_set] - k_rule - cls_e) * log_noise
        )
    )
    w_cls = pt.exp(log_cls - shift[cls_set])
    w_aug = _scatter_sum(n_sets * N_RULES, cls_set * N_RULES + cls_rule, w_cls).reshape((n_sets, N_RULES))

    w_nb = pt.exp(log_nb - shift[:, None])
    w_nb_tail = pt.cumsum(w_nb[:, ::-1], axis=1)[:, ::-1]  # sum over radii >= r
    covered = w_nb_tail[cell_set, cell_dmin]

    z = w_rules.sum(axis=1) + w_aug.sum(axis=1) + w_nb.sum(axis=1)
    numer = (
        (w_rules[cell_set] * cell_rule_member).sum(axis=1)
        + (w_aug[cell_set] * cell_rule_member).sum(axis=1)
        + _scatter_sum(n_cells, chit_cell, w_cls[chit_cls])
        + covered
    )
    in_concept = numer / z[cell_set]

    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept[row_cell], 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    trial_cells = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(trial_cells))
    cell_index = {c: i for i, c in enumerate(cells)}

    nb_member, ex_log_x, ex_mask = [], [], []
    for st in unique:
        dist = np.abs(NUMBERS[:, None] - np.array(st)[None, :]).min(axis=1)
        nb_member.append((dist[None, :] <= RADII[:, None]).astype("float64"))
        lx = np.zeros(MAX_EX)
        lx[: len(st)] = LOG_NUMBERS[np.array(st) - 1]
        ex_log_x.append(lx)
        ex_mask.append((np.arange(MAX_EX) < len(st)).astype("float64"))
    cell_dmin = [min(abs(t - x) for x in unique[i]) for i, t in cells]

    cls_rows, cls_extra, set_cls = [], [], []
    for i, st in enumerate(unique):
        ex_count = np.bincount(np.array(st), minlength=101)[1:]
        mine = []
        for r in range(N_RULES):
            in_rule = RULE_MEMBERSHIP[r] > 0
            centres = sorted({x for x in st if not in_rule[x - 1]})
            for x in centres:
                for h in range(H_MAX + 1):
                    extra = np.arange(max(1, x - h), min(100, x + h) + 1)
                    extra = extra[~in_rule[extra - 1]]
                    mine.append(len(cls_rows))
                    cls_rows.append((i, r, h, len(extra), ex_count[extra - 1].sum(), len(centres)))
                    cls_extra.append(set(extra.tolist()))
        set_cls.append(mine)
    chit_cell, chit_cls = [], []
    for c, (i, t) in enumerate(cells):
        for j in set_cls[i]:
            if t in cls_extra[j]:
                chit_cell.append(c)
                chit_cls.append(j)
    cls = np.array(cls_rows, dtype="float64")

    return {
        "rule_k": np.stack([RULE_MEMBERSHIP[:, np.array(s) - 1].sum(axis=1) for s in unique]),
        "set_n": np.array([float(len(s)) for s in unique]),
        "nb_member": np.concatenate(nb_member, axis=0),
        "ex_log_x": np.array(ex_log_x),
        "ex_mask": np.array(ex_mask),
        "cell_dmin": np.array(cell_dmin, dtype="int64"),
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "row_cell": np.array([cell_index[c] for c in trial_cells], dtype="int64"),
        "cls_set": cls[:, 0].astype("int64"),
        "cls_rule": cls[:, 1].astype("int64"),
        "cls_h": cls[:, 2].astype("int64"),
        "cls_extra_n": cls[:, 3],
        "cls_e": cls[:, 4],
        "cls_m": cls[:, 5],
        "chit_cell": np.array(chit_cell, dtype="int64"),
        "chit_cls": np.array(chit_cls, dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
