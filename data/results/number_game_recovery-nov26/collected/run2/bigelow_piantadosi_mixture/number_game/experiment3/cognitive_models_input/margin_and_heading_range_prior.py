"""Margin-and-heading range prior (refines margin_beyond_examples_range_prior).

The incumbent: Bayesian concept learning over 45 arithmetic and digit rules, the
18 "digits add up to s" concepts (an inferred family share) and every interval
of 1..100, distinct examples drawn without replacement with 10% strays, a
guessing lapse, a width-graded range prior, an inferred discount on open-ended
ranges (starting at 1 or ending at 100), and a margin credit
exp(beta * (room below + room above) / 100) for how far a range reaches past
the examples. The one change, taken from open_ended_ranges_heading_direction:
people read the direction of the last step (final example up or down from the
one before) as where the program is heading, so the expected room is tilted
toward it, multiplying a range's weight by
exp(eta * (room ahead - room behind) / 100); the prior is normalised over
ranges per set. Single-example sets get no tilt.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
N_TERMS = 14  # Taylor terms; |x| <= 1 (room-difference) or <= 0.5 (centred room): error < 1e-4 for |coef| < 3


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
DIGIT_SUM = np.array([sum(int(c) for c in str(n)) for n in NUMBERS])
SUM_CONCEPTS = [list(NUMBERS[DIGIT_SUM == s]) for s in range(1, 19)]
N_SUMS = len(SUM_CONCEPTS)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + SUM_CONCEPTS + INTERVALS], dtype="float64")
SIZE = MEMBERSHIP.sum(axis=1)
N_FINITE = N_RULES + N_SUMS
INTERVAL_MEMBERS = MEMBERSHIP[N_FINITE:]
INTERVAL_LO = np.array([iv[0] for iv in INTERVALS], dtype="float64")
INTERVAL_HI = np.array([iv[-1] for iv in INTERVALS], dtype="float64")
INTERVAL_WIDTH = (INTERVAL_HI - INTERVAL_LO + 1).astype(int)
INTERVAL_OPEN = ((INTERVAL_LO == 1) | (INTERVAL_HI == 100)).astype(int)
# Intervals are grouped by (width, open-ended flag); only non-empty groups kept.
_GROUP_KEYS = sorted(set(zip(INTERVAL_WIDTH.tolist(), INTERVAL_OPEN.tolist())))
_GROUP_OF = {g: i for i, g in enumerate(_GROUP_KEYS)}
INTERVAL_GROUP = np.array([_GROUP_OF[(w, o)] for w, o in zip(INTERVAL_WIDTH, INTERVAL_OPEN)])
N_GROUPS = len(_GROUP_KEYS)
GROUP_WIDTH = np.array([g[0] for g in _GROUP_KEYS], dtype="float64")
GROUP_OPEN = np.array([g[1] for g in _GROUP_KEYS], dtype="float64")
GROUP_ONEHOT = (INTERVAL_GROUP[:, None] == np.arange(N_GROUPS)[None, :]).astype("float64")
FACTORIAL = np.array([math.factorial(k) for k in range(N_TERMS)], dtype="float64")


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


def heading(examples):
    return int(np.sign(examples[-1] - examples[-2])) if len(examples) > 1 else 0


def room_classes(examples):
    """Per interval: room ahead / behind (in the heading direction; up if none), / 100.

    Ranges split into three classes, each needing a one-variable Taylor series:
    room only ahead (or none): factor exp((beta + eta) * ahead), series in ahead - 1/2;
    room only behind: exp((beta - eta) * behind), series in behind - 1/2;
    room on both sides (the range covers the examples): ahead + behind = (width - 1 - span) / 100
    is fixed by the group, so exp(beta * that) * exp(eta * (ahead - behind)), series in ahead - behind.
    """
    above = np.clip(INTERVAL_HI - max(examples), 0, None) / 100.0
    below = np.clip(min(examples) - INTERVAL_LO, 0, None) / 100.0
    ahead, behind = (below, above) if heading(examples) < 0 else (above, below)
    both = (ahead > 0) & (behind > 0)
    only_ahead = behind == 0
    only_behind = (ahead == 0) & (behind > 0)
    k = np.arange(N_TERMS)[None, :]
    powers_a = only_ahead[:, None] * (ahead - 0.5)[:, None] ** k
    powers_b = only_behind[:, None] * (behind - 0.5)[:, None] ** k
    powers_c = both[:, None] * (ahead - behind)[:, None] ** k
    return np.stack([powers_a, powers_b], axis=1), powers_c  # (intervals, 2, k), (intervals, k)


def series(c):
    """c^k / k! for k < N_TERMS (k = 0 kept apart: d(c^0)/dc is NaN at c = 0)."""
    return pt.concatenate([pt.ones(1), c ** np.arange(1, N_TERMS)]) / FACTORIAL


def block_sums(gw, beta, eta, ab_m, c_m, span):
    """Sum over intervals of (moment-weighted) range prior factor, from the three classes."""
    a, b = beta + eta, beta - eta
    coef_ab = gw[:, None, None] * pt.stack([pt.exp(a / 2) * series(a), pt.exp(b / 2) * series(b)])[None]
    coef_c = (gw * pt.exp(beta * (GROUP_WIDTH - 1) / 100))[:, None] * series(eta)[None, :]
    return pt.dot(ab_m, coef_ab.reshape((-1,))) + pt.dot(c_m, coef_c.reshape((-1,))) * pt.exp(-beta * span / 100)


with pm.Model() as model:
    rule_log_evidence = pm.Data("rule_log_evidence", np.zeros(1))  # (sets,), heading sets first
    sum_log_evidence = pm.Data("sum_log_evidence", np.zeros(1))  # (sets,)
    interval_log_scale = pm.Data("interval_log_scale", np.zeros(1))  # (sets,)
    rule_in_cell = pm.Data("rule_in_cell", np.zeros(1))  # (cells,), cells of heading sets first
    sum_in_cell = pm.Data("sum_in_cell", np.zeros(1))  # (cells,)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    moments = {}
    for part in ["head", "flat"]:  # sets with a last-step heading / single-example sets
        for kind, width in [("prior", "set"), ("evidence", "set"), ("member", "cell")]:
            moments[part, kind, "ab"] = pm.Data(f"{part}_{kind}_ab", np.zeros((1, N_GROUPS * 2 * N_TERMS)))
            moments[part, kind, "c"] = pm.Data(f"{part}_{kind}_c", np.zeros((1, N_GROUPS * N_TERMS)))
        moments[part, "set_span"] = pm.Data(f"{part}_set_span", np.zeros(1))
        moments[part, "cell_span"] = pm.Data(f"{part}_cell_span", np.zeros(1))
        moments[part, "cell_set"] = pm.Data(f"{part}_cell_set", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 1.0)  # width bias of the range prior
    log_sum_share = pm.Normal("log_sum_share", -3.0, 1.5)
    log_open = pm.Normal("log_open", 0.0, 1.0)  # open-ended vs bounded range
    beta = pm.Normal("beta", 0.0, 1.0)  # credit for a range's reach beyond the examples
    eta = pm.Normal("eta", 0.0, 1.0)  # tilt of that reach toward the last-step heading

    group_w = pt.exp(gamma * np.log(GROUP_WIDTH) + log_open * GROUP_OPEN)
    log_ev_parts, in_cell_parts = [], []
    for part, tilt in [("head", eta), ("flat", pt.zeros(()))]:
        sums = {
            kind: block_sums(
                group_w,
                beta,
                tilt,
                moments[part, kind, "ab"],
                moments[part, kind, "c"],
                moments[part, "cell_span" if kind == "member" else "set_span"],
            )
            for kind in ["prior", "evidence", "member"]
        }
        log_ev_parts.append(pt.log(sums["evidence"]) - pt.log(sums["prior"]))
        in_cell_parts.append(sums["member"] / sums["evidence"][moments[part, "cell_set"]])
    interval_log_evidence = pt.concatenate(log_ev_parts) + interval_log_scale
    interval_in_cell = pt.concatenate(in_cell_parts)

    log_rules_only = rule_log_evidence - np.log(N_RULES)
    log_sums_only = sum_log_evidence - np.log(N_SUMS) + log_sum_share
    p_sum = pm.math.sigmoid(log_sums_only - log_rules_only)[cell_set]
    finite_in_cell = (1 - p_sum) * rule_in_cell + p_sum * sum_in_cell
    log_finite = (
        pt.log(lambda_0) + pt.logaddexp(log_rules_only, log_sums_only) - pt.logaddexp(0.0, log_sum_share)
    )
    log_intervals = pt.log1p(-lambda_0) + interval_log_evidence
    p_rule = pm.math.sigmoid(log_finite - log_intervals)[cell_set]
    in_concept = p_rule * finite_in_cell + (1 - p_rule) * interval_in_cell

    p_concept = pt.clip(in_concept[row_cell], 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = sorted(dict.fromkeys(sets), key=lambda s: heading(s) == 0)  # heading sets first (stable)
    set_index = {s: i for i, s in enumerate(unique)}
    keys = [(set_index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = sorted(dict.fromkeys(keys), key=lambda c: c[0])
    cell_index = {c: i for i, c in enumerate(cells)}
    n_sets, n_cells = len(unique), len(cells)
    is_head = np.array([heading(s) != 0 for s in unique])
    rule_ev, sum_ev, scale, span = np.zeros(n_sets), np.zeros(n_sets), np.zeros(n_sets), np.zeros(n_sets)
    rule_in_set = np.zeros((n_sets, 100))
    sum_in_set = np.zeros((n_sets, 100))
    prior_ab = np.zeros((n_sets, N_GROUPS, 2, N_TERMS))
    prior_c = np.zeros((n_sets, N_GROUPS, N_TERMS))
    evid_ab, evid_c = np.zeros_like(prior_ab), np.zeros_like(prior_c)
    member_ab = np.zeros((n_cells, N_GROUPS, 2, N_TERMS))
    member_c = np.zeros((n_cells, N_GROUPS, N_TERMS))
    onehot_t = GROUP_ONEHOT.T
    for i, s in enumerate(unique):
        ll = set_log_lik(s)
        rule_ll, sum_ll, int_ll = ll[:N_RULES], ll[N_RULES:N_FINITE], ll[N_FINITE:]
        rule_ev[i] = logsumexp(rule_ll)
        sum_ev[i] = logsumexp(sum_ll)
        rule_in_set[i] = np.exp(rule_ll - rule_ev[i]) @ MEMBERSHIP[:N_RULES]
        sum_in_set[i] = np.exp(sum_ll - sum_ev[i]) @ MEMBERSHIP[N_RULES:N_FINITE]
        scale[i] = int_ll.max()
        span[i] = max(s) - min(s)
        p_ab, p_c = room_classes(s)
        weight = np.exp(int_ll - scale[i])
        prior_ab[i] = np.einsum("gi,ijk->gjk", onehot_t, p_ab)
        prior_c[i] = onehot_t @ p_c
        w_ab, w_c = weight[:, None, None] * p_ab, weight[:, None] * p_c
        evid_ab[i] = np.einsum("gi,ijk->gjk", onehot_t, w_ab)
        evid_c[i] = onehot_t @ w_c
        for t in [t for (si, t) in cells if si == i]:
            inside = INTERVAL_MEMBERS[:, t - 1] > 0
            member_ab[cell_index[(i, t)]] = np.einsum("gi,ijk->gjk", onehot_t[:, inside], w_ab[inside])
            member_c[cell_index[(i, t)]] = onehot_t[:, inside] @ w_c[inside]
    cell_set = np.array([c[0] for c in cells], dtype="int64")
    cell_target = np.array([c[1] - 1 for c in cells], dtype="int64")
    n_hs = int(is_head.sum())
    head_cell = cell_set < n_hs
    out = {
        "rule_log_evidence": rule_ev,
        "sum_log_evidence": sum_ev,
        "interval_log_scale": scale,
        "rule_in_cell": rule_in_set[cell_set, cell_target],
        "sum_in_cell": sum_in_set[cell_set, cell_target],
        "cell_set": cell_set,
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
    for part, sset, cset, offset in [
        ("head", slice(0, n_hs), head_cell, 0),
        ("flat", slice(n_hs, n_sets), ~head_cell, n_hs),
    ]:
        out[f"{part}_prior_ab"] = prior_ab[sset].reshape(-1, N_GROUPS * 2 * N_TERMS)
        out[f"{part}_prior_c"] = prior_c[sset].reshape(-1, N_GROUPS * N_TERMS)
        out[f"{part}_evidence_ab"] = evid_ab[sset].reshape(-1, N_GROUPS * 2 * N_TERMS)
        out[f"{part}_evidence_c"] = evid_c[sset].reshape(-1, N_GROUPS * N_TERMS)
        out[f"{part}_member_ab"] = member_ab[cset].reshape(-1, N_GROUPS * 2 * N_TERMS)
        out[f"{part}_member_c"] = member_c[cset].reshape(-1, N_GROUPS * N_TERMS)
        out[f"{part}_set_span"] = span[sset]
        out[f"{part}_cell_span"] = span[cell_set[cset]]
        out[f"{part}_cell_set"] = cell_set[cset] - offset
    return out


SAMPLER_SETTINGS = {"target_accept": 0.9}
