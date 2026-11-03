"""Range location gradient prior: a refinement of narrow_range_specificity_bias.

People do Bayesian concept learning over 45 arithmetic and digit rules (equal
prior) and every interval of 1..100. Examples are distinct draws from the
concept's members not yet shown (sampling without replacement), 10% of them
stray, and there is a guessing lapse. As in the incumbent, the range prior is
biased by width: an interval of size s has prior weight proportional to
s**gamma. The one change is that the range prior also depends on where the
range sits on the number line. Its log prior rises (beta > 0) or falls
(beta < 0) linearly with its lower bound, so ranges starting high up are
judged more (or less) plausible than equally wide ranges starting low. This
makes generalization from a cluster of examples asymmetric about it.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)
N_TERMS = 24  # Taylor terms for the lower-bound factor within each width


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
INTERVAL_MEMBERS = MEMBERSHIP[N_RULES:]
LO = np.array([iv[0] for iv in INTERVALS], dtype="float64")
WIDTH = SIZE[N_RULES:].astype(int)
WIDTHS = np.arange(1, 101, dtype="float64")
# Lower bound on a 0..1 scale, written as a per-width centre plus an offset
# u in [-0.5, 0.5]: exp(beta * lower) = exp(beta * centre) * sum_k (beta u)^k / k!.
CENTRE = ((102.0 - WIDTHS) / 2.0 - 1.0) / 99.0
U = (LO - 1.0) / 99.0 - CENTRE[WIDTH - 1]
U_POW = np.stack([U**k / math.factorial(k) for k in range(N_TERMS)], axis=1)  # (intervals, terms)
LOWER = (LO - 1.0) / 99.0
LOG_SIZE = np.log(WIDTH.astype("float64"))


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


def width_moments(weights):
    """Sum weights * u^k / k! within each width: (rows, intervals) -> (rows, widths * terms)."""
    out = np.zeros((len(weights), 100, N_TERMS))
    for w in range(1, 101):
        mask = WIDTH == w
        out[:, w - 1] = weights[:, mask] @ U_POW[mask]
    return out.reshape(len(weights), -1)


# The likelihoods are parameter-free. prepare_observed reduces the intervals
# to Taylor moments per (set, width) for the evidence and per (set, target)
# cell for the mass of the intervals containing the target, so the model only
# needs two matrix-vector products with parameter-dependent coefficients.

with pm.Model() as model:
    rule_log_evidence = pm.Data("rule_log_evidence", np.zeros(1))  # (sets,)
    interval_shift = pm.Data("interval_shift", np.zeros(1))  # (sets,) max interval log lik
    set_moments = pm.Data("set_moments", np.zeros((1, 100 * N_TERMS)))  # (sets, widths * terms)
    cell_moments = pm.Data("cell_moments", np.zeros((1, 100 * N_TERMS)))  # (cells, widths * terms)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_rule_in = pm.Data("cell_rule_in", np.zeros(1))  # (cells,) rule-posterior membership of the target
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_0 = pm.Beta("lambda_0", 2, 2)
    gamma = pm.Normal("gamma", 0.0, 1.0)  # width bias of the range prior
    beta = pm.Normal("beta", 0.0, 2.0)  # location gradient of the range prior (per lower bound 1 -> 100)

    # Range prior over all 5050 intervals: size**gamma * exp(beta * lower), normalised.
    log_z = pt.logsumexp(gamma * LOG_SIZE + beta * LOWER)
    log_width_part = gamma * np.log(WIDTHS) + beta * CENTRE - log_z  # (widths,)
    powers = [pt.ones(())]
    for _ in range(N_TERMS - 1):
        powers.append(powers[-1] * beta)
    coeff = (pt.exp(log_width_part)[:, None] * pt.stack(powers)[None, :]).reshape((-1,))

    set_sum = pt.maximum(pt.dot(set_moments, coeff), 1e-300)
    interval_log_evidence = pt.log(set_sum) + interval_shift
    interval_in_concept = pt.dot(cell_moments, coeff) / set_sum[cell_set]

    log_rules = pt.log(lambda_0) - np.log(N_RULES) + rule_log_evidence
    log_intervals = pt.log1p(-lambda_0) + interval_log_evidence
    p_rule = pm.math.sigmoid(log_rules - log_intervals)[cell_set]
    in_concept = p_rule * cell_rule_in + (1 - p_rule) * interval_in_concept

    p_concept = pt.clip(in_concept[row_cell], 0.0, 1.0)
    lapse = pm.Beta("lapse", 2, 8)
    guess_yes = pm.Beta("guess_yes", 2, 2)
    p_yes = pm.Deterministic("p_yes", pt.clip((1 - lapse) * p_concept + lapse * guess_yes, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target` and `rating`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    log_lik_all = np.stack([set_log_lik(s) for s in unique])
    rule_ll = log_lik_all[:, :N_RULES]
    rule_ev = logsumexp(rule_ll, axis=1)
    rule_in = np.exp(rule_ll - rule_ev[:, None]) @ MEMBERSHIP[:N_RULES]
    interval_ll = log_lik_all[:, N_RULES:]
    shift = interval_ll.max(axis=1)
    weights = np.exp(interval_ll - shift[:, None])
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    c_set = np.array([c[0] for c in cells], dtype="int64")
    c_target = np.array([c[1] for c in cells], dtype="int64") - 1
    return {
        "rule_log_evidence": rule_ev,
        "interval_shift": shift,
        "set_moments": width_moments(weights),
        "cell_moments": width_moments(weights[c_set] * INTERVAL_MEMBERS[:, c_target].T),
        "cell_set": c_set,
        "cell_rule_in": rule_in[c_set, c_target],
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


SAMPLER_SETTINGS = {"target_accept": 0.9}
