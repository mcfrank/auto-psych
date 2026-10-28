"""Relational analogy transfer.

People reason about the examples relationally, by analogy, as well as by
surface similarity: they notice how the examples relate to each other (the
difference or the ratio between two examples) and ask whether the target
relates to some example in the same way. A target reached by applying an
observed relation to another example (40 after 20, 30; 32 after 8, 16) gets
extra support for "yes" beyond its feature similarity. Everything else (the
exemplar similarity with leading-digit match, rules, list concept, span,
momentum, zigzag breadth, criterion relaxation and upward start boundary) is
as in leading_digit_string_similarity.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4
NDIM = 5  # magnitude, final digit, parity, graded shared factor, leading digit
SESSION_LEN = 450.0


def _factor_mismatch(t, x):
    m = min(t, x)
    if m <= 1:
        return 0.0
    return 1.0 - math.log(math.gcd(t, x)) / math.log(m)


def _relational_match(ex, t):
    """How closely t matches a new number made by carrying a relation seen
    between two examples (their difference or ratio) over to an example."""
    if len(ex) < 2:
        return 0.0
    best = 0.0
    for i, a in enumerate(ex):
        for j, b in enumerate(ex):
            if i == j:
                continue
            for x in ex:
                y = x + b - a
                if y not in ex:
                    best = max(best, math.exp(-abs(y - t) / 1.0))
                q = x * b / a
                if not any(abs(q - e) < 1e-9 for e in ex):
                    best = max(best, math.exp(-abs(math.log(t) - math.log(q)) / 0.03))
    return best


def prepare_observed(rows):
    n = len(rows)
    mism = np.ones((n, NDIM, MAXK))
    coh = np.zeros((n, NDIM))
    lag = np.zeros((n, MAXK))
    mask = np.zeros((n, MAXK))
    rule = np.zeros(n)
    digit_rule = np.zeros(n)
    position = np.zeros(n)
    ahead = np.zeros(n)
    below_start = np.zeros(n)
    disorder = np.zeros(n)
    is_example = np.zeros(n)
    rating = np.zeros(n, dtype="int64")
    relation = np.zeros(n)
    cache = {}
    for r, row in enumerate(rows):
        s = str(row["number_set"])
        ex = [int(x) for x in s.split(",")][:MAXK]
        t = int(row["target"])
        k = len(ex)
        if s not in cache:
            pairs = [(ex[i], ex[j]) for i in range(k) for j in range(i + 1, k)]
            c = np.zeros(NDIM)
            if pairs:
                c[0] = np.mean([max(0.0, 1.0 - abs(a - b) / 50.0) for a, b in pairs])
                c[1] = np.mean([a % 10 == b % 10 for a, b in pairs])
                c[2] = np.mean([a % 2 == b % 2 for a, b in pairs])
                c[3] = np.mean([1.0 - _factor_mismatch(a, b) for a, b in pairs])
                c[4] = np.mean([str(a)[0] == str(b)[0] for a, b in pairs])
            g = reduce(math.gcd, ex) if k >= 2 else 0
            d = ex[0] % 10 if k >= 2 and len({x % 10 for x in ex}) == 1 else -1
            cache[s] = (c, g, d)
        c, g, d = cache[s]
        if d >= 0:
            digit_rule[r] = (k - 1) * math.log(10.0) * (1.0 if t % 10 == d else -1.0)
        coh[r] = c
        # Analogical transfer of an observed relation (the change).
        relation[r] = _relational_match(ex, t)
        is_example[r] = 1.0 if t in ex else 0.0
        if g >= 3:
            rule[r] = (k - 1) * math.log(g) * (1.0 if t % g == 0 else -1.0)
        # +1 beyond the last example in the direction of travel, -1 behind
        # the start, 0 inside the span or with no direction (one example).
        travel = (ex[-1] > ex[0]) - (ex[-1] < ex[0])
        out = (t > max(ex)) - (t < min(ex))
        ahead[r] = float(travel * out)
        # Below the starting point of an upward-counting sequence (the change).
        below_start[r] = 1.0 if (travel > 0 and t < ex[0]) else 0.0
        # Share of direction reversals between successive steps in shown
        # order (0 monotone, 1 full zigzag), centred; 0 with fewer than 3.
        if k >= 3:
            steps = [b - a for a, b in zip(ex, ex[1:])]
            revs = [(a > 0) != (b > 0) for a, b in zip(steps, steps[1:])]
            disorder[r] = float(np.mean(revs)) - 0.5
        for i, x in enumerate(ex):
            mism[r, 0, i] = max(0, min(ex) - t, t - max(ex)) / 100.0
            mism[r, 1, i] = 0.0 if t % 10 == x % 10 else 1.0
            mism[r, 2, i] = 0.0 if t % 2 == x % 2 else 1.0
            mism[r, 3, i] = _factor_mismatch(t, x)
            # Leading written digit match (the change).
            mism[r, 4, i] = 0.0 if str(t)[0] == str(x)[0] else 1.0
            lag[r, i] = k - 1 - i
            mask[r, i] = 1.0
        ti = row.get("trial_index")
        if ti is not None and str(ti).strip() != "":
            position[r] = min(max(float(ti), 0.0), SESSION_LEN) / SESSION_LEN
        rv = row.get("rating")
        rating[r] = int(float(rv)) if rv is not None and str(rv).strip() != "" else 0
    return {"mismatch": mism, "coherence": coh, "lag": lag, "ex_mask": mask,
            "rule_evidence": rule, "digit_rule_evidence": digit_rule,
            "session_position": position, "ahead": ahead, "below_start": below_start,
            "disorder": disorder, "is_example": is_example, "relation_match": relation,
            "rating": rating}


with pm.Model() as model:
    mismatch = pm.Data("mismatch", np.zeros((1, NDIM, MAXK)))
    coherence = pm.Data("coherence", np.zeros((1, NDIM)))
    lag = pm.Data("lag", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))
    rule_evidence = pm.Data("rule_evidence", np.zeros(1))
    digit_rule_evidence = pm.Data("digit_rule_evidence", np.zeros(1))
    session_position = pm.Data("session_position", np.zeros(1))
    ahead = pm.Data("ahead", np.zeros(1))
    below_start = pm.Data("below_start", np.zeros(1))
    disorder = pm.Data("disorder", np.zeros(1))
    is_example = pm.Data("is_example", np.zeros(1))
    relation_match = pm.Data("relation_match", np.zeros(1))

    w_base = pm.HalfNormal("w_base", sigma=np.array([10.0, 2.0, 2.0, 2.0, 2.0]))
    tuning = pm.HalfNormal("tuning", sigma=2.0, shape=NDIM)
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    rule_weight = pm.HalfNormal("rule_weight", sigma=1.0)
    digit_rule_weight = pm.HalfNormal("digit_rule_weight", sigma=1.0)
    # How far the yes-criterion relaxes from session start to end.
    criterion_drift = pm.Normal("criterion_drift", mu=0.0, sigma=1.0)
    # Time constant of the relaxation, as a share of the session:
    # small = fast early relaxation then plateau; large = nearly linear.
    relax_tau = pm.Beta("relax_tau", alpha=2.0, beta=4.0)
    relax_shape = ((1.0 - pt.exp(-session_position / relax_tau))
                   / (1.0 - pt.exp(-1.0 / relax_tau)))
    # Evidence from the "fixed list of these numbers" concept.
    list_weight = pm.HalfNormal("list_weight", sigma=2.0)

    # Evidence against numbers below an upward sequence's start (the change).
    start_penalty = pm.HalfNormal("start_penalty", sigma=2.0)

    # Support from a target matching an analogically transferred relation.
    analogy_weight = pm.HalfNormal("analogy_weight", sigma=1.0)

    # Momentum: shrinks size mismatch ahead of the sequence, stretches it behind.
    momentum = pm.Normal("momentum", mu=0.0, sigma=1.0)
    # Zigzag: a disorderly sequence shrinks the size mismatch.
    zigzag_sensitivity = pm.Normal("zigzag_sensitivity", mu=0.0, sigma=1.0)
    size_scale = pt.exp(-momentum * ahead - zigzag_sensitivity * disorder)
    mismatch = pt.set_subtensor(
        mismatch[:, 0, :], mismatch[:, 0, :] * size_scale[:, None])

    attn = w_base[None, :] * pt.exp(tuning[None, :] * coherence)
    dist = pt.sum(attn[:, :, None] * mismatch, axis=1)
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    familiarity = pt.sum(mem * sim, axis=1)

    logit = (bias + criterion_drift * relax_shape + gain * familiarity
             + rule_weight * rule_evidence
             + digit_rule_weight * digit_rule_evidence
             + list_weight * is_example
             - start_penalty * below_start
             + analogy_weight * relation_match)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(logit), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
