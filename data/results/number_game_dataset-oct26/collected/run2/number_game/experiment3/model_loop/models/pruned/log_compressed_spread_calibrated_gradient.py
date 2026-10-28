"""Log-compressed spread-calibrated size gradient (refinement of
spread_calibrated_size_gradient).

Everything in spread_calibrated_size_gradient is kept (recency-weighted
similarity on size, final digit, parity and graded shared factor with
coherence-stretched attention, the "multiples of g" / "ends in d" rules, the
relaxing criterion, the filled-in span, order-directed momentum and the
spread-calibrated yardstick). The single change: the distance by which a target
falls outside the examples' span is felt on a compressed, logarithm-like mental
number line, log(n + c) with the compression offset c inferred (large c
recovers the linear line), so equal linear gaps feel larger among small
numbers than among large ones and small-number sets are generalized more
narrowly beyond their span.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4
NDIM = 4  # magnitude, final digit, parity, graded shared factor
SESSION_LEN = 450.0


def _factor_mismatch(t, x):
    m = min(t, x)
    if m <= 1:
        return 0.0
    return 1.0 - math.log(math.gcd(t, x)) / math.log(m)


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
    spread = np.zeros(n)
    tgt = np.ones(n)
    edge = np.ones(n)
    side = np.zeros(n)
    rating = np.zeros(n, dtype="int64")
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
            g = reduce(math.gcd, ex) if k >= 2 else 0
            d = ex[0] % 10 if k >= 2 and len({x % 10 for x in ex}) == 1 else -1
            cache[s] = (c, g, d)
        c, g, d = cache[s]
        if d >= 0:
            digit_rule[r] = (k - 1) * math.log(10.0) * (1.0 if t % 10 == d else -1.0)
        coh[r] = c
        if g >= 3:
            rule[r] = (k - 1) * math.log(g) * (1.0 if t % g == 0 else -1.0)
        # +1 beyond the last example in the direction of travel, -1 behind
        # the start, 0 inside the span or with no direction (one example).
        travel = (ex[-1] > ex[0]) - (ex[-1] < ex[0])
        out = (t > max(ex)) - (t < min(ex))
        ahead[r] = float(travel * out)
        # Examples' spread (range) on the 0-1 scale; 0 for a single example.
        spread[r] = (max(ex) - min(ex)) / 100.0
        tgt[r] = float(t)
        edge[r] = float(min(ex) if out < 0 else max(ex))
        side[r] = float(abs(out))
        for i, x in enumerate(ex):
            mism[r, 0, i] = max(0, min(ex) - t, t - max(ex)) / 100.0
            mism[r, 1, i] = 0.0 if t % 10 == x % 10 else 1.0
            mism[r, 2, i] = 0.0 if t % 2 == x % 2 else 1.0
            mism[r, 3, i] = _factor_mismatch(t, x)
            lag[r, i] = k - 1 - i
            mask[r, i] = 1.0
        ti = row.get("trial_index")
        if ti is not None and str(ti).strip() != "":
            position[r] = min(max(float(ti), 0.0), SESSION_LEN) / SESSION_LEN
        rv = row.get("rating")
        rating[r] = int(float(rv)) if rv is not None and str(rv).strip() != "" else 0
    return {"mismatch": mism, "coherence": coh, "lag": lag, "ex_mask": mask,
            "rule_evidence": rule, "digit_rule_evidence": digit_rule,
            "session_position": position, "ahead": ahead,
            "spread": spread, "target_value": tgt, "span_edge": edge,
            "outside_span": side,
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
    spread = pm.Data("spread", np.zeros(1))
    target_value = pm.Data("target_value", np.ones(1))
    span_edge = pm.Data("span_edge", np.ones(1))
    outside_span = pm.Data("outside_span", np.zeros(1))

    w_base = pm.HalfNormal("w_base", sigma=np.array([10.0, 2.0, 2.0, 2.0]))
    tuning = pm.HalfNormal("tuning", sigma=2.0, shape=NDIM)
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    rule_weight = pm.HalfNormal("rule_weight", sigma=1.0)
    digit_rule_weight = pm.HalfNormal("digit_rule_weight", sigma=1.0)
    # How far the yes-criterion relaxes from session start to end (the change).
    criterion_drift = pm.Normal("criterion_drift", mu=0.0, sigma=1.0)

    # Momentum: shrinks size mismatch ahead of the sequence, stretches it behind.
    momentum = pm.Normal("momentum", mu=0.0, sigma=1.0)
    # Spread calibration (the change): outside-span distance is measured in
    # units that grow with the examples' spread, flattening the gradient.
    spread_sens = pm.HalfNormal("spread_sens", sigma=2.0)
    size_scale = pt.exp(-momentum * ahead) / (1.0 + spread_sens * spread)
    # Log compression (the change): the outside-span gap is measured as
    # |log(t + c) - log(edge + c)|, normalised so the 1-100 line has length ~1;
    # a large offset c recovers the linear gap.
    log_offset = pm.Normal("log_offset", mu=2.0, sigma=1.5)
    c = pt.exp(log_offset)
    log_gap = (outside_span * pt.abs(pt.log(target_value + c) - pt.log(span_edge + c))
               / (pt.log(100.0 + c) - pt.log(1.0 + c)))
    mismatch = pt.set_subtensor(
        mismatch[:, 0, :], (log_gap * size_scale)[:, None] * ex_mask)

    attn = w_base[None, :] * pt.exp(tuning[None, :] * coherence)
    dist = pt.sum(attn[:, :, None] * mismatch, axis=1)
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    familiarity = pt.sum(mem * sim, axis=1)

    logit = (bias + criterion_drift * session_position + gain * familiarity
             + rule_weight * rule_evidence
             + digit_rule_weight * digit_rule_evidence)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(logit), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
