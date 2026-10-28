"""Dual-route exemplar model whose rule route ranges over step progressions.

Refinement of dual_route_factor_and_digit_rule_exemplar: people judge a target
by its recency-weighted exponential similarity to the remembered examples
(size, final digit, parity, graded shared factor; attention stretched by how
much the examples resemble one another) and consult an abstracted rule at the
decision as separate evidence. The single change is the space of rules: every
progression "numbers congruent to the examples modulo s", for each step s >= 2
that divides all differences between examples. Each step has prior weight
s**(-complexity), times a bonus for the decimal step 10; its likelihood is the
coincidence improbability s**(k-1). The rule evidence for a target is the
posterior-weighted signed coincidence evidence (k-1)*log(s) of the steps (+ if
the target lies on the progression, - otherwise). "Ends in d" and "multiples
of g" are special cases (s = 10; remainder 0).
"""
import math
from functools import reduce

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4
NDIM = 4  # magnitude, final digit, parity, graded shared factor
NS = 12   # max number of candidate steps (divisors of a difference <= 99)


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
    log_step = np.zeros((n, NS))
    on_step = np.zeros((n, NS))
    step_mask = np.zeros((n, NS))
    is_decimal = np.zeros((n, NS))
    n_more = np.zeros(n)
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
            diffs = [abs(x - ex[0]) for x in ex[1:]]
            dgcd = reduce(math.gcd, diffs) if diffs else 0
            steps = [st for st in range(2, 100) if dgcd > 0 and dgcd % st == 0]
            cache[s] = (c, steps[:NS])
        c, steps = cache[s]
        coh[r] = c
        n_more[r] = k - 1
        for j, st in enumerate(steps):
            log_step[r, j] = math.log(st)
            on_step[r, j] = 1.0 if (t - ex[0]) % st == 0 else -1.0
            step_mask[r, j] = 1.0
            is_decimal[r, j] = 1.0 if st == 10 else 0.0
        for i, x in enumerate(ex):
            mism[r, 0, i] = abs(t - x) / 100.0
            mism[r, 1, i] = 0.0 if t % 10 == x % 10 else 1.0
            mism[r, 2, i] = 0.0 if t % 2 == x % 2 else 1.0
            mism[r, 3, i] = _factor_mismatch(t, x)
            lag[r, i] = k - 1 - i
            mask[r, i] = 1.0
        rating[r] = int(float(row["rating"]))
    return {"mismatch": mism, "coherence": coh, "lag": lag, "ex_mask": mask,
            "log_step": log_step, "on_step": on_step, "step_mask": step_mask,
            "is_decimal": is_decimal, "n_more": n_more, "rating": rating}


with pm.Model() as model:
    mismatch = pm.Data("mismatch", np.zeros((1, NDIM, MAXK)))
    coherence = pm.Data("coherence", np.zeros((1, NDIM)))
    lag = pm.Data("lag", np.zeros((1, MAXK)))
    ex_mask = pm.Data("ex_mask", np.ones((1, MAXK)))
    log_step = pm.Data("log_step", np.zeros((1, NS)))
    on_step = pm.Data("on_step", np.zeros((1, NS)))
    step_mask = pm.Data("step_mask", np.zeros((1, NS)))
    is_decimal = pm.Data("is_decimal", np.zeros((1, NS)))
    n_more = pm.Data("n_more", np.zeros(1))

    w_base = pm.HalfNormal("w_base", sigma=np.array([10.0, 2.0, 2.0, 2.0]))
    tuning = pm.HalfNormal("tuning", sigma=2.0, shape=NDIM)
    retention = pm.Beta("retention", alpha=2.0, beta=2.0)
    bias = pm.Normal("bias", mu=-1.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    # Rule route over step progressions (the change).
    rule_weight = pm.HalfNormal("rule_weight", sigma=1.0)
    step_complexity = pm.HalfNormal("step_complexity", sigma=2.0)
    decimal_bonus = pm.Normal("decimal_bonus", mu=0.0, sigma=2.0)

    attn = w_base[None, :] * pt.exp(tuning[None, :] * coherence)
    dist = pt.sum(attn[:, :, None] * mismatch, axis=1)
    sim = pt.exp(-dist)
    mem = ex_mask * pt.exp(lag * pt.log(retention))
    mem = mem / pt.sum(mem, axis=1, keepdims=True)
    familiarity = pt.sum(mem * sim, axis=1)

    # Posterior over consistent steps: prior s^-complexity (decimal bonus),
    # likelihood s^(k-1) from the coincidence of the examples lining up.
    coinc = n_more[:, None] * log_step
    logw = coinc - step_complexity * log_step + decimal_bonus * is_decimal
    logw = pt.switch(step_mask > 0, logw, -1e6)
    logw = logw - pt.max(logw, axis=1, keepdims=True)
    w = pt.exp(logw) * step_mask
    w = w / (pt.sum(w, axis=1, keepdims=True) + 1e-12)
    rule_evidence = pt.sum(w * coinc * on_step, axis=1)

    logit = bias + gain * familiarity + rule_weight * rule_evidence
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(logit), 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
