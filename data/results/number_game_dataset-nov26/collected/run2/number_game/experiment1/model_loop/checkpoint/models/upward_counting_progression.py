"""Upward counting progression.

People think of the hidden program as a counter: it starts at some number and
keeps adding a fixed step, producing numbers upward until it runs out of room
at 100. They infer step and starting point from the examples (preferring
starting points at or just below the smallest example, and favouring counters
that make the examples likely, so more examples narrow the guess), and say
"yes" with their belief that such a counter reaches the target. Targets above
the largest example on the counter's path are readily accepted; targets below
the smallest example only if the counter is thought to start that low.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _hypotheses(examples):
    """All counters (start a, step d) whose upward path 1..100 hits every example."""
    ex = sorted(set(examples))
    m = ex[0]
    g = reduce(math.gcd, [x - m for x in ex[1:]], 0)
    hyps = []
    for d in range(1, 100):
        if g and g % d:
            continue
        r = m % d or d
        a = r
        while a <= m:
            if a + d <= 100:  # a counter produces at least two numbers
                size = (100 - a) // d + 1
                hyps.append((a, d, (a - r) // d, size))
            a += d
    return hyps


def prepare_observed(rows):
    sets = sorted(set(r["number_set"] for r in rows))
    sidx = {s: i for i, s in enumerate(sets)}
    info = {}
    for s in sets:
        ex = [int(x) for x in s.split(",")]
        info[s] = (len(ex), _hypotheses(ex))
    H = max(len(h) for _, h in info.values())
    S, N = len(sets), len(rows)
    logsize = np.zeros((S, H))
    logd = np.zeros((S, H))
    depth = np.zeros((S, H))
    mask = np.zeros((S, H))
    for s, (n, hy) in info.items():
        i = sidx[s]
        for k, (a, d, j, size) in enumerate(hy):
            logsize[i, k] = n * math.log(size)
            logd[i, k] = math.log(d)
            depth[i, k] = j
            mask[i, k] = 1.0
    member = np.zeros((N, H))
    onehot = np.zeros((N, S))
    rating = np.zeros(N, dtype="int64")
    for t, r in enumerate(rows):
        s = r["number_set"]
        tg = int(r["target"])
        onehot[t, sidx[s]] = 1.0
        for k, (a, d, j, size) in enumerate(info[s][1]):
            if tg >= a and (tg - a) % d == 0:
                member[t, k] = 1.0
        rating[t] = int(float(r["rating"]))
    return {"cnt_logsize": logsize, "cnt_logstep": logd, "cnt_depth": depth,
            "cnt_mask": mask, "cnt_member": member, "cnt_set_onehot": onehot,
            "rating": rating}


with pm.Model() as model:
    logsize = pm.Data("cnt_logsize", np.zeros((1, 1)))
    logstep = pm.Data("cnt_logstep", np.zeros((1, 1)))
    depth = pm.Data("cnt_depth", np.zeros((1, 1)))
    mask = pm.Data("cnt_mask", np.ones((1, 1)))
    member = pm.Data("cnt_member", np.zeros((1, 1)))
    onehot = pm.Data("cnt_set_onehot", np.ones((1, 1)))

    # strength of the size principle (evidence weight)
    beta = pm.HalfNormal("beta", sigma=1.5)
    # preference for small (gamma > 0) or large steps
    gamma = pm.Normal("gamma", mu=0.0, sigma=1.0)
    # how fast belief falls off for counters starting further below the smallest example
    kappa = pm.HalfNormal("kappa", sigma=2.0)
    eps = pm.Beta("eps", alpha=2.0, beta=5.0)
    bias = pm.Beta("bias", alpha=2.0, beta=2.0)

    logw = -beta * logsize - gamma * logstep - kappa * depth
    logw = pt.switch(mask > 0, logw, -1e10)
    logw = logw - pt.max(logw, axis=1, keepdims=True)
    w = pt.exp(logw)
    post = w / pt.sum(w, axis=1, keepdims=True)
    p_member = pt.sum(pt.dot(onehot, post) * member, axis=1)
    p = (1.0 - eps) * p_member + eps * bias
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
