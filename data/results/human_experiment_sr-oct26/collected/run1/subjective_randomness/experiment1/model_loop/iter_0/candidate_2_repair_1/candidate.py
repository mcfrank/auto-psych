"""Online transition-learning surprise.

People judge randomness by trying to predict each flip from the one before it
as they read left to right, learning the transition tendencies on the fly from
the flips seen so far (a Bayesian learner with symmetric pseudocount `alpha`
per outcome); a sequence looks random to the extent it stays surprising to this
online learner. Order matters: transitions that become predictable early
(streaks, strict alternation, repeating habits) reduce perceived randomness.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

N_STEPS = 7  # sequences have at most 8 flips -> at most 7 predicted transitions


def _steps(seq):
    """Per predicted flip: (#earlier same-context transitions to this outcome,
    #earlier same-context transitions, valid mask), padded to N_STEPS."""
    seq = seq.strip().upper()
    counts = {}
    match, total, valid = [0.0] * N_STEPS, [0.0] * N_STEPS, [0.0] * N_STEPS
    for t in range(1, min(len(seq), N_STEPS + 1)):
        ctx, nxt = seq[t - 1], seq[t]
        c = counts.setdefault(ctx, {"H": 0, "T": 0})
        match[t - 1] = float(c[nxt])
        total[t - 1] = float(c["H"] + c["T"])
        valid[t - 1] = 1.0
        c[nxt] += 1
    return match, total, valid


def compute_features(sequence_a, sequence_b):
    out = {}
    for tag, seq in (("a", sequence_a), ("b", sequence_b)):
        m, n, v = _steps(seq)
        for t in range(N_STEPS):
            out[f"m{t}_{tag}"] = m[t]
            out[f"n{t}_{tag}"] = n[t]
            out[f"v{t}_{tag}"] = v[t]
    return out


with pm.Model() as model:
    data = {}
    for tag in ("a", "b"):
        for kind in ("m", "n", "v"):
            cols = [
                pm.Data(f"{kind}{t}_{tag}", np.zeros(1, dtype="float64"))
                for t in range(N_STEPS)
            ]
            data[(kind, tag)] = pt.stack(cols, axis=1)  # (trials, steps)

    # Prior strength of the online learner (pseudocount per outcome).
    alpha = pm.LogNormal("alpha", mu=0.0, sigma=1.0)
    # Sensitivity of choice to the difference in total surprise.
    beta = pm.HalfNormal("beta", sigma=2.0)

    def surprise(tag):
        m, n, v = data[("m", tag)], data[("n", tag)], data[("v", tag)]
        logp = pt.log(m + alpha) - pt.log(n + 2.0 * alpha)
        return -pt.sum(v * logp, axis=1)

    s_a = surprise("a")
    s_b = surprise("b")
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * (s_a - s_b)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
