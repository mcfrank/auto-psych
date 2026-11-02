"""Stepwise progression induction.

People assume the hidden program is a counting procedure: it starts at some
number and steps upward by a fixed stride until it stops (e.g. "from 22 to 82
in steps of 4", or "every number from 40 to 60" for stride 1). They infer which
start, stride and stop produced the examples -- preferring small strides and,
by the size principle, short progressions -- and say yes to a target if the
inferred progressions land on it.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N_MAX = 100
D_MAX = 99


def _progression_weights(examples):
    """Per stride d: summed size-principle likelihood of every consistent
    progression (total, and per target 1..100 restricted to those containing it)."""
    xs = np.array(examples, dtype=float)
    n = len(xs)
    lo_x, hi_x = int(min(examples)), int(max(examples))
    log_w = np.full((D_MAX, N_MAX + 1), -np.inf)  # column 0 = total
    for d in range(1, D_MAX + 1):
        r = examples[0] % d
        if any(x % d != r for x in examples):
            continue
        if d > 1 and lo_x == hi_x:
            continue  # a single-number program is counted once, under stride 1
        los = np.arange(lo_x, 0, -d)
        his = np.arange(hi_x, N_MAX + 1, d)
        tot = 0.0
        per_t = np.zeros(N_MAX + 1)
        scale = None
        for lo in los:
            for hi in his:
                k = (hi - lo) // d + 1
                lw = -n * np.log(k)
                if scale is None:
                    scale = lw
                w = np.exp(lw - scale)
                tot += w
                per_t[lo:hi + 1:d] += w
        log_w[d - 1, 0] = np.log(tot) + scale
        with np.errstate(divide="ignore"):
            log_w[d - 1, 1:] = np.log(per_t[1:]) + scale
    return log_w


def prepare_observed(rows):
    cache = {}
    n_trials = len(rows)
    ws = np.zeros((n_trials, D_MAX))
    wt = np.zeros((n_trials, D_MAX))
    for i, row in enumerate(rows):
        s = row["number_set"]
        if s not in cache:
            ex = [int(x) for x in str(s).split(",")]
            lw = _progression_weights(ex)
            cache[s] = lw - np.max(lw[:, 0])
        lw = cache[s]
        t = int(row["target"])
        ws[i] = np.exp(lw[:, 0])
        wt[i] = np.exp(lw[:, t])
    return {
        "stride_weight_all": ws,
        "stride_weight_target": wt,
        "stride_index": np.arange(D_MAX, dtype="float64"),
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    w_all = pm.Data("stride_weight_all", np.ones((1, D_MAX)))
    w_tgt = pm.Data("stride_weight_target", np.ones((1, D_MAX)))
    d_idx = pm.Data("stride_index", np.arange(D_MAX, dtype="float64"))

    # How quickly larger strides become less natural (geometric prior on stride).
    stride_decay = pm.HalfNormal("stride_decay", sigma=1.0)
    # Commitment: sharpening of the progression posterior into a yes/no answer.
    sharpness = pm.LogNormal("sharpness", mu=0.0, sigma=0.7)
    # Lapse to a default yes-rate.
    lapse = pm.Beta("lapse", alpha=1.5, beta=8.0)
    guess = pm.Beta("guess", alpha=2.0, beta=2.0)

    prior = pt.exp(-stride_decay * d_idx)
    num = pt.sum(w_tgt * prior[None, :], axis=1)
    den = pt.sum(w_all * prior[None, :], axis=1)
    post = pt.clip(num / den, 1e-6, 1 - 1e-6)
    logit_post = pt.log(post) - pt.log1p(-post)
    sharp = pm.math.sigmoid(sharpness * logit_post)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(lapse * guess + (1 - lapse) * sharp, 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
