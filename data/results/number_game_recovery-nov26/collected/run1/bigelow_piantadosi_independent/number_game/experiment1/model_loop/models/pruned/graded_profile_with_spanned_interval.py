"""Graded feature profile matching with soft neighbourhoods and the spanned interval.

Refinement of graded_profile_with_neighborhoods: people match the target against
a frequency-weighted, distinctiveness-weighted profile of the examples'
arithmetic and surface properties plus each example's soft numerical
neighbourhood; with two or more examples the profile also contains the interval
the examples span, shared by all of them, weighted by the same rarity rule
(narrow spans count more) and with soft edges of the same fitted width. Targets
between the examples thus gain support even when far from each example.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N = np.arange(1, 101)
K = 4


def _properties():
    props = []
    props.append(N % 2 == 1)
    props.append(N % 2 == 0)
    props.append(np.array([n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1)) for n in N]))
    props.append(np.isin(N, [k * k for k in range(1, 11)]))
    props.append(np.isin(N, [k ** 3 for k in range(1, 5)]))
    for b in range(2, 11):
        props.append(np.isin(N, [b ** e for e in range(1, 7) if b ** e <= 100]))
    for m in range(3, 13):
        props.append(N % m == 0)
    for d in range(10):
        props.append(N % 10 == d)
    for d in range(10):
        props.append(np.array([str(d) in str(n) for n in N]))
    for t in range(10):
        props.append((N // 10) == t)
    return np.array(props, dtype=float)  # (P, 100)


PROPS = _properties()
LOG_SIZE = np.log(PROPS.sum(axis=1))


def prepare_observed(rows):
    n = len(rows)
    log_frac = np.zeros((n, PROPS.shape[0]))
    present = np.zeros_like(log_frac)
    match = np.zeros_like(log_frac)
    ex_dist = np.zeros((n, K, K))      # |e_j - e_k|
    tgt_dist = np.zeros((n, K))        # |t - e_j|
    ex_mask = np.zeros((n, K))
    span_on = np.zeros(n)              # 1 if >= 2 distinct examples
    span_len = np.ones(n)              # max - min + 1
    span_out = np.zeros(n)             # distance of target outside [min, max]
    for i, r in enumerate(rows):
        ex = [int(x) for x in r["number_set"].split(",")][:K]
        t = int(r["target"])
        frac = PROPS[:, [e - 1 for e in ex]].mean(axis=1)
        present[i] = (frac > 0).astype(float)
        log_frac[i] = np.log(np.where(frac > 0, frac, 1.0))
        match[i] = PROPS[:, t - 1]
        m = len(ex)
        ex_mask[i, :m] = 1.0
        e = np.array(ex, dtype=float)
        ex_dist[i, :m, :m] = np.abs(e[:, None] - e[None, :])
        tgt_dist[i, :m] = np.abs(t - e)
        lo, hi = min(ex), max(ex)
        if hi > lo:
            span_on[i] = 1.0
            span_len[i] = hi - lo + 1
            span_out[i] = max(lo - t, t - hi, 0)
    return {
        "log_frac": log_frac,
        "present": present,
        "target_match": match,
        "ex_dist": ex_dist,
        "tgt_dist": tgt_dist,
        "ex_mask": ex_mask,
        "span_on": span_on,
        "span_len": span_len,
        "span_out": span_out,
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    log_frac = pm.Data("log_frac", np.zeros((1, PROPS.shape[0])))
    present = pm.Data("present", np.zeros((1, PROPS.shape[0])))
    target_match = pm.Data("target_match", np.zeros((1, PROPS.shape[0])))
    ex_dist = pm.Data("ex_dist", np.zeros((1, K, K)))
    tgt_dist = pm.Data("tgt_dist", np.zeros((1, K)))
    ex_mask = pm.Data("ex_mask", np.zeros((1, K)))
    span_on = pm.Data("span_on", np.zeros(1))
    span_len = pm.Data("span_len", np.ones(1))
    span_out = pm.Data("span_out", np.zeros(1))

    gamma = pm.HalfNormal("gamma", sigma=1.0)       # distinctiveness weighting
    kappa = pm.HalfNormal("kappa", sigma=3.0)       # steepness of graded sharing
    width = pm.LogNormal("width", mu=np.log(3.0), sigma=0.8)  # neighbourhood / edge width
    intercept = pm.Normal("intercept", mu=-2.0, sigma=2.0)
    slope = pm.HalfNormal("slope", sigma=5.0)

    # Arithmetic / surface properties.
    w = pt.exp(-gamma * pt.as_tensor_variable(LOG_SIZE))
    share = present * pt.exp(kappa * log_frac)
    weighted = share * w
    num = pt.sum(weighted * target_match, axis=1)
    den = pt.sum(weighted, axis=1)

    # Soft neighbourhood of each example as one more graded property.
    q = pt.exp(-1.0 / width)
    edge = q / (1.0 - q + 1e-9)                     # effective size of one soft edge
    nb_size = pt.minimum(1.0 + 2.0 * edge, 100.0)
    nb_w = pt.exp(-gamma * pt.log(nb_size))
    n_ex = pt.sum(ex_mask, axis=1, keepdims=True)
    near = pt.exp(-ex_dist / width) * ex_mask[:, None, :]
    nb_frac = pt.sum(near, axis=2) / n_ex
    nb_share = ex_mask * pt.exp(kappa * pt.log(pt.maximum(nb_frac, 1e-12)))
    nb_match = pt.exp(-tgt_dist / width)
    num = num + nb_w * pt.sum(nb_share * nb_match, axis=1)
    den = den + nb_w * pt.sum(nb_share, axis=1)

    # The one change: the interval spanned by the examples, shared by all of
    # them, weighted by its (soft) size, with soft edges of the same width.
    span_size = pt.minimum(span_len + 2.0 * edge, 100.0)
    span_w = span_on * pt.exp(-gamma * pt.log(span_size))
    span_match = pt.exp(-span_out / width)
    num = num + span_w * span_match
    den = den + span_w + 1e-9
    frac = num / den

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(intercept + slope * frac), 1e-6, 1 - 1e-6))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
