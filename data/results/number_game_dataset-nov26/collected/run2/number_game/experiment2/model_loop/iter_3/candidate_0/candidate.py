"""Specificity (suspicious-coincidence) weighted property overlap.

People weigh each property the examples share by how much of a coincidence
that sharing would be: a property that covers few numbers carries more weight
than a broad one, and this tilt toward the rarer properties grows with the
number of examples that share them (weight multiplied by
exp(gamma * n_examples * specificity), specificity = log(100/size)/log(100)).
Otherwise "yes" is a sigmoid of the salience-weighted share of the examples'
shared profile the target has, as in progression_graded_magnitude_band
(family saliences, seed multiples, arithmetic progression, graded one-sided
magnitude bands, final-digit overshadowing, the last example's lingering digit).
"""
import math
import numpy as np
import pymc as pm
import pytensor.tensor as pt

_N = range(1, 101)
N_FAM = 8
FD = 5


def _props():
    props = []
    props.append((0, {n for n in _N if n % 2 == 1}))
    props.append((0, {n for n in _N if n % 2 == 0}))
    props.append((1, {n for n in _N if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))}))
    props.append((2, {k * k for k in range(1, 11)}))
    props.append((2, {k ** 3 for k in range(1, 5)}))
    for b in range(2, 11):
        props.append((3, {b ** k for k in range(1, 8) if b ** k <= 100}))
    for m in range(3, 13):
        props.append((7 if m in (5, 10) else 4, {n for n in _N if n % m == 0}))
    for d in range(10):
        props.append((5, {n for n in _N if n % 10 == d}))
    for d in range(10):
        props.append((6, {n for n in _N if str(d) in str(n)}))
    return props


_PROPS = _props()
J = len(_PROPS)
FAM_ONEHOT = np.zeros((J, N_FAM))
SPEC = np.zeros(J)
IS_FD = np.zeros(J)
for _j, (_f, _p) in enumerate(_PROPS):
    FAM_ONEHOT[_j, _f] = 1.0
    SPEC[_j] = math.log(100.0 / len(_p)) / math.log(100.0)
    IS_FD[_j] = 1.0 if _f == FD else 0.0


def _spec(size):
    return math.log(100.0 / max(size, 1)) / math.log(100.0)


def prepare_observed(rows):
    T = len(rows)
    shared = np.zeros((T, J))
    hit = np.zeros((T, J))
    f = {k: np.zeros(T) for k in ["n_ex", "below_gap", "above_gap", "vis", "seed_shared",
                                  "seed_hit", "seed_spec", "prog_shared", "prog_hit",
                                  "prog_spec", "last_on", "last_hit"]}
    for t, r in enumerate(rows):
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        tg = int(r["target"])
        for j, (_fam, p) in enumerate(_PROPS):
            if all(e in p for e in ex):
                shared[t, j] = 1.0
                hit[t, j] = 1.0 if tg in p else 0.0
        f["n_ex"][t] = len(ex)
        f["below_gap"][t] = max(0, min(ex) - tg) / 100.0
        f["above_gap"][t] = max(0, tg - max(ex)) / 100.0
        sd = len({e % 10 for e in ex}) == 1
        f["vis"][t] = 1.0 if (len(ex) >= 2 and sd) else 0.0
        seed = min(ex)
        is_seed = seed >= 2 and all(e % seed == 0 for e in ex)
        if is_seed:
            f["seed_shared"][t] = 1.0
            f["seed_hit"][t] = 1.0 if tg % seed == 0 else 0.0
            f["seed_spec"][t] = _spec(100 // seed)
        srt = sorted(ex)
        gap = srt[1] - srt[0] if len(srt) >= 2 else 0
        if gap > 0 and all(srt[i + 1] - srt[i] == gap for i in range(len(srt) - 1)):
            f["prog_shared"][t] = 1.0
            f["prog_hit"][t] = 1.0 if (tg >= srt[0] and (tg - srt[0]) % gap == 0) else 0.0
            f["prog_spec"][t] = _spec((100 - srt[0]) // gap + 1)
        if len(ex) >= 2 and not sd:
            f["last_on"][t] = 1.0
            f["last_hit"][t] = 1.0 if tg % 10 == ex[-1] % 10 else 0.0
    out = {"shared": shared, "target_hit": hit}
    out.update(f)
    out["rating"] = np.asarray([int(r["rating"]) for r in rows], dtype="int64")
    return out


with pm.Model() as model:
    shared = pm.Data("shared", np.zeros((1, J)))
    hit = pm.Data("target_hit", np.zeros((1, J)))
    n_ex = pm.Data("n_ex", np.zeros(1))
    below = pm.Data("below_gap", np.zeros(1))
    above = pm.Data("above_gap", np.zeros(1))
    vis = pm.Data("vis", np.zeros(1))
    seed_shared = pm.Data("seed_shared", np.zeros(1))
    seed_hit = pm.Data("seed_hit", np.zeros(1))
    seed_spec = pm.Data("seed_spec", np.zeros(1))
    prog_shared = pm.Data("prog_shared", np.zeros(1))
    prog_hit = pm.Data("prog_hit", np.zeros(1))
    prog_spec = pm.Data("prog_spec", np.zeros(1))
    last_on = pm.Data("last_on", np.zeros(1))
    last_hit = pm.Data("last_hit", np.zeros(1))

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    salience = pm.Dirichlet("salience", a=np.full(N_FAM + 4, 1.5))
    retain = pm.Beta("retain", alpha=2.0, beta=2.0)
    recency = pm.HalfNormal("recency", sigma=1.0)
    band_width = pm.HalfNormal("band_width", sigma=0.5)
    # Coincidence sensitivity: how much rarer properties gain per example.
    gamma = pm.HalfNormal("gamma", sigma=0.5)

    width = 0.02 + band_width
    ge_min = pt.exp(-((below / width) ** 2))
    le_max = pt.exp(-((above / width) ** 2))
    other_w = 1.0 - (1.0 - retain) * vis                       # (T,)

    fam_sal = pt.dot(FAM_ONEHOT, salience[:N_FAM])            # (J,)
    boost = pt.exp(gamma * n_ex[:, None] * SPEC[None, :])     # (T, J)
    ow = IS_FD[None, :] + (1.0 - IS_FD[None, :]) * other_w[:, None]
    w = shared * fam_sal[None, :] * ow * boost
    num = pt.sum(w * hit, axis=1)
    den = pt.sum(w, axis=1)

    num = num + other_w * (salience[N_FAM] * ge_min + salience[N_FAM + 1] * le_max)
    den = den + other_w * (salience[N_FAM] + salience[N_FAM + 1])
    w_seed = other_w * salience[N_FAM + 2] * pt.exp(gamma * n_ex * seed_spec)
    num = num + w_seed * seed_hit
    den = den + w_seed * seed_shared
    w_prog = other_w * salience[N_FAM + 3] * pt.exp(gamma * n_ex * prog_spec)
    num = num + w_prog * prog_hit
    den = den + w_prog * prog_shared
    w_last = recency * salience[FD]
    num = num + w_last * last_hit
    den = den + w_last * last_on

    overlap = num / den
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * overlap))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
