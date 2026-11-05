"""Specificity-weighted (tempered size principle) property overlap.

People weigh each property the examples share by the size principle: a
property that picks out few numbers is a more suspicious coincidence and
counts for more than a broad one, but only partially -- its weight scales as
(property size / 100) ** (-alpha) with an inferred alpha (alpha = 1 would be
the normative single-example likelihood ratio). "Yes" is a sigmoid of the
specificity- and salience-weighted share of the examples' shared property
profile the target has, with the incumbent's properties otherwise unchanged
(family saliences, one-sided graded magnitude bounds, seed multiples,
arithmetic progression, a visible shared final digit overshadowing the rest,
and the last example's lingering final digit). So lone examples with narrow
properties (64) are generalized more narrowly than nondescript ones (37).
"""
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
FAM_IDX = np.array([f for f, _ in _PROPS], dtype="int64")
IS_FD = (FAM_IDX == FD).astype("float64")
LOGSIZE = np.array([np.log(len(p) / 100.0) for _, p in _PROPS])


def _ls(k):
    return float(np.log(max(k, 1) / 100.0))


def prepare_observed(rows):
    T = len(rows)
    sh = np.zeros((T, J))
    hit = np.zeros((T, J))
    keys = ["below_gap", "above_gap", "vis", "seed_shared", "seed_hit",
            "prog_shared", "prog_hit", "last_on", "last_hit",
            "ls_ge", "ls_le", "ls_seed", "ls_prog"]
    o = {k: np.zeros(T) for k in keys}
    for t, r in enumerate(rows):
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        tg = int(r["target"])
        for j, (_f, p) in enumerate(_PROPS):
            if all(e in p for e in ex):
                sh[t, j] = 1.0
                hit[t, j] = 1.0 if tg in p else 0.0
        lo, hi = min(ex), max(ex)
        o["below_gap"][t] = max(0, lo - tg) / 100.0
        o["above_gap"][t] = max(0, tg - hi) / 100.0
        o["ls_ge"][t] = _ls(101 - lo)
        o["ls_le"][t] = _ls(hi)
        shared_digit = len({e % 10 for e in ex}) == 1
        o["vis"][t] = 1.0 if (len(ex) >= 2 and shared_digit) else 0.0
        is_seed = lo >= 2 and all(e % lo == 0 for e in ex)
        o["seed_shared"][t] = 1.0 if is_seed else 0.0
        o["seed_hit"][t] = 1.0 if (is_seed and tg % lo == 0) else 0.0
        o["ls_seed"][t] = _ls(100 // lo) if is_seed else 0.0
        srt = sorted(ex)
        gap = srt[1] - srt[0] if len(srt) >= 2 else 0
        is_prog = gap > 0 and all(srt[i + 1] - srt[i] == gap for i in range(len(srt) - 1))
        o["prog_shared"][t] = 1.0 if is_prog else 0.0
        o["prog_hit"][t] = 1.0 if (is_prog and tg >= lo and (tg - lo) % gap == 0) else 0.0
        o["ls_prog"][t] = _ls(len(range(lo, 101, gap))) if is_prog else 0.0
        linger = len(ex) >= 2 and not shared_digit
        o["last_on"][t] = 1.0 if linger else 0.0
        o["last_hit"][t] = 1.0 if (linger and tg % 10 == ex[-1] % 10) else 0.0
    o["rule_shared"] = sh
    o["rule_hit"] = hit
    o["rating"] = np.asarray([int(r["rating"]) for r in rows], dtype="int64")
    return o


with pm.Model() as model:
    sh = pm.Data("rule_shared", np.zeros((1, J), dtype="float64"))
    hit = pm.Data("rule_hit", np.zeros((1, J), dtype="float64"))
    d = {k: pm.Data(k, np.zeros(1, dtype="float64")) for k in
         ["below_gap", "above_gap", "vis", "seed_shared", "seed_hit",
          "prog_shared", "prog_hit", "last_on", "last_hit",
          "ls_ge", "ls_le", "ls_seed", "ls_prog"]}

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    salience = pm.Dirichlet("salience", a=np.full(N_FAM + 4, 1.5))
    retain = pm.Beta("retain", alpha=2.0, beta=2.0)
    recency = pm.HalfNormal("recency", sigma=1.0)
    band_width = pm.HalfNormal("band_width", sigma=0.5)
    # Tempered size principle: weight of a property ~ (size/100) ** (-alpha).
    alpha = pm.HalfNormal("alpha", sigma=0.5)

    width = 0.02 + band_width
    ge_min = pm.math.exp(-((d["below_gap"] / width) ** 2))
    le_max = pm.math.exp(-((d["above_gap"] / width) ** 2))
    other_w = 1.0 - (1.0 - retain) * d["vis"]

    # Rule properties: salience x specificity, non-digit families overshadowed.
    spec = pt.exp(-alpha * pt.as_tensor_variable(LOGSIZE))            # (J,)
    base = salience[FAM_IDX] * spec                                    # (J,)
    w_rule = base[None, :] * (IS_FD[None, :] + (1.0 - IS_FD[None, :]) * other_w[:, None])
    num = pt.sum(w_rule * hit, axis=1)
    den = pt.sum(w_rule * sh, axis=1)

    w_ge = other_w * salience[N_FAM] * pt.exp(-alpha * d["ls_ge"])
    w_le = other_w * salience[N_FAM + 1] * pt.exp(-alpha * d["ls_le"])
    w_seed = other_w * salience[N_FAM + 2] * pt.exp(-alpha * d["ls_seed"])
    w_prog = other_w * salience[N_FAM + 3] * pt.exp(-alpha * d["ls_prog"])
    w_last = recency * salience[FD] * pt.exp(-alpha * np.log(0.1))
    num = num + w_ge * ge_min + w_le * le_max + w_seed * d["seed_hit"] \
        + w_prog * d["prog_hit"] + w_last * d["last_hit"]
    den = den + w_ge + w_le + w_seed * d["seed_shared"] \
        + w_prog * d["prog_shared"] + w_last * d["last_on"]
    overlap = num / den
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * overlap))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
