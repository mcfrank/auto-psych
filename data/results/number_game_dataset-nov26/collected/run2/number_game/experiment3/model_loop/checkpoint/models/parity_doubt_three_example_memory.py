"""Parity-bounded ambiguity doubt with a three-example working-memory limit.

Refinement of parity_bounded_ambiguity_doubt by one grafted component, taken
from working_memory_three_example_limit: only three examples fit in working
memory. When four are shown, with inferred probability `drop` one example has
dropped out, and the whole judgment (salience-weighted share of the shared
profile, rule-ambiguity doubt and its parity-kept flat "yes") is made from the
three remembered examples, averaged over which was lost. Sets of one to three
examples are judged exactly as by the incumbent.
"""
import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NSUB = 5  # full set, then the four 3-example subsets (copies for n < 4)

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
        fam = 7 if m in (5, 10) else 4
        props.append((fam, {n for n in _N if n % m == 0}))
    for d in range(10):
        props.append((5, {n for n in _N if n % 10 == d}))
    for d in range(10):
        props.append((6, {n for n in _N if str(d) in str(n)}))
    return props


_PROPS = _props()


def _feat(ex, t):
    out = {}
    for f in range(N_FAM):
        shared = [p for fam, p in _PROPS if fam == f and all(e in p for e in ex)]
        out[f"sh_{f}"] = float(len(shared))
        out[f"hit_{f}"] = float(sum(1 for p in shared if t in p))
    span_on = len(ex) >= 2
    out["single_example"] = 0.0 if span_on else 1.0
    # Number of candidate rules (property categories) every example fits,
    # centred at 6; for a lone example this is its own category count.
    out["rule_count"] = float(sum(1 for fam, p in _PROPS if all(e in p for e in ex))) - 6.0
    out["span_active"] = 1.0 if span_on else 0.0
    # Doubt-mode parity constraint: examples all share a parity, target breaks it.
    pars = {e % 2 for e in ex}
    out["shared_parity_break"] = 1.0 if (len(pars) == 1 and t % 2 not in pars) else 0.0
    out["span_gap"] = (max(0, min(ex) - t, t - max(ex)) / 100.0) if span_on else 0.0
    out["below_gap"] = max(0, min(ex) - t) / 100.0
    out["above_gap"] = max(0, t - max(ex)) / 100.0
    shared_digit = len({e % 10 for e in ex}) == 1
    out["visible_digit_pattern"] = 1.0 if (len(ex) >= 2 and shared_digit) else 0.0
    seed = min(ex)
    is_seed = seed >= 2 and all(e % seed == 0 for e in ex)
    out["seed_shared"] = 1.0 if is_seed else 0.0
    out["seed_hit"] = 1.0 if (is_seed and t % seed == 0) else 0.0
    srt = sorted(ex)
    gap = srt[1] - srt[0] if len(srt) >= 2 else 0
    is_prog = gap > 0 and all(srt[i + 1] - srt[i] == gap for i in range(len(srt) - 1))
    out["progression_shared"] = 1.0 if is_prog else 0.0
    out["progression_hit"] = 1.0 if (is_prog and t >= srt[0] and (t - srt[0]) % gap == 0) else 0.0
    linger = len(ex) >= 2 and not shared_digit
    out["last_digit_active"] = 1.0 if linger else 0.0
    out["last_digit_match"] = 1.0 if (linger and t % 10 == ex[-1] % 10) else 0.0
    out["first_digit_match"] = 1.0 if (linger and t % 10 == ex[0] % 10) else 0.0
    return out


KEYS = ["single_example", "rule_count", "span_active", "shared_parity_break", "span_gap",
        "below_gap", "above_gap", "visible_digit_pattern", "seed_shared", "seed_hit",
        "progression_shared", "progression_hit", "last_digit_active", "last_digit_match",
        "first_digit_match"] + [f"sh_{f}" for f in range(N_FAM)] + [f"hit_{f}" for f in range(N_FAM)]


def prepare_observed(rows):
    n = len(rows)
    out = {k: np.zeros((n, NSUB)) for k in KEYS}
    out["over_capacity"] = np.zeros(n)
    rating = np.zeros(n, dtype="int64")
    cache = {}
    for i, r in enumerate(rows):
        key = (str(r["number_set"]), str(r["target"]))
        if key not in cache:
            ex = [int(x) for x in key[0].split(",")]
            t = int(key[1])
            if len(ex) >= 4:
                subs = [ex] + [list(c) for c in itertools.combinations(ex, 3)]
            else:
                subs = [ex] * NSUB
            fs = [_feat(s, t) for s in subs]
            cache[key] = (np.array([[f[k] for f in fs] for k in KEYS]), 1.0 if len(ex) >= 4 else 0.0)
        mat, oc = cache[key]
        for j, k in enumerate(KEYS):
            out[k][i] = mat[j]
        out["over_capacity"][i] = oc
        rating[i] = int(float(r["rating"]))
    out["rating"] = rating
    return out


with pm.Model() as model:
    D = {k: pm.Data(k, np.zeros((1, NSUB))) for k in KEYS}
    oc = pm.Data("over_capacity", np.zeros(1))
    sh = [D[f"sh_{f}"] for f in range(N_FAM)]
    hit = [D[f"hit_{f}"] for f in range(N_FAM)]

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    salience = pm.Dirichlet("salience", a=np.full(N_FAM + 5, 1.5))
    retain = pm.Beta("retain", alpha=2.0, beta=2.0)
    recency = pm.HalfNormal("recency", sigma=1.0)
    primacy = pm.HalfNormal("primacy", sigma=1.0)
    band_width = pm.LogNormal("band_width", mu=np.log(0.25), sigma=0.5)
    span_w = pm.LogNormal("span_width", mu=np.log(0.05), sigma=0.5)
    width = 0.02 + band_width
    ge_min = pt.exp(-((D["below_gap"] / width) ** 2))
    le_max = pt.exp(-((D["above_gap"] / width) ** 2))

    other_w = 1.0 - (1.0 - retain) * D["visible_digit_pattern"]
    num = other_w * (salience[N_FAM] * ge_min + salience[N_FAM + 1] * le_max
                     + salience[N_FAM + 2] * D["seed_hit"])
    den = other_w * (salience[N_FAM] + salience[N_FAM + 1] + salience[N_FAM + 2] * D["seed_shared"])
    for f in range(N_FAM):
        w = salience[f] if f == FD else salience[f] * other_w
        num = num + w * hit[f]
        den = den + w * sh[f]
    w_last = recency * salience[FD]
    num = num + w_last * D["last_digit_match"]
    den = den + w_last * D["last_digit_active"]
    w_prog = other_w * salience[N_FAM + 3]
    num = num + w_prog * D["progression_hit"]
    den = den + w_prog * D["progression_shared"]
    w_first = primacy * salience[FD]
    num = num + w_first * D["first_digit_match"]
    den = den + w_first * D["last_digit_active"]
    span_sim = D["span_active"] * pt.exp(-(D["span_gap"] / (0.01 + span_w)))
    w_span = other_w * salience[N_FAM + 4]
    num = num + w_span * span_sim
    den = den + w_span * D["span_active"]
    p_judge = pm.math.sigmoid(intercept + slope * num / den)

    d0 = pm.Normal("doubt0", 0.0, 1.5)
    db = pm.HalfNormal("doubt_rich", 1.0)
    dm = pm.Normal("doubt_multi", -2.0, 1.5)
    dbm = pm.HalfNormal("doubt_rich_multi", 1.0)
    ry = pm.Beta("random_yes", 2.0, 2.0)
    parity_keep = pm.Beta("parity_keep", alpha=2.0, beta=2.0)
    single = D["single_example"]
    rich = D["rule_count"]
    doubt_t = pm.math.sigmoid(single * (d0 + db * rich) + (1.0 - single) * (dm + dbm * rich))
    p_doubt = ry * (1.0 - parity_keep * D["shared_parity_break"])
    p_ver = doubt_t * p_doubt + (1.0 - doubt_t) * p_judge  # (trials, NSUB)

    # The one change: with four examples, one drops out of memory with
    # probability `drop`; the judgment then averages over which was lost.
    drop = pm.Beta("drop", 2.0, 2.0)
    p_mix = (1.0 - drop * oc) * p_ver[:, 0] + drop * oc * pt.mean(p_ver[:, 1:], axis=1)
    p_yes = pm.Deterministic("p_yes", pt.clip(p_mix, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
