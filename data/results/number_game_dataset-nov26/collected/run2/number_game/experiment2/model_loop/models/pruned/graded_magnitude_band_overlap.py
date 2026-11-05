"""Graded magnitude-band similarity in the seed/overshadowing/recency overlap model.

Refinement of seed_overshadowing_last_digit_recency by one change: the two
one-sided magnitude bounds ("at least the smallest example", "at most the
largest example") are no longer all-or-none tests. They are graded magnitude
similarities that fall off with a Gaussian gradient (inferred width, shared by
both sides) in the target's distance below the smallest or above the largest
example. Everything else is unchanged: "yes" is a sigmoid of the
salience-weighted share of the examples' shared property profile the target
has, with base-ten multiples as their own family, multiples of the seed when
the smallest example divides all others, a visible shared final digit
overshadowing other properties, and the last example's final digit lingering.
"""
import numpy as np
import pymc as pm

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


def compute_features(number_set, target):
    ex = [int(x) for x in str(number_set).split(",")]
    t = int(target)
    out = {}
    for f in range(N_FAM):
        shared = [p for fam, p in _PROPS if fam == f and all(e in p for e in ex)]
        out[f"sh_{f}"] = float(len(shared))
        out[f"hit_{f}"] = float(sum(1 for p in shared if t in p))
    # Distance (on the 1-100 scale, /100) below the smallest / above the largest example.
    out["below_gap"] = max(0, min(ex) - t) / 100.0
    out["above_gap"] = max(0, t - max(ex)) / 100.0
    shared_digit = len({e % 10 for e in ex}) == 1
    out["visible_digit_pattern"] = 1.0 if (len(ex) >= 2 and shared_digit) else 0.0
    seed = min(ex)
    is_seed = seed >= 2 and all(e % seed == 0 for e in ex)
    out["seed_shared"] = 1.0 if is_seed else 0.0
    out["seed_hit"] = 1.0 if (is_seed and t % seed == 0) else 0.0
    linger = len(ex) >= 2 and not shared_digit
    out["last_digit_active"] = 1.0 if linger else 0.0
    out["last_digit_match"] = 1.0 if (linger and t % 10 == ex[-1] % 10) else 0.0
    return out


with pm.Model() as model:
    sh = [pm.Data(f"sh_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    hit = [pm.Data(f"hit_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    below = pm.Data("below_gap", np.zeros(1, dtype="float64"))
    above = pm.Data("above_gap", np.zeros(1, dtype="float64"))
    vis = pm.Data("visible_digit_pattern", np.zeros(1, dtype="float64"))
    seed_shared = pm.Data("seed_shared", np.zeros(1, dtype="float64"))
    seed_hit = pm.Data("seed_hit", np.zeros(1, dtype="float64"))
    last_on = pm.Data("last_digit_active", np.zeros(1, dtype="float64"))
    last_hit = pm.Data("last_digit_match", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    salience = pm.Dirichlet("salience", a=np.full(N_FAM + 3, 1.5))
    retain = pm.Beta("retain", alpha=2.0, beta=2.0)
    recency = pm.HalfNormal("recency", sigma=1.0)
    # Width of the magnitude-similarity gradient (fraction of the 1-100 scale).
    band_width = pm.HalfNormal("band_width", sigma=0.5)

    width = 0.02 + band_width
    ge_min = pm.math.exp(-((below / width) ** 2))
    le_max = pm.math.exp(-((above / width) ** 2))

    other_w = 1.0 - (1.0 - retain) * vis
    num = other_w * (salience[N_FAM] * ge_min + salience[N_FAM + 1] * le_max
                     + salience[N_FAM + 2] * seed_hit)
    den = other_w * (salience[N_FAM] + salience[N_FAM + 1] + salience[N_FAM + 2] * seed_shared)
    for f in range(N_FAM):
        w = salience[f] if f == FD else salience[f] * other_w
        num = num + w * hit[f]
        den = den + w * sh[f]
    w_last = recency * salience[FD]
    num = num + w_last * last_hit
    den = den + w_last * last_on
    overlap = num / den
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * overlap))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
