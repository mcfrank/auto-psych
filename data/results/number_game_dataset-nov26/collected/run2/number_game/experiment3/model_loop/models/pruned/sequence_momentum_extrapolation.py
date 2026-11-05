"""Sequence-momentum extrapolation of magnitude.

People read the examples in the order shown as the program's running output,
and the direction it was travelling (the last example larger or smaller than
the first) tells them where it is heading next. So magnitude generalization
reaches further beyond the examples on the side the sequence was moving
toward and less far on the side it came from (an inferred momentum that
stretches the magnitude band ahead and shrinks it behind); a lone example has
no direction. Everything else is the current best account
(parity_bounded_ambiguity_doubt) unchanged: the salience-weighted share of
the examples' shared profile with the in-span interval similarity, and the
parity-keeping, rule-ambiguity-driven flat-rate doubt.
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
    span_on = len(ex) >= 2
    out["single_example"] = 0.0 if span_on else 1.0
    # Number of candidate rules (property categories) every example fits,
    # centred at 6; for a lone example this is its own category count.
    out["rule_count"] = float(sum(1 for fam, p in _PROPS if all(e in p for e in ex))) - 6.0
    out["span_active"] = 1.0 if span_on else 0.0
    # Doubt-mode parity constraint: examples all share a parity, target breaks it.
    pars = {e % 2 for e in ex}
    out["shared_parity_break"] = 1.0 if (len(pars) == 1 and t % 2 not in pars) else 0.0
    # Direction of travel through the examples as read: +1 if the last shown
    # example is larger than the first, -1 if smaller, 0 for a lone example.
    trend = (ex[-1] > ex[0]) - (ex[-1] < ex[0]) if span_on else 0
    out["travel_direction"] = float(trend)
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
    prog_shared = pm.Data("progression_shared", np.zeros(1, dtype="float64"))
    prog_hit = pm.Data("progression_hit", np.zeros(1, dtype="float64"))
    span_on = pm.Data("span_active", np.zeros(1, dtype="float64"))
    first_hit = pm.Data("first_digit_match", np.zeros(1, dtype="float64"))
    single = pm.Data("single_example", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    salience = pm.Dirichlet("salience", a=np.full(N_FAM + 5, 1.5))
    retain = pm.Beta("retain", alpha=2.0, beta=2.0)
    recency = pm.HalfNormal("recency", sigma=1.0)
    primacy = pm.HalfNormal("primacy", sigma=1.0)
    band_width = pm.LogNormal("band_width", mu=np.log(0.25), sigma=0.5)
    width = 0.02 + band_width
    # Directional extrapolation: magnitude generalization reaches further on
    # the side the sequence was heading (log-width shifted by +momentum) and
    # less far behind it (-momentum); no shift for a lone example.
    travel = pm.Data("travel_direction", np.zeros(1, dtype="float64"))
    momentum = pm.Normal("momentum", mu=0.0, sigma=0.5)
    stretch_up = pm.math.exp(momentum * travel)
    stretch_down = pm.math.exp(-momentum * travel)
    ge_min = pm.math.exp(-((below / (width * stretch_down)) ** 2))
    le_max = pm.math.exp(-((above / (width * stretch_up)) ** 2))

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
    # Arithmetic-progression concept: its own salience, overshadowed like
    # the other non-digit properties by a visible shared final digit.
    w_prog = other_w * salience[N_FAM + 3]
    num = num + w_prog * prog_hit
    den = den + w_prog * prog_shared
    # Grafted component: the first example's final digit lingers too.
    w_first = primacy * salience[FD]
    num = num + w_first * first_hit
    den = den + w_first * last_on
    span_w = pm.LogNormal("span_width", mu=np.log(0.05), sigma=0.5)
    sw = 0.01 + span_w
    span_sim = span_on * pm.math.exp(-(below + above) / sw)  # symmetric, as in the incumbent
    w_span = other_w * salience[N_FAM + 4]
    num = num + w_span * span_sim
    den = den + w_span * span_on
    overlap = num / den
    p_judge = pm.math.sigmoid(intercept + slope * overlap)
    # Rule-ambiguity doubt: grows with the number of rules all examples fit.
    # Lone examples as in the incumbent; the change is that it also operates
    # on multi-example sets, with its own base rate and slope.
    rich = pm.Data("rule_count", np.zeros(1, dtype="float64"))
    d0 = pm.Normal("doubt0", 0.0, 1.5)
    db = pm.HalfNormal("doubt_rich", 1.0)
    dm = pm.Normal("doubt_multi", -2.0, 1.5)
    dbm = pm.HalfNormal("doubt_rich_multi", 1.0)
    ry = pm.Beta("random_yes", 2.0, 2.0)
    multi = 1.0 - single
    doubt_t = pm.Deterministic(
        "doubt", pm.math.sigmoid(single * (d0 + db * rich) + multi * (dm + dbm * rich))
    )
    # Grafted change: doubt mode keeps the examples' shared parity, so its flat
    # "yes" is discounted by an inferred fraction for parity-breaking targets.
    par_break = pm.Data("shared_parity_break", np.zeros(1, dtype="float64"))
    parity_keep = pm.Beta("parity_keep", alpha=2.0, beta=2.0)
    p_doubt = ry * (1.0 - parity_keep * par_break)
    p_yes = pm.Deterministic(
        "p_yes", pm.math.clip(doubt_t * p_doubt + (1.0 - doubt_t) * p_judge, 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
