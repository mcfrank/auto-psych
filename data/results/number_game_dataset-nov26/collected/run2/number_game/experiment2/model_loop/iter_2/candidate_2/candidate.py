"""Single-example magnitude neighborhood.

When only one example is shown, its magnitude itself becomes a candidate
property ("numbers about this size"): a target earns credit for it that falls
off smoothly (Gaussian) with its distance from the example, in both directions,
with its own inferred salience and width. Otherwise "yes" is a sigmoid of the
salience-weighted share of the examples' shared property profile the target has,
as in gap_step_progression_overlap (family saliences, base-ten multiples own
family, one-sided bounds, seed multiples, arithmetic-progression concept, a
visible shared final digit overshadowing other properties, the last example's
lingering final digit). Disagreement with the best model: one-example sets,
where near targets (above or below) gain and far targets lose relative to it.
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
    out["at_least_min"] = 1.0 if t >= min(ex) else 0.0
    out["at_most_max"] = 1.0 if t <= max(ex) else 0.0
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
    out["near_dist"] = min(abs(t - e) for e in ex) / 100.0
    out["single"] = 1.0 if len(ex) == 1 else 0.0
    linger = len(ex) >= 2 and not shared_digit
    out["last_digit_active"] = 1.0 if linger else 0.0
    out["last_digit_match"] = 1.0 if (linger and t % 10 == ex[-1] % 10) else 0.0
    return out


with pm.Model() as model:
    sh = [pm.Data(f"sh_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    hit = [pm.Data(f"hit_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    ge_min = pm.Data("at_least_min", np.zeros(1, dtype="float64"))
    le_max = pm.Data("at_most_max", np.zeros(1, dtype="float64"))
    vis = pm.Data("visible_digit_pattern", np.zeros(1, dtype="float64"))
    seed_shared = pm.Data("seed_shared", np.zeros(1, dtype="float64"))
    seed_hit = pm.Data("seed_hit", np.zeros(1, dtype="float64"))
    last_on = pm.Data("last_digit_active", np.zeros(1, dtype="float64"))
    last_hit = pm.Data("last_digit_match", np.zeros(1, dtype="float64"))
    prog_shared = pm.Data("progression_shared", np.zeros(1, dtype="float64"))
    prog_hit = pm.Data("progression_hit", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    salience = pm.Dirichlet("salience", a=np.full(N_FAM + 4, 1.5))
    retain = pm.Beta("retain", alpha=2.0, beta=2.0)
    recency = pm.HalfNormal("recency", sigma=1.0)

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
    near_dist = pm.Data("near_dist", np.zeros(1, dtype="float64"))
    single = pm.Data("single", np.zeros(1, dtype="float64"))
    # Width of the magnitude neighborhood (1-100 scale / 100) and its salience.
    width = pm.LogNormal("width", mu=np.log(0.5), sigma=0.7)
    s_near = pm.HalfNormal("s_near", sigma=0.5)
    nb = pm.math.exp(-((near_dist / width) ** 2))
    # Only a lone example is read as a magnitude neighborhood.
    w_near = single * s_near
    num = num + w_near * nb
    den = den + w_near
    overlap = num / den
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * overlap))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
