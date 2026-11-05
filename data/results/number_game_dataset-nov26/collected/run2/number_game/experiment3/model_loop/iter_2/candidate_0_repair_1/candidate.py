"""Power-chain generation, on top of rule_ambiguity_single_doubt.

When several examples are shown, people try to reconstruct how the program
generated them, and one operation they readily try is repeated multiplication:
if all examples are powers of a common base (4, 64; 9, 81; 2, 8, 16), they read
the program as "keep multiplying by the base" and take the other links of
that chain as produced numbers. This chain concept, weighted by an inferred
boost over the powers family salience, is
not overshadowed by a visible shared final digit (it describes how the numbers
were made, not how they look). Everything else is the best account unchanged:
the salience-weighted share of the examples' shared profile with the in-span
interval similarity, and a richness-dependent flat-rate doubt on lone examples.
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
    out["n1_richness"] = (float(sum(1 for fam,p in _PROPS if ex[0] in p)) - 6.0) if not span_on else 0.0
    out["span_active"] = 1.0 if span_on else 0.0
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
    chain = None
    if len(set(ex)) >= 2:
        for b in range(2, 11):
            pw = {b ** k for k in range(1, 8) if b ** k <= 100}
            if all(e in pw for e in ex):
                chain = pw
                break
    out["chain_shared"] = 1.0 if chain else 0.0
    out["chain_hit"] = 1.0 if (chain and t in chain) else 0.0
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
    span_gap = pm.Data("span_gap", np.zeros(1, dtype="float64"))
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
    span_sim = span_on * pm.math.exp(-(span_gap / (0.01 + span_w)))
    w_span = other_w * salience[N_FAM + 4]
    num = num + w_span * span_sim
    den = den + w_span * span_on
    ch_s = pm.Data("chain_shared", np.zeros(1, dtype="float64"))
    ch_h = pm.Data("chain_hit", np.zeros(1, dtype="float64"))
    # Power-chain concept: its weight is the powers family's salience scaled
    # by an inferred generative boost, and it is not overshadowed by a shared
    # digit (kept out of the Dirichlet: few sets qualify, so a free salience
    # component there is weakly identified).
    chain_boost = pm.LogNormal("chain_boost", mu=0.0, sigma=0.75)
    w_ch = chain_boost * salience[3]
    num = num + w_ch * ch_h
    den = den + w_ch * ch_s
    overlap = num / den
    p_judge = pm.math.sigmoid(intercept + slope * overlap)
    # The one change: doubt on a lone example depends on how many candidate
    # rules (property categories) that example belongs to (centred at 6).
    rich = pm.Data('n1_richness', np.zeros(1, dtype='float64'))
    d0 = pm.Normal('doubt0', 0.0, 1.5)
    db = pm.Normal('doubt_rich', 0.0, 1.0)
    ry = pm.Beta('random_yes', 2.0, 2.0)
    doubt_t = pm.math.sigmoid(d0 + db * rich)
    yes_floor = doubt_t*ry
    judge_share = 1.0-doubt_t
    pm.Deterministic("doubt", 1.0 - judge_share)
    p_single = yes_floor + judge_share * p_judge
    p_yes = pm.Deterministic("p_yes", single * p_single + (1.0 - single) * p_judge)

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
