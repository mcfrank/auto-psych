"""Final-digit overshadowing feature overlap.

Refinement of one_sided_magnitude_bound_overlap: people say "yes" in
proportion to the salience-weighted share of the examples' shared property
profile that the target also has (family saliences, base-ten multiples their
own family, two one-sided magnitude bounds, no size principle). The one
change: when several (>=2) examples visibly share a final digit, that
notational pattern overshadows every other shared property, whose saliences
are all discounted by one inferred factor. So on shared-final-digit sets,
targets keeping the digit are endorsed more and targets matching only the
other properties less than under the incumbent.
"""
import numpy as np
import pymc as pm

_N = range(1, 101)
N_FAM = 8  # family 5 = final digit; bounds are salience entries 8, 9
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
    out["visible_digit_pattern"] = 1.0 if (len(ex) >= 2 and len({e % 10 for e in ex}) == 1) else 0.0
    return out


with pm.Model() as model:
    sh = [pm.Data(f"sh_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    hit = [pm.Data(f"hit_{f}", np.zeros(1, dtype="float64")) for f in range(N_FAM)]
    ge_min = pm.Data("at_least_min", np.zeros(1, dtype="float64"))
    le_max = pm.Data("at_most_max", np.zeros(1, dtype="float64"))
    vis = pm.Data("visible_digit_pattern", np.zeros(1, dtype="float64"))

    intercept = pm.Normal("intercept", mu=-2.0, sigma=1.5)
    slope = pm.HalfNormal("slope", sigma=5.0)
    salience = pm.Dirichlet("salience", a=np.full(N_FAM + 2, 1.5))
    # fraction of salience the other properties keep under a visible digit pattern
    retain = pm.Beta("retain", alpha=2.0, beta=2.0)

    other_w = 1.0 - (1.0 - retain) * vis
    num = other_w * (salience[N_FAM] * ge_min + salience[N_FAM + 1] * le_max)
    den = other_w * (salience[N_FAM] + salience[N_FAM + 1])
    for f in range(N_FAM):
        w = salience[f] if f == FD else salience[f] * other_w
        num = num + w * hit[f]
        den = den + w * sh[f]
    overlap = num / den
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + slope * overlap))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
