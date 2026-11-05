"""Feature-overlap exemplar similarity.

People judge the target by its featural similarity to the shown examples, not
by inferring one hidden rule. Each number is a set of properties (parity,
prime, square, cube, powers, multiples, final digit, contained digits,
decade); the target's mean Jaccard overlap with the examples drives a graded
probability of saying yes.
"""
import numpy as np
import pymc as pm


def _props(n):
    s = set()
    s.add("odd" if n % 2 else "even")
    if n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1)):
        s.add("prime")
    if round(n ** 0.5) ** 2 == n:
        s.add("square")
    if round(n ** (1 / 3)) ** 3 == n:
        s.add("cube")
    for b in range(2, 11):
        p = 1
        while p <= 100:
            if p == n:
                s.add(f"pow{b}")
            p *= b
    for m in list(range(3, 10)) + [11, 12]:
        if n % m == 0:
            s.add(f"mult{m}")
    s.add(f"end{n % 10}")
    for ch in set(str(n)):
        s.add(f"has{ch}")
    s.add(f"dec{min((n - 1) // 10, 9)}")
    return s


_P = {n: _props(n) for n in range(1, 101)}


def compute_features(number_set, target):
    xs = [int(x) for x in str(number_set).split(",") if x.strip()]
    t = int(target)
    ft = _P[t]
    sims = [len(ft & _P[x]) / len(ft | _P[x]) for x in xs]
    return {"mean_overlap": float(np.mean(sims))}


with pm.Model() as model:
    mean_overlap = pm.Data("mean_overlap", np.zeros(1, dtype="float64"))
    intercept = pm.Normal("intercept", mu=-1.0, sigma=2.0)
    sensitivity = pm.HalfNormal("sensitivity", sigma=10.0)
    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept + sensitivity * (mean_overlap - 0.3))
    )
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
