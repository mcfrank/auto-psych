"""Coherence-relative featural similarity.

People judge a target by its similarity to the examples, where two numbers are
similar to the extent that they share number properties (parity, primality,
square/cube/power, divisibility by 3-12, final digit, digits contained, and a
magnitude band of width 20), each shared property weighted by its information
content log(100/|property|). The criterion is relative: "yes" becomes likely
when the target's mean similarity to the examples reaches (a fraction of) the
examples' mean similarity to one another, so loosely related sets accept
loosely related targets and tightly related sets demand the same tight
relation. A lone example has no partner, so it is judged against a fixed
default criterion.
"""
import itertools
import numpy as np
import pymc as pm


def _is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def _similarity_table():
    N = range(1, 101)
    feats = [
        {n for n in N if n % 2 == 1},
        {n for n in N if n % 2 == 0},
        {n for n in N if _is_prime(n)},
        {k * k for k in range(1, 11)},
        {k ** 3 for k in range(1, 5)},
    ]
    for b in range(2, 11):
        feats.append({b ** k for k in range(1, 8) if b ** k <= 100})
    for m in range(3, 13):
        feats.append({n for n in N if n % m == 0})
    for d in range(10):
        feats.append({n for n in N if n % 10 == d})
    for d in range(10):
        feats.append({n for n in N if str(d) in str(n)})
    for lo in range(1, 101, 20):
        feats.append(set(range(lo, lo + 20)))
    feats = [f for f in feats if 0 < len(f) < 100]
    M = np.array([[1.0 if n in f else 0.0 for n in N] for f in feats])
    w = np.log(100.0 / M.sum(axis=1))
    return (M * w[:, None]).T @ M  # 100 x 100 shared-information similarity


_SIM = _similarity_table()


def prepare_observed(rows):
    sim_t, coh, lone, y = [], [], [], []
    for r in rows:
        ex = sorted({int(float(x)) for x in str(r["number_set"]).split(",")})
        t = int(float(r["target"]))
        sim_t.append(float(np.mean([_SIM[t - 1, e - 1] for e in ex])))
        if len(ex) > 1:
            coh.append(float(np.mean([_SIM[a - 1, b - 1] for a, b in itertools.combinations(ex, 2)])))
            lone.append(0.0)
        else:
            coh.append(0.0)
            lone.append(1.0)
        y.append(int(float(r["rating"])))
    return {
        "target_similarity": np.array(sim_t, dtype="float64"),
        "set_coherence": np.array(coh, dtype="float64"),
        "lone_example": np.array(lone, dtype="float64"),
        "rating": np.array(y, dtype="int64"),
    }


with pm.Model() as model:
    target_similarity = pm.Data("target_similarity", np.zeros(1, dtype="float64"))
    set_coherence = pm.Data("set_coherence", np.zeros(1, dtype="float64"))
    lone_example = pm.Data("lone_example", np.zeros(1, dtype="float64"))

    # response bias at the criterion
    bias = pm.Normal("bias", mu=0.0, sigma=2.0)
    # sensitivity of "yes" to similarity above/below the criterion
    slope = pm.HalfNormal("slope", sigma=1.0)
    # fraction of the examples' mutual similarity the target must reach
    coh_frac = pm.HalfNormal("coh_frac", sigma=1.0)
    # default criterion for a lone example
    lone_crit = pm.Normal("lone_crit", mu=2.0, sigma=2.0)

    criterion = lone_example * lone_crit + (1.0 - lone_example) * coh_frac * set_coherence
    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(bias + slope * (target_similarity - criterion))
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
