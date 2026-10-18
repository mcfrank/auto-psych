"""Example-by-example Bayesian generalization (no evidence integration).

People generalize like an ideal Bayesian concept learner -- weighing candidate
concepts (same final digit, same parity, same decade, multiples of 3-10, or a
numerical interval) by their prior plausibility and the size principle -- but
with one resource limit: they cannot integrate the examples jointly, so they
generalize from each example on its own and answer "yes" according to the
average of these single-example generalizations. Evidence never accumulates
across examples, so judgments do not sharpen as examples are added.
"""
import numpy as np
import pymc as pm

K = 4
N = 100
FAMILIES = ("digit", "parity", "decade", "multiple", "interval")
F = len(FAMILIES)
NUMS = np.arange(1, N + 1)


def _family_hypotheses():
    """Per family: list of boolean membership vectors over 1..100."""
    fam = {
        "digit": [NUMS % 10 == d for d in range(10)],
        "parity": [NUMS % 2 == r for r in range(2)],
        "decade": [NUMS // 10 == d for d in range(11)],
        "multiple": [NUMS % k == 0 for k in range(3, 11)],
    }
    return fam


def _single_example_tables():
    """For each example x: evidence Z_f(x) and predictive P_f(t | x) per family.

    Within a family hypotheses are equally likely a priori; the likelihood of
    one example is 1/|h| for h containing it (the size principle).
    """
    fam = _family_hypotheses()
    Z = np.zeros((N + 1, F))
    P = np.zeros((N + 1, F, N + 1))
    # Interval family: all [a, b] with 1 <= a <= b <= 100, uniform prior.
    n_int = N * (N + 1) / 2
    for x in range(1, N + 1):
        for fi, name in enumerate(FAMILIES[:-1]):
            hs = fam[name]
            z = 0.0
            pred = np.zeros(N)
            for h in hs:
                if h[x - 1]:
                    w = (1.0 / len(hs)) / h.sum()
                    z += w
                    pred += w * h
            Z[x, fi] = z
            P[x, fi, 1:] = pred / z if z > 0 else 0.0
        z = 0.0
        pred = np.zeros(N)
        for a in range(1, x + 1):
            for b in range(x, N + 1):
                w = (1.0 / n_int) / (b - a + 1)
                z += w
                pred[a - 1:b] += w
        Z[x, F - 1] = z
        P[x, F - 1, 1:] = pred / z
    return Z, P


_Z, _P = _single_example_tables()


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    out = {}
    for k in range(K):
        for fi, name in enumerate(FAMILIES):
            if k < len(xs):
                out[f"z_{name}_{k}"] = float(_Z[xs[k], fi] * 100.0)
                out[f"p_{name}_{k}"] = float(_P[xs[k], fi, t])
            else:
                out[f"z_{name}_{k}"] = 1.0
                out[f"p_{name}_{k}"] = 0.0
        out[f"slot_w{k}"] = 1.0 / len(xs) if k < len(xs) else 0.0
    return out


with pm.Model() as model:
    zs = [[pm.Data(f"z_{name}_{k}", np.ones(1, dtype="float64")) for name in FAMILIES]
          for k in range(K)]
    ps = [[pm.Data(f"p_{name}_{k}", np.zeros(1, dtype="float64")) for name in FAMILIES]
          for k in range(K)]
    slot_w = [pm.Data(f"slot_w{k}", np.zeros(1, dtype="float64")) for k in range(K)]

    # Prior plausibility of each concept family.
    family_prior = pm.Dirichlet("family_prior", a=np.ones(F))

    belief = 0.0
    for k in range(K):
        num = 0.0
        den = 0.0
        for fi in range(F):
            wz = family_prior[fi] * zs[k][fi]
            num = num + wz * ps[k][fi]
            den = den + wz
        belief = belief + slot_w[k] * num / den

    intercept = pm.Normal("intercept", mu=-2.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * belief))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
