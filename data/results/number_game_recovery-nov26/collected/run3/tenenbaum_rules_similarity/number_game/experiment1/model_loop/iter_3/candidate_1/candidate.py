"""Property-matching exemplar similarity (GCM over mathematical properties).

People do not infer a hidden concept: they store the shown examples as
exemplars and judge the target by its similarity to them, where two numbers
are similar to the extent that they share familiar mathematical properties
(parity, primality, square/cube, power of a base, multiple of a number, final
digit, digits contained) and are close in magnitude. Each kind of property has
its own attention weight, similarity decays exponentially with the
attention-weighted count of mismatching properties, and "yes" follows a Luce
choice between the mean similarity to the exemplars and a response criterion.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EX = 4
FAMILIES = ["parity", "prime", "square_cube", "power", "multiple",
            "last_digit", "contains_digit"]


def _is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def _properties(n):
    """Binary property vectors of number n, grouped by family."""
    powers = []
    for b in range(2, 11):
        p, s = b, set()
        while p <= 100:
            s.add(p)
            p *= b
        powers.append(1.0 if n in s else 0.0)
    return {
        "parity": [float(n % 2 == 1), float(n % 2 == 0)],
        "prime": [float(_is_prime(n))],
        "square_cube": [float(round(n ** 0.5) ** 2 == n),
                        float(round(n ** (1 / 3)) ** 3 == n)],
        "power": powers,
        "multiple": [float(n % m == 0) for m in (3, 4, 5, 6, 7, 8, 9, 11, 12)],
        "last_digit": [float(n % 10 == d) for d in range(10)],
        "contains_digit": [float(str(d) in str(n)) for d in range(10)],
    }


def compute_features(number_set, target):
    """Per-example family mismatch counts and distance to the target."""
    examples = [int(x) for x in number_set.split(",")][:MAX_EX]
    t = int(target)
    pt_ = _properties(t)
    out = {}
    for j in range(MAX_EX):
        present = j < len(examples)
        x = examples[j] if present else t
        px = _properties(x)
        for f in FAMILIES:
            out[f"mm_{f}_{j}"] = float(sum(abs(a - b) for a, b in zip(pt_[f], px[f]))) if present else 0.0
        out[f"dist_{j}"] = abs(t - x) / 100.0 if present else 0.0
        out[f"mask_{j}"] = 1.0 if present else 0.0
    return out


with pm.Model() as model:
    mm = {(f, j): pm.Data(f"mm_{f}_{j}", np.zeros(1, dtype="float64"))
          for j in range(MAX_EX) for f in FAMILIES}
    dist = [pm.Data(f"dist_{j}", np.zeros(1, dtype="float64")) for j in range(MAX_EX)]
    mask = [pm.Data(f"mask_{j}", np.zeros(1, dtype="float64")) for j in range(MAX_EX)]

    # Attention to each kind of property, and to magnitude.
    w = pm.HalfNormal("w", sigma=1.0, shape=len(FAMILIES))
    w_mag = pm.HalfNormal("w_mag", sigma=5.0)
    # Response criterion (log scale) of the Luce choice.
    log_k = pm.Normal("log_k", mu=-1.0, sigma=2.0)

    sims = []
    for j in range(MAX_EX):
        d = w_mag * dist[j]
        for i, f in enumerate(FAMILIES):
            d = d + w[i] * mm[(f, j)]
        sims.append(mask[j] * pt.exp(-d))
    n_ex = mask[0] + mask[1] + mask[2] + mask[3]
    mean_sim = (sims[0] + sims[1] + sims[2] + sims[3]) / pt.maximum(n_ex, 1.0)

    p = pm.math.sigmoid(pt.log(mean_sim + 1e-12) - log_k)
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
