"""Feature-prototype typicality (category-density prototype model).

People do not infer a rule; they abstract a prototype of the example set — how
often each familiar number property (odd/even, prime, square, cube, power,
multiple of 3-12) and each final digit occurs among the examples, plus the
examples' mean magnitude and spread — tracking each property independently and
hedging the estimates toward the property's base rate among 1-100 when there
are few examples. "Yes" rises with how much more typical the target is of this
prototype than of numbers in general (summed per-property and magnitude
likelihood ratios).
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMS = np.arange(1, 101)


def _is_prime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def _powers(b):
    s, v = set(), b
    while v <= 100:
        s.add(v)
        v *= b
    return s


SQUARES = {i * i for i in range(1, 11)}
CUBES = {i ** 3 for i in range(1, 5)}
PROPS = (
    [lambda n: n % 2 == 1, _is_prime, lambda n: n in SQUARES, lambda n: n in CUBES]
    + [(lambda s: (lambda n: n in s))(_powers(b)) for b in (2, 3, 5)]
    + [(lambda m: (lambda n: n % m == 0))(m) for m in range(3, 13)]
)
# binary property matrix over 1..100 and base rates
FMAT = np.array([[1.0 if p(int(n)) else 0.0 for p in PROPS] for n in NUMS])
BASE = FMAT.mean(axis=0)
K = FMAT.shape[1]


def prepare_observed(rows):
    n_ex, cnt, ft, dcnt, mu, ss, tgt, y = [], [], [], [], [], [], [], []
    for r in rows:
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        t = int(r["target"])
        idx = np.array(ex) - 1
        n_ex.append(len(ex))
        cnt.append(FMAT[idx].sum(axis=0))
        ft.append(FMAT[t - 1])
        dcnt.append(sum(1.0 for x in ex if x % 10 == t % 10))
        m = float(np.mean(ex))
        mu.append(m)
        ss.append(float(np.sum((np.array(ex) - m) ** 2)))
        tgt.append(float(t))
        y.append(int(float(r.get("rating", 0) or 0)))
    return {
        "n_examples": np.array(n_ex, dtype="float64"),
        "prop_counts": np.array(cnt, dtype="float64"),
        "target_props": np.array(ft, dtype="float64"),
        "digit_matches": np.array(dcnt, dtype="float64"),
        "example_mean": np.array(mu, dtype="float64"),
        "example_ss": np.array(ss, dtype="float64"),
        "target_value": np.array(tgt, dtype="float64"),
        "rating": np.array(y, dtype="int64"),
    }


with pm.Model() as model:
    n_examples = pm.Data("n_examples", np.ones(1, dtype="float64"))
    prop_counts = pm.Data("prop_counts", np.zeros((1, K), dtype="float64"))
    target_props = pm.Data("target_props", np.zeros((1, K), dtype="float64"))
    digit_matches = pm.Data("digit_matches", np.zeros(1, dtype="float64"))
    example_mean = pm.Data("example_mean", np.full(1, 50.0))
    example_ss = pm.Data("example_ss", np.zeros(1, dtype="float64"))
    target_value = pm.Data("target_value", np.full(1, 50.0))

    # pseudo-count pulling prototype feature rates toward base rates
    log_alpha = pm.Normal("log_alpha", mu=0.0, sigma=1.0)
    alpha = pm.Deterministic("alpha", pt.exp(log_alpha))
    # baseline magnitude spread (on the 1-100 scale) added to the examples' spread
    spread0 = pm.HalfNormal("spread0", sigma=30.0)
    # decision: sensitivity to typicality and criterion
    beta = pm.HalfNormal("beta", sigma=1.0)
    criterion = pm.Normal("criterion", mu=0.0, sigma=3.0)

    base = pt.as_tensor_variable(BASE)[None, :]
    n = n_examples[:, None]
    theta = (prop_counts + alpha * base) / (n + alpha)
    llr_props = pt.sum(
        target_props * pt.log(theta / base)
        + (1.0 - target_props) * pt.log((1.0 - theta) / (1.0 - base)),
        axis=1,
    )
    theta_d = (digit_matches + alpha * 0.1) / (n_examples + alpha)
    llr_digit = pt.log(theta_d / 0.1)
    var = example_ss / n_examples + spread0 ** 2 + 1.0
    llr_mag = (
        -0.5 * (target_value - example_mean) ** 2 / var
        - 0.5 * pt.log(2 * np.pi * var)
        + np.log(100.0)
    )
    typicality = llr_props + llr_digit + llr_mag

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(beta * (typicality - criterion)), 1e-6, 1 - 1e-6)
    )
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
