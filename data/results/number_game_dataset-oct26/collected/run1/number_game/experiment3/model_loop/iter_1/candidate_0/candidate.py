"""Mental simulation of the program by mutating examples.

People imagine a handful of further outputs of the program, each produced by
taking one of the examples and altering it in a typical way (keep its final
digit and change its decade; keep its decade and change its final digit within
the same parity or to any digit; step through multiples of the examples' common
factor; or a random number), and they say "yes" if the target turns up at least
once among the imagined numbers. Acceptance is 1 - (1 - q)^k, where q is the
chance one imagined sample equals the target and k the number of samples, so it
saturates for easily generated targets and is near zero for hard ones.
"""
import math
from functools import reduce

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    n = float(len(xs))
    nums = range(1, 101)

    def share(pred_t, pred_n):
        # mean over examples of P(sample == t) under uniform choice among
        # numbers 1-100 satisfying the operation's constraint for that example
        tot = 0.0
        for x in xs:
            pool = [m for m in nums if pred_n(x, m)]
            if pool and pred_n(x, t):
                tot += 1.0 / len(pool)
        return tot / n

    g = reduce(math.gcd, xs)
    mult_pool = [m for m in nums if g >= 3 and m % g == 0]
    q_mult = (1.0 / len(mult_pool)) if (g >= 3 and t % g == 0) else 0.0
    return {
        "q_keep_digit": share(None, lambda x, m: m % 10 == x % 10),
        "q_keep_decade_parity": share(None, lambda x, m: m // 10 == x // 10 and m % 2 == x % 2),
        "q_keep_decade": share(None, lambda x, m: m // 10 == x // 10),
        "q_factor_step": q_mult,
    }


with pm.Model() as model:
    q_digit = pm.Data("q_keep_digit", np.zeros(1, dtype="float64"))
    q_dpar = pm.Data("q_keep_decade_parity", np.zeros(1, dtype="float64"))
    q_dec = pm.Data("q_keep_decade", np.zeros(1, dtype="float64"))
    q_fac = pm.Data("q_factor_step", np.zeros(1, dtype="float64"))

    # How often each kind of alteration is imagined (last = random number).
    w = pm.Dirichlet("op_weights", a=np.ones(5))
    # Number of imagined outputs.
    k = pm.LogNormal("n_samples", mu=np.log(10.0), sigma=1.0)

    q = (w[0] * q_digit + w[1] * q_dpar + w[2] * q_dec + w[3] * q_fac
         + w[4] * 0.01)
    p_raw = -pt.expm1(k * pt.log1p(-q))
    p_yes = pm.Deterministic("p_yes", pt.clip(p_raw, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
