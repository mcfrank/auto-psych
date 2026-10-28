"""Factorized rule-and-range Bayesian concept learning.

People are Bayesian concept learners who think the program's numbers obey both
an arithmetic rule (a residue class "x mod m == r" for m = 1..12, where m = 1
means no pattern; this covers even/odd, multiples of m and "ends in d") and a
numeric range (an interval of 1..100), each inferred from the examples with
the size principle. The one distortion: rule and range are inferred as two
separate, independent questions (a factorized approximation to joint
inference), and the probability of "yes" is the product of the posterior
probability that the target fits the rule and that it lies in the range,
with occasional lapses to guessing.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MODS = np.arange(1, 13)
NM = len(MODS)
SIZES = np.arange(1, 101, dtype="float64")


def _count_intervals(a, b):
    """Number of intervals of each size L (1..100) within 1..100 containing [a, b]."""
    L = SIZES.astype(int)
    lo_max = np.minimum(a, 101 - L)
    lo_min = np.maximum(1, b - L + 1)
    return np.maximum(0, lo_max - lo_min + 1).astype("float64")


def prepare_observed(rows):
    n = len(rows)
    consistent = np.zeros((n, NM))      # residue class mod m shared by all examples
    rule_loglik = np.zeros((n, NM))     # n_examples * -log |class|
    target_in = np.zeros((n, NM))       # target in that class
    num_count = np.zeros((n, 100))      # intervals containing examples and target
    den_count = np.zeros((n, 100))      # intervals containing examples
    n_ex = np.zeros(n)
    rating = np.zeros(n, dtype="int64")
    cache = {}
    for i, row in enumerate(rows):
        s = str(row["number_set"])
        t = int(row["target"])
        ex = [int(x) for x in s.split(",")]
        if s not in cache:
            cons, ll = np.zeros(NM), np.zeros(NM)
            for j, m in enumerate(MODS):
                r = ex[0] % m
                if all(x % m == r for x in ex):
                    size = sum(1 for v in range(1, 101) if v % m == r)
                    cons[j] = 1.0
                    ll[j] = -len(ex) * np.log(size)
            cache[s] = (cons, ll, _count_intervals(min(ex), max(ex)))
        cons, ll, den = cache[s]
        consistent[i] = cons
        rule_loglik[i] = ll
        target_in[i] = [1.0 if t % m == ex[0] % m else 0.0 for m in MODS]
        target_in[i] *= cons
        num_count[i] = _count_intervals(min(ex + [t]), max(ex + [t]))
        den_count[i] = den
        n_ex[i] = len(ex)
        rating[i] = int(float(row["rating"]))
    return {"rule_consistent": consistent, "rule_loglik": rule_loglik,
            "rule_target_in": target_in, "range_num": num_count,
            "range_den": den_count, "n_examples": n_ex, "rating": rating}


LOG_M = np.log(MODS.astype("float64"))
IS_ONE = (MODS == 1).astype("float64")
IS_TEN = (MODS == 10).astype("float64")

with pm.Model() as model:
    rule_consistent = pm.Data("rule_consistent", np.ones((1, NM)))
    rule_loglik = pm.Data("rule_loglik", np.zeros((1, NM)))
    rule_target_in = pm.Data("rule_target_in", np.zeros((1, NM)))
    range_num = pm.Data("range_num", np.ones((1, 100)))
    range_den = pm.Data("range_den", np.ones((1, 100)))
    n_examples = pm.Data("n_examples", np.ones(1))

    # Rule prior: log-weight of "no pattern" (m = 1), decay with step size m,
    # and a bonus for decimal final-digit rules (m = 10).
    w_none = pm.Normal("w_none", mu=0.0, sigma=3.0)
    step_decay = pm.HalfNormal("step_decay", sigma=2.0)
    bonus_ten = pm.Normal("bonus_ten", mu=0.0, sigma=2.0)
    # Range prior: Erlang density on interval size with scale range_scale.
    range_scale = pm.LogNormal("range_scale", mu=np.log(20.0), sigma=1.0)
    lapse = pm.Beta("lapse", 2.0, 8.0)
    guess = pm.Beta("guess", 2.0, 2.0)

    # Rule posterior (only classes consistent with every example).
    prior_logit = (w_none * IS_ONE - step_decay * LOG_M * (1 - IS_ONE)
                   + bonus_ten * IS_TEN)
    log_post = prior_logit[None, :] + rule_loglik + pt.log(rule_consistent + 1e-30)
    log_post = log_post - pt.max(log_post, axis=1, keepdims=True)
    u = pt.exp(log_post) * rule_consistent
    p_rule = pt.sum(u * rule_target_in, axis=1) / pt.sum(u, axis=1)

    # Range posterior: per-interval weight Erlang(L) * L^-n.
    log_w = (np.log(SIZES)[None, :] - SIZES[None, :] / range_scale
             - n_examples[:, None] * np.log(SIZES)[None, :])
    log_w = log_w - pt.max(log_w, axis=1, keepdims=True)
    w = pt.exp(log_w)
    p_range = pt.sum(w * range_num, axis=1) / pt.sum(w * range_den, axis=1)

    p = (1 - lapse) * p_rule * p_range + lapse * guess
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
