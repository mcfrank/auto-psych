"""Forward-counting range sampling.

People assume the hidden program produces a numerical range by counting upward
from its start, so the examples they see are taken to be early, low members of
the range rather than evenly spread across it. A range that starts near the
examples and extends above them explains them better than one extending below,
with the strength (and sign) of this counting-forward assumption fitted. So
people generalise upward from the examples more than downward.

Built on multiplicative_not_exponential_concepts (primitive costs, extra cost on
powers/squares/cubes, size principle, Erlang intervals, near-all-or-none
commitment); the one change is the linearly tilted within-interval sampling
density, p(x | [a, b]) = (1 + g * u) / |[a, b]| with u = 1 - 2 (x - a) / (b - a).
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N = 100
NUMS = np.arange(1, N + 1)


def _rules():
    # Each rule with its description length and whether it needs exponentiation.
    r, c, e = [NUMS % 2 == 1, NUMS % 2 == 0], [1, 1], [0, 0]
    primes = np.array([n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1)) for n in NUMS])
    r.append(primes)
    r.append(np.isin(NUMS, [k * k for k in range(1, 11)]))
    r.append(np.isin(NUMS, [k ** 3 for k in range(1, 5)]))
    c += [2, 2, 2]
    e += [0, 1, 1]
    for b in range(2, 11):
        r.append(np.isin(NUMS, [b ** k for k in range(0, 8) if b ** k <= N]))
        c.append(2)
        e.append(1)
    p2 = [2 ** k for k in range(0, 7)]
    r.append(np.isin(NUMS, [x for x in p2 if x != 32]))
    r.append(np.isin(NUMS, p2 + [37]))
    c += [3, 3]
    e += [1, 1]
    for m in list(range(3, 10)) + [11, 12]:
        r.append(NUMS % m == 0)
        c.append(2)
        e.append(0)
    for d in range(10):
        r.append(NUMS % 10 == d)
        c.append(2)
        e.append(0)
    for d in range(10):
        r.append(np.array([str(d) in str(n) for n in NUMS]))
        c.append(2)
        e.append(0)
    return np.array(r, dtype=float), np.array(c, dtype=float), np.array(e, dtype=float)


RULES, RULE_COST, IS_EXP = _rules()
_iv = [(a, b) for a in range(1, N + 1) for b in range(a, N + 1)]
IV_A = np.array([a for a, b in _iv], dtype=float)
IV_B = np.array([b for a, b in _iv], dtype=float)
INTERVALS = np.array([(NUMS >= a) & (NUMS <= b) for a, b in _iv], dtype=float)
MEMBER = np.vstack([RULES, INTERVALS])  # H x 100
INT_SIZES = INTERVALS.sum(1)
LOG_INT_SIZES = np.log(INT_SIZES)
NR, NI = RULES.shape[0], INTERVALS.shape[0]
NOISE = 0.1


def prepare_observed(rows):
    sets = []
    idx = {}
    set_i, tgt = [], []
    for r in rows:
        s = str(r["number_set"]).strip()
        if s not in idx:
            idx[s] = len(sets)
            sets.append([int(x) for x in s.split(",")])
        set_i.append(idx[s])
        tgt.append(int(float(r["target"])) - 1)
    S = len(sets)
    rule_ll = np.zeros((S, NR))
    int_ll0 = np.zeros((S, NI))  # untilted interval log-likelihood
    flat, coef = [], []  # sparse (set, interval) slots and tilt coefficients
    rsize = RULES.sum(1)
    span = np.maximum(IV_B - IV_A, 1.0)
    for i, ex in enumerate(sets):
        for x in ex:
            rule_ll[i] += np.log(RULES[:, x - 1] * (1 - NOISE) / rsize + NOISE / N)
            inside = INTERVALS[:, x - 1]
            base = inside * (1 - NOISE) / INT_SIZES
            int_ll0[i] += np.log(base + NOISE / N)
            u = np.where(IV_B > IV_A, 1.0 - 2.0 * (x - IV_A) / span, 0.0)
            v = base * u / (base + NOISE / N)
            nz = np.nonzero(v)[0]
            flat.append(i * NI + nz)
            coef.append(v[nz])
    return {
        "rule_loglik": rule_ll,
        "int_loglik0": int_ll0,
        "tilt_slot": np.concatenate(flat).astype("int64"),
        "tilt_coef": np.concatenate(coef),
        "set_idx": np.array(set_i, dtype="int64"),
        "target_idx": np.array(tgt, dtype="int64"),
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    rule_loglik = pm.Data("rule_loglik", np.zeros((1, NR)))
    int_loglik0 = pm.Data("int_loglik0", np.zeros((1, NI)))
    tilt_slot = pm.Data("tilt_slot", np.zeros(1, dtype="int64"))
    tilt_coef = pm.Data("tilt_coef", np.zeros(1))
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))

    lam = pm.Beta("lambda_rules", 2.0, 2.0)  # prior mass on rules vs intervals
    cost = pm.HalfNormal("primitive_cost", sigma=5.0)  # log-prior cost per extra primitive
    exp_cost = pm.Normal("exponent_cost", 0.0, 3.0)  # extra cost of power/square/cube concepts
    z_tilt = pm.Normal("z_forward", 0.0, 1.0)
    g = pm.Deterministic("forward_tilt", pt.tanh(z_tilt))  # >0: examples taken as low members of a range
    log_beta = pm.Normal("log_beta", 0.5, 1.0)  # commitment exponent (log scale)
    beta = pm.Deterministic("beta", pt.exp(log_beta))
    lapse = pm.Beta("lapse", 1.0, 10.0)
    log_sigma = pm.Normal("log_sigma", np.log(20.0), 1.0)  # Erlang scale of interval size
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))

    # Interval likelihood under the forward-tilted sampling density.
    # log(base (1 + g u) + c) = log(base + c) + log1p(g * base u / (base + c)),
    # summed over examples; only intervals containing an example are tilted.
    tilt_ll = pt.inc_subtensor(
        pt.zeros(int_loglik0.shape[0] * NI)[tilt_slot], pt.log1p(g * tilt_coef)
    )
    int_ll = int_loglik0 + tilt_ll.reshape((int_loglik0.shape[0], NI))  # S x NI
    set_loglik = pt.concatenate([rule_loglik, int_ll], axis=1)

    rule_logp = -cost * (RULE_COST - 1.0) - exp_cost * IS_EXP
    rule_logp = rule_logp - pt.logsumexp(rule_logp)
    int_logp = LOG_INT_SIZES - INT_SIZES / sigma
    int_logp = int_logp - pt.logsumexp(int_logp)
    log_prior = pt.concatenate([pt.log(lam) + rule_logp, pt.log1p(-lam) + int_logp])
    logpost = beta * (set_loglik + log_prior[None, :])
    w = pt.special.softmax(logpost, axis=1)  # S x H
    p_in = pt.dot(w, pt.as_tensor_variable(MEMBER))  # S x 100
    p_raw = p_in[set_idx, target_idx]
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(lapse * 0.5 + (1 - lapse) * p_raw, 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
