"""Forward-counting final-digit concepts with factor-graded multiples.

Refinement of forward_counting_final_digit_concepts: people build number
concepts from mental primitives (a fitted cost per extra primitive, a fitted
extra cost on power/square/cube concepts, a fitted prior discount on "ends in
d" rules, Erlang-weighted intervals, the size principle, near-all-or-none
commitment), and picture a range as counted upward from its start (a fitted
tilt g of the within-interval sampling density). The one change, taken from
factor_graded_multiples_concepts: "multiples of m" rules are less available
the larger m is, because divisibility is checked by recalling a times table
and larger tables are less fluent -- a multiples rule pays an extra prior
cost of mult_cost * log(m / 3).
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
IS_ENDS = np.zeros(len(RULE_COST))
IS_ENDS[-20:-10] = 1.0  # the ten "ends in d" rules
LOG_FACTOR = np.zeros(len(RULE_COST))
LOG_FACTOR[16:25] = np.log(np.array(list(range(3, 10)) + [11, 12]) / 3.0)  # multiples of m
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
            base = INTERVALS[:, x - 1] * (1 - NOISE) / INT_SIZES
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
    end_bonus = pm.Normal("ending_bonus", 0.0, 3.0)  # log-prior discount on "ends in d" rules
    mult_cost = pm.Normal("mult_cost", 0.0, 3.0)  # extra cost per log-unit of the factor m
    z_tilt = pm.Normal("z_forward", 0.0, 1.0)
    g = pm.Deterministic("forward_tilt", pt.tanh(z_tilt))  # >0: examples taken as low members of a range
    log_beta = pm.Normal("log_beta", 0.5, 1.0)  # commitment exponent (log scale)
    beta = pm.Deterministic("beta", pt.exp(log_beta))
    lapse = pm.Beta("lapse", 1.0, 10.0)
    log_sigma = pm.Normal("log_sigma", np.log(20.0), 1.0)  # Erlang scale of interval size
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))

    # log(base (1 + g u) + c) = log(base + c) + log1p(g * base u / (base + c)),
    # summed over examples; only intervals containing an example are tilted.
    tilt_ll = pt.inc_subtensor(
        pt.zeros(int_loglik0.shape[0] * NI)[tilt_slot], pt.log1p(g * tilt_coef)
    )
    int_ll = int_loglik0 + tilt_ll.reshape((int_loglik0.shape[0], NI))  # S x NI
    set_loglik = pt.concatenate([rule_loglik, int_ll], axis=1)

    rule_logp = -cost * (RULE_COST - 1.0) - exp_cost * IS_EXP + end_bonus * IS_ENDS - mult_cost * LOG_FACTOR
    rule_logp = rule_logp - pt.logsumexp(rule_logp)  # normalised within rules
    int_logp = LOG_INT_SIZES - INT_SIZES / sigma
    int_logp = int_logp - pt.logsumexp(int_logp)  # normalised within intervals
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
