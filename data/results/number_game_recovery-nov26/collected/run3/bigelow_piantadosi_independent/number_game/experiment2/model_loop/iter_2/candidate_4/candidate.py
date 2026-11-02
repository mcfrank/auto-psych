"""Prime-salient concepts on the hundred-chart row-ranges model.

Refinement of hundred_chart_row_ranges, whose content follows. The one added
change (from prime_salient_concepts): "prime numbers" is a familiar
school-math category with a fitted prior bonus over other compound rules, so
after all-prime sets people say yes to other primes more than to odd
composites.

Base model: refinement of factor_graded_multiples_concepts: everything is kept (concepts
from mental primitives with a fitted cost per extra primitive, a final-digit
discount, extra costs for powers/cubes and for squares, multiples costs graded
by log(m / 3), Erlang-weighted intervals, size principle, near-all-or-none
commitment). The one change: people picture 1-100 as a hundred chart of ten
rows (1-10, 11-20, ..., 91-100), so intervals made of whole rows (from 10k+1
to 10j, e.g. 1-50, 21-40, 81-90, 1-100) get a fitted prior bonus over
intervals with arbitrary endpoints. When no rule fits, people fall back on a
broad round block of the chart rather than a tight range hugging the examples.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N = 100
NUMS = np.arange(1, N + 1)


def _rules():
    # Each rule with its description length (number of primitives),
    # whether it is a final-digit ("ends in d") rule, whether it is a
    # power/cube rule, whether it is "perfect squares", and log(m/3) for
    # "multiples of m".
    r, c = [NUMS % 2 == 1, NUMS % 2 == 0], [1, 1]
    primes = np.array([n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1)) for n in NUMS])
    r.append(primes)
    r.append(np.isin(NUMS, [k * k for k in range(1, 11)]))
    r.append(np.isin(NUMS, [k ** 3 for k in range(1, 5)]))
    c += [2, 2, 2]
    for b in range(2, 11):
        r.append(np.isin(NUMS, [b ** k for k in range(0, 8) if b ** k <= N]))
        c.append(2)
    p2 = [2 ** k for k in range(0, 7)]
    r.append(np.isin(NUMS, [x for x in p2 if x != 32]))
    r.append(np.isin(NUMS, p2 + [37]))
    c += [3, 3]
    for m in list(range(3, 10)) + [11, 12]:
        r.append(NUMS % m == 0)
        c.append(2)
    n_before = len(r)
    for d in range(10):
        r.append(NUMS % 10 == d)
        c.append(2)
    ends = np.zeros(len(r) + 10)
    ends[n_before:len(r)] = 1.0
    for d in range(10):
        r.append(np.array([str(d) in str(n) for n in NUMS]))
        c.append(2)
    exp_ = np.zeros(len(r))
    exp_[4:16] = 1.0  # cubes, powers of 2-10, two powers-of-2 variants
    sq = np.zeros(len(r))
    sq[3] = 1.0  # perfect squares
    log_m = np.zeros(len(r))
    log_m[18:27] = np.log(np.array(list(range(3, 10)) + [11, 12]) / 3.0)  # multiples of m
    pr = np.zeros(len(r))
    pr[2] = 1.0  # prime numbers
    return np.array(r, dtype=float), np.array(c, dtype=float), ends, exp_, sq, log_m, pr


RULES, RULE_COST, IS_END, IS_EXP, IS_SQ, LOG_FACTOR, IS_PRIME = _rules()
_iv = [(a, b) for a in range(1, N + 1) for b in range(a, N + 1)]
INTERVALS = np.array([(NUMS >= a) & (NUMS <= b) for a, b in _iv], dtype=float)
IS_ROWS = np.array([1.0 if a % 10 == 1 and b % 10 == 0 else 0.0 for a, b in _iv])  # whole chart rows
MEMBER = np.vstack([RULES, INTERVALS])  # H x 100
N_RULES = RULES.shape[0]
N_INT = INTERVALS.shape[0]
SIZES = MEMBER.sum(1)
INT_SIZES = INTERVALS.sum(1)
LOG_INT_SIZES = np.log(INT_SIZES)
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
    loglik = np.zeros((len(sets), MEMBER.shape[0]))
    for i, ex in enumerate(sets):
        for x in ex:
            pin = MEMBER[:, x - 1] * (1 - NOISE) / SIZES
            loglik[i] += np.log(pin + NOISE / N)
    return {
        "set_loglik": loglik,
        "set_idx": np.array(set_i, dtype="int64"),
        "target_idx": np.array(tgt, dtype="int64"),
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    set_loglik = pm.Data("set_loglik", np.zeros((1, MEMBER.shape[0])))
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))

    lam = pm.Beta("lambda_rules", 2.0, 2.0)  # prior mass on rules vs intervals
    cost = pm.HalfNormal("primitive_cost", sigma=5.0)  # log-prior cost per extra primitive
    log_beta = pm.Normal("log_beta", 0.5, 1.0)  # commitment exponent (log scale)
    beta = pm.Deterministic("beta", pt.exp(log_beta))
    lapse = pm.Beta("lapse", 1.0, 10.0)
    end_bonus = pm.Normal("final_digit_bonus", 0.0, 2.0)  # log-prior bonus for "ends in d" rules
    exp_cost = pm.Normal("exponent_cost", 0.0, 3.0)  # extra log-prior cost of power/cube rules
    sq_cost = pm.Normal("square_cost", 0.0, 3.0)  # extra log-prior cost of "perfect squares"
    mult_cost = pm.Normal("mult_cost", 0.0, 3.0)  # extra cost per log-unit of the factor m
    row_bonus = pm.Normal("row_bonus", 0.0, 5.0)  # log-prior bonus for whole-row intervals
    prime_bonus = pm.Normal("prime_bonus", 0.0, 3.0)  # log-prior bonus for "prime numbers"
    log_sigma = pm.Normal("log_sigma", np.log(20.0), 1.0)  # Erlang scale of interval size
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))

    rule_logp = -cost * (RULE_COST - 1.0) + end_bonus * IS_END - exp_cost * IS_EXP - sq_cost * IS_SQ - mult_cost * LOG_FACTOR + prime_bonus * IS_PRIME
    rule_logp = rule_logp - pt.logsumexp(rule_logp)  # normalised within rules
    int_logp = LOG_INT_SIZES - INT_SIZES / sigma + row_bonus * IS_ROWS
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
