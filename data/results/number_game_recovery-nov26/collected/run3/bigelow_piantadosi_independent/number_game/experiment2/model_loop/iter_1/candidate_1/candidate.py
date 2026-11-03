"""Uncertainty-triggered guessing on costly-squares final-digit concept search.

People infer the hidden concept as the simple concept-search learner does
(concepts built from mental primitives with a fitted cost per extra primitive,
a discount for "ends in d" rules, separate extra costs for powers/cubes and for
perfect squares, Erlang-weighted intervals, the size principle and
near-all-or-none commitment). The one hypothesis is in the decision rule: the
closer their belief that the target belongs is to a toss-up, the more likely
they are to stop deliberating and simply guess (with a fitted guess rate).
The probability of guessing is a logistic function of the belief's
uncertainty 4p(1-p), so targets the concept clearly decides are answered from
the belief, while torn-about targets are answered near a coin flip.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N = 100
NUMS = np.arange(1, N + 1)


def _rules():
    # Each rule with its description length (number of primitives) and
    # whether it is a final-digit ("ends in d") rule,
    # whether it is a power/cube rule, and whether it is "perfect squares".
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
    return np.array(r, dtype=float), np.array(c, dtype=float), ends, exp_, sq


RULES, RULE_COST, IS_END, IS_EXP, IS_SQ = _rules()
_iv = [(a, b) for a in range(1, N + 1) for b in range(a, N + 1)]
INTERVALS = np.array([(NUMS >= a) & (NUMS <= b) for a, b in _iv], dtype=float)
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
    end_bonus = pm.Normal("final_digit_bonus", 0.0, 2.0)  # log-prior bonus for "ends in d" rules
    exp_cost = pm.Normal("exponent_cost", 0.0, 3.0)  # extra log-prior cost of power/cube rules
    sq_cost = pm.Normal("square_cost", 0.0, 3.0)  # extra log-prior cost of "perfect squares"
    log_sigma = pm.Normal("log_sigma", np.log(20.0), 1.0)  # Erlang scale of interval size
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))

    rule_logp = -cost * (RULE_COST - 1.0) + end_bonus * IS_END - exp_cost * IS_EXP - sq_cost * IS_SQ
    rule_logp = rule_logp - pt.logsumexp(rule_logp)  # normalised within rules
    int_logp = LOG_INT_SIZES - INT_SIZES / sigma
    int_logp = int_logp - pt.logsumexp(int_logp)  # normalised within intervals
    log_prior = pt.concatenate([pt.log(lam) + rule_logp, pt.log1p(-lam) + int_logp])
    logpost = beta * (set_loglik + log_prior[None, :])
    w = pt.special.softmax(logpost, axis=1)  # S x H
    p_in = pt.dot(w, pt.as_tensor_variable(MEMBER))  # S x 100
    p_raw = p_in[set_idx, target_idx]
    # Decision rule: guess when the belief is near a toss-up.
    guess_rate = pm.Beta("guess_rate", 2.0, 2.0)  # P(yes) when guessing
    guess_base = pm.Normal("guess_base", -4.0, 1.5)  # log-odds of guessing when certain
    guess_slope = pm.HalfNormal("guess_slope", 8.0)  # rise in guessing with uncertainty
    uncertainty = 4.0 * p_raw * (1.0 - p_raw)  # 0 when sure, 1 at p = 0.5
    p_guess = pm.math.sigmoid(guess_base + guess_slope * uncertainty)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(p_guess * guess_rate + (1 - p_guess) * p_raw, 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
