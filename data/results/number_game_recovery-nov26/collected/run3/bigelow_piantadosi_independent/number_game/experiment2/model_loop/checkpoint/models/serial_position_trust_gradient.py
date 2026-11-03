"""Serial-position trust gradient on the prime-over-odd concept learner.

People revise their guess about the hidden rule as each example arrives, so
they trust an example more the later it comes: the chance that an example is
set aside as a stray the program did not really mean falls steadily (in
logit, at a fitted slope) from the first example shown to the last, centred
on the usual 10% stray rate. Everything else is the prime_over_odd_reading
learner: primitive-built rules with a per-primitive cost, final-digit
discount, costs on powers/cubes and squares, times-table-graded multiples,
"prime" preferred over "odd", Erlang ranges with a whole-hundred-chart-row
bonus, the size principle and near-all-or-none commitment.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N = 100
NUMS = np.arange(1, N + 1)


def _rules():
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
    log_m[18:27] = np.log(np.array(list(range(3, 10)) + [11, 12]) / 3.0)  # as in the incumbent
    pr = np.zeros(len(r))
    pr[2] = 1.0  # prime numbers gain the preference
    pr[0] = -1.0  # odd numbers lose it
    return np.array(r, dtype=float), np.array(c, dtype=float), ends, exp_, sq, log_m, pr


RULES, RULE_COST, IS_END, IS_EXP, IS_SQ, LOG_FACTOR, PRIME_VS_ODD = _rules()
_iv = [(a, b) for a in range(1, N + 1) for b in range(a, N + 1)]
INTERVALS = np.array([(NUMS >= a) & (NUMS <= b) for a, b in _iv], dtype=float)
IS_ROWS = np.array([1.0 if a % 10 == 1 and b % 10 == 0 else 0.0 for a, b in _iv])
MEMBER = np.vstack([RULES, INTERVALS])  # H x 100
H = MEMBER.shape[0]
SIZES = MEMBER.sum(1)
INT_SIZES = INTERVALS.sum(1)
LOG_INT_SIZES = np.log(INT_SIZES)
# Serial positions, centred: first example -0.5, last +0.5, a lone example 0.
POS_LEVELS = np.array([-0.5, -1.0 / 6.0, 0.0, 1.0 / 6.0, 0.5])
NZ = len(POS_LEVELS)


def _pos_level(k, n):
    z = 0.0 if n == 1 else -0.5 + k / (n - 1)
    return int(np.argmin(np.abs(POS_LEVELS - z)))


def prepare_observed(rows):
    sets, idx, set_i, tgt = [], {}, [], []
    for r in rows:
        s = str(r["number_set"]).strip()
        if s not in idx:
            idx[s] = len(sets)
            sets.append([int(x) for x in s.split(",")])
        set_i.append(idx[s])
        tgt.append(int(float(r["target"])) - 1)
    # For each set: how many examples sit at each serial-position level, and
    # how many of them each hypothesis contains (flattened level-major).
    counts = np.zeros((len(sets), NZ))
    member = np.zeros((len(sets), NZ, H))
    for i, ex in enumerate(sets):
        for k, x in enumerate(ex):
            z = _pos_level(k, len(ex))
            counts[i, z] += 1.0
            member[i, z] += MEMBER[:, x - 1]
    return {
        "pos_counts": counts,
        "pos_member": member.reshape(len(sets), NZ * H),
        "set_idx": np.array(set_i, dtype="int64"),
        "target_idx": np.array(tgt, dtype="int64"),
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    pos_counts = pm.Data("pos_counts", np.zeros((1, NZ)))
    pos_member = pm.Data("pos_member", np.zeros((1, NZ * H)))
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))

    lam = pm.Beta("lambda_rules", 2.0, 2.0)
    cost = pm.HalfNormal("primitive_cost", sigma=5.0)
    log_beta = pm.Normal("log_beta", 0.5, 1.0)
    beta = pm.Deterministic("beta", pt.exp(log_beta))
    lapse = pm.Beta("lapse", 1.0, 10.0)
    end_bonus = pm.Normal("final_digit_bonus", 0.0, 2.0)
    exp_cost = pm.Normal("exponent_cost", 0.0, 3.0)
    sq_cost = pm.Normal("square_cost", 0.0, 3.0)
    mult_cost = pm.Normal("mult_cost", 0.0, 3.0)
    row_bonus = pm.Normal("row_bonus", 0.0, 5.0)
    prime_over_odd = pm.Normal("prime_over_odd", 0.0, 3.0)
    log_sigma = pm.Normal("log_sigma", np.log(20.0), 1.0)
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))
    # Trust gradient: negative means later examples are less often dismissed.
    trust_slope = pm.Normal("trust_slope", 0.0, 1.5)

    stray = pm.math.sigmoid(np.log(1.0 / 9.0) + trust_slope * POS_LEVELS)  # stray rate per position
    # log P(example | h) = log(stray/N) + [x in h] * log(1 + (1-stray) N / (stray |h|))
    gain = pt.log1p((1.0 - stray)[:, None] * N / (stray[:, None] * SIZES[None, :]))  # NZ x H
    set_loglik = pt.dot(pos_counts, pt.log(stray / N))[:, None] + sum(
        pos_member[:, z * H:(z + 1) * H] * gain[z][None, :] for z in range(NZ)
    )  # S x H

    rule_logp = -cost * (RULE_COST - 1.0) + end_bonus * IS_END - exp_cost * IS_EXP - sq_cost * IS_SQ - mult_cost * LOG_FACTOR + prime_over_odd * PRIME_VS_ODD
    rule_logp = rule_logp - pt.logsumexp(rule_logp)
    int_logp = LOG_INT_SIZES - INT_SIZES / sigma + row_bonus * IS_ROWS
    int_logp = int_logp - pt.logsumexp(int_logp)
    log_prior = pt.concatenate([pt.log(lam) + rule_logp, pt.log1p(-lam) + int_logp])
    logpost = beta * (set_loglik + log_prior[None, :])
    w = pt.special.softmax(logpost, axis=1)
    p_in = pt.dot(w, pt.as_tensor_variable(MEMBER))
    p_raw = p_in[set_idx, target_idx]
    p_yes = pm.Deterministic("p_yes", pt.clip(lapse * 0.5 + (1 - lapse) * p_raw, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
