"""Fives bonus on the prime-over-odd hundred-chart model.

Refinement of prime_over_odd_reading, with everything kept: primitive-built rules
with a per-primitive cost, the final-digit discount, costs on powers/cubes and
squares, times-table-graded multiples, one preference that moves log prior from
"odd numbers" onto "prime numbers", Erlang ranges with a bonus for whole
hundred-chart rows, the size principle and near-all-or-none commitment. The
one change is that "multiples of 5" can be read straight off the numeral (it
ends in 0 or 5) and counting by fives is drilled, so that rule gets its own
fitted prior bonus. After sets of multiples of 5, people say yes to other
multiples of 5 and no to their neighbours more sharply than the incumbent
predicts.

Implementation note: an interval's posterior weight depends only on its size,
whether it is made of whole chart rows, and how many examples it contains. So
intervals are summed exactly in groups of those three properties, with counts
computed once per set in prepare_observed. This gives the same predictions as
enumerating all 5050 intervals, at a fraction of the cost.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N = 100
NUMS = np.arange(1, N + 1)
MULT5 = 18  # row index of "multiples of 5"


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
    five = np.zeros(len(r))
    five[MULT5] = 1.0  # multiples of 5 get the fives bonus
    return np.array(r, dtype=float), np.array(c, dtype=float), ends, exp_, sq, log_m, pr, five


RULES, RULE_COST, IS_END, IS_EXP, IS_SQ, LOG_FACTOR, PRIME_VS_ODD, IS_FIVE = _rules()
N_RULES = RULES.shape[0]
RULE_SIZES = RULES.sum(1)
NOISE = 0.1
MAX_N = 4

# Interval groups: (size, whole-rows flag, number of examples contained).
_iv = [(a, b) for a in range(1, N + 1) for b in range(a, N + 1)]
IV_A = np.array([a for a, _ in _iv])
IV_B = np.array([b for _, b in _iv])
IV_SIZE = IV_B - IV_A + 1
IV_ROW = ((IV_A % 10 == 1) & (IV_B % 10 == 0)).astype(int)
SR_KEY = (IV_SIZE - 1) * 2 + IV_ROW  # 0..199
SR_COUNT = np.bincount(SR_KEY, minlength=200).astype(float)
SR_SIZE = np.repeat(np.arange(1, N + 1), 2).astype(float)
SR_ROW = np.tile([0.0, 1.0], N)
SR_LOGCOUNT = np.where(SR_COUNT > 0, np.log(np.maximum(SR_COUNT, 1.0)), -1e10)
G = 200 * (MAX_N + 1)
G_SIZE = np.repeat(SR_SIZE, MAX_N + 1)
G_ROW = np.repeat(SR_ROW, MAX_N + 1)
G_C = np.tile(np.arange(MAX_N + 1), 200).astype(float)
G_SR = np.repeat(np.arange(200), MAX_N + 1)


def prepare_observed(rows):
    sets, idx, set_i, tgt = [], {}, [], []
    for r in rows:
        s = str(r["number_set"]).strip()
        if s not in idx:
            idx[s] = len(sets)
            sets.append([int(x) for x in s.split(",")])
        set_i.append(idx[s])
        tgt.append(int(float(r["target"])))
    S = len(sets)
    rule_ll = np.zeros((S, N_RULES))
    n_ex = np.zeros(S)
    cnt_set = np.zeros((S, G))
    iv_c = []
    for i, ex in enumerate(sets):
        for x in ex:
            pin = RULES[:, x - 1] * (1 - NOISE) / RULE_SIZES
            rule_ll[i] += np.log(pin + NOISE / N)
        n_ex[i] = len(ex)
        c = sum(((IV_A <= x) & (IV_B >= x)).astype(int) for x in ex)
        iv_c.append(c)
        cnt_set[i] = np.bincount(SR_KEY * (MAX_N + 1) + c, minlength=G)
    pair_idx, pairs, pidx = [], [], {}
    for s, t in zip(set_i, tgt):
        if (s, t) not in pidx:
            pidx[(s, t)] = len(pairs)
            pairs.append((s, t))
        pair_idx.append(pidx[(s, t)])
    cnt_pair = np.zeros((len(pairs), G))
    for j, (s, t) in enumerate(pairs):
        inside = (IV_A <= t) & (IV_B >= t)
        cnt_pair[j] = np.bincount(SR_KEY[inside] * (MAX_N + 1) + iv_c[s][inside], minlength=G)
    pair_set = np.array([s for s, _ in pairs], dtype="int64")
    pair_tgt = np.array([t for _, t in pairs])
    set_mask = np.where(cnt_set > 0, 0.0, -1e10)
    return {
        "rule_loglik": rule_ll,
        "n_examples": n_ex,
        "group_count_set": cnt_set,
        "group_mask_set": set_mask,
        "group_count_pair": cnt_pair,
        "rule_member_pair": RULES[:, pair_tgt - 1].T.copy(),
        "pair_set": pair_set,
        "pair_idx": np.array(pair_idx, dtype="int64"),
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    rule_loglik = pm.Data("rule_loglik", np.zeros((1, N_RULES)))
    n_examples = pm.Data("n_examples", np.ones(1))
    group_count_set = pm.Data("group_count_set", np.zeros((1, G)))
    group_mask_set = pm.Data("group_mask_set", np.zeros((1, G)))
    group_count_pair = pm.Data("group_count_pair", np.zeros((1, G)))
    rule_member_pair = pm.Data("rule_member_pair", np.zeros((1, N_RULES)))
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))
    pair_idx = pm.Data("pair_idx", np.zeros(1, dtype="int64"))

    lam = pm.Beta("lambda_rules", 2.0, 2.0)  # prior mass on rules vs intervals
    cost = pm.HalfNormal("primitive_cost", sigma=5.0)  # log-prior cost per extra primitive
    log_beta = pm.Normal("log_beta", 0.5, 1.0)  # commitment exponent (log scale)
    beta = pm.Deterministic("beta", pt.exp(log_beta))
    lapse = pm.Beta("lapse", 1.0, 10.0)
    end_bonus = pm.Normal("final_digit_bonus", 0.0, 2.0)
    exp_cost = pm.Normal("exponent_cost", 0.0, 3.0)
    sq_cost = pm.Normal("square_cost", 0.0, 3.0)
    mult_cost = pm.Normal("mult_cost", 0.0, 3.0)
    row_bonus = pm.Normal("row_bonus", 0.0, 5.0)
    prime_over_odd = pm.Normal("prime_over_odd", 0.0, 3.0)
    fives_bonus = pm.Normal("fives_bonus", 0.0, 2.0)  # the one change: log-prior bonus for "multiples of 5"
    log_sigma = pm.Normal("log_sigma", np.log(20.0), 1.0)
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))

    rule_logp = (-cost * (RULE_COST - 1.0) + end_bonus * IS_END - exp_cost * IS_EXP - sq_cost * IS_SQ
                 - mult_cost * LOG_FACTOR + prime_over_odd * PRIME_VS_ODD + fives_bonus * IS_FIVE)
    rule_logp = rule_logp - pt.logsumexp(rule_logp)
    sr_logp = np.log(SR_SIZE) - SR_SIZE / sigma + row_bonus * SR_ROW  # per interval of that size/row type
    int_norm = pt.logsumexp(sr_logp + SR_LOGCOUNT)
    g_prior = sr_logp[G_SR] - int_norm  # per interval, by group
    g_ll_in = G_C * np.log((1 - NOISE) / G_SIZE + NOISE / N)  # S-independent part
    g_ll = g_ll_in[None, :] + (n_examples[:, None] - G_C[None, :]) * np.log(NOISE / N)

    lp_rule = beta * (rule_loglik + pt.log(lam) + rule_logp[None, :])  # S x R
    lp_grp = beta * (g_ll + pt.log1p(-lam) + g_prior[None, :]) + group_mask_set  # S x G
    m = pt.maximum(pt.max(lp_rule, axis=1), pt.max(lp_grp, axis=1))[:, None]
    e_rule = pt.exp(lp_rule - m)
    e_grp = pt.exp(lp_grp - m)
    z = pt.sum(e_rule, axis=1) + pt.sum(e_grp * group_count_set, axis=1)
    num = pt.sum(e_rule[pair_set] * rule_member_pair, axis=1) + pt.sum(e_grp[pair_set] * group_count_pair, axis=1)
    p_raw = (num / z[pair_set])[pair_idx]  # distinct (set, target) pairs -> trials
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(lapse * 0.5 + (1 - lapse) * p_raw, 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
