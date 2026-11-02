"""Pedagogical example selection on the forward-continuation learner.

People do not treat the examples as random draws from the program's output:
they assume a helpful teacher picked them, and that the teacher favours
diagnostic numbers -- ones that few candidate stretches of the number line
contain, i.e. numbers near the ends of 1-100. So a concept with many such
telling members is judged unlikely when none of them was shown ("if the
program reached the 90s, I would have been shown one"). Formally a concept's
effective size is the sum of its members' distinctiveness raised to a fitted
strength (0 = the plain size principle). Everything else is
forward_continuation_response_bias, unchanged: primitive-built rules with
costs, final-digit discount, power and square costs, graded times tables,
prime bonus and odd cost, repdigits, privileged fives, hundred-chart ranges,
set-size-graded evidence, near-all-or-none commitment, the round-number veto
and the upward response lean.
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
    pr[2] = 1.0  # prime numbers: their own bonus
    od = np.zeros(len(r))
    od[0] = 1.0  # odd numbers: their own cost
    fv = np.zeros(len(r))
    fv[18] = 1.0  # multiples of 5: their own bonus
    r.append(np.isin(NUMS, [11 * k for k in range(1, 10)]))  # repdigits
    c.append(2)
    ends, exp_, sq, log_m, pr, od, fv = (np.append(a, 0.0) for a in (ends, exp_, sq, log_m, pr, od, fv))
    rep = np.zeros(len(r))
    rep[-1] = 1.0
    return np.array(r, dtype=float), np.array(c, dtype=float), ends, exp_, sq, log_m, pr, od, rep, fv


RULES, RULE_COST, IS_END, IS_EXP, IS_SQ, LOG_FACTOR, IS_PRIME, IS_ODD, IS_REP, IS_FIVES = _rules()
assert RULES[18, 4] == 1 and RULES[18].sum() == 20  # row 18 is multiples of 5
N_RULES = RULES.shape[0]
RULE_SIZES = RULES.sum(1)
NOISE = 0.1
MAX_N = 4

# All intervals [a, b] laid out on a 100 x 100 grid (row a, column b; a <= b valid).
_A = np.arange(1, N + 1)[:, None] * np.ones((1, N), dtype=int)
_B = np.arange(1, N + 1)[None, :] * np.ones((N, 1), dtype=int)
IV_VALID = (_A <= _B)
IV_MASK = np.where(IV_VALID, 0.0, -1e10)
IV_SIZE = np.maximum(_B - _A + 1, 1).astype(float)
IV_ROW = ((_A % 10 == 1) & (_B % 10 == 0) & IV_VALID).astype(float)


def _distinctiveness():
    """How diagnostic each number is: the inverse of how much size-weighted
    interval mass contains it (numbers near 1 or 100 lie in few intervals).
    Returned as standardised log values."""
    a, b = _A[IV_VALID], _B[IV_VALID]
    d_iv = np.array([np.sum(((a <= x) & (b >= x)) / (b - a + 1.0)) for x in NUMS]) / len(a)
    lw = -np.log(d_iv)
    return (lw - lw.mean()) / lw.std()


LOG_DISTINCT = _distinctiveness()


def _blocks(ex, t=None):
    """Rectangles of (start a, end b) over which an interval holds a fixed number
    k of the examples, clipped to intervals containing target t if given.
    Each is (k, a_lo, a_hi, b_lo, b_hi), 1-based inclusive."""
    u = sorted(set(ex))
    a_rng, b_rng = [], []
    lo = 1
    for j, v in enumerate(u):  # a in [lo, v]: examples strictly below a
        a_rng.append((sum(x < lo for x in ex), lo, v))
        lo = v + 1
    a_rng.append((len(ex), lo, N))
    b_rng.append((0, 1, u[0] - 1))
    for j, v in enumerate(u):  # b in [v, next - 1]: examples at or below b
        nxt = u[j + 1] - 1 if j + 1 < len(u) else N
        b_rng.append((sum(x <= v for x in ex), v, nxt))
    out = []
    for ca, a0, a1 in a_rng:
        for cb, b0, b1 in b_rng:
            hi_a, lo_b = (a1, b0) if t is None else (min(a1, t), max(b0, t))
            if cb >= ca and a0 <= hi_a and lo_b <= b1:
                out.append((cb - ca, a0, hi_a, lo_b, b1))
    return out


N_SLOT = 4 * (MAX_N + 1) ** 2


def _corners(n, blocks):
    """Per corner slot: flat index into the containment-sum table, its sign, and
    the (set size, count) table it reads. The table at [a, b] sums every
    interval [a', b'] with a' <= a and b' >= b, i.e. all intervals containing
    [a, b], so a rectangle's corners only add intervals no smaller than it."""
    idx, sgn, tab = np.zeros(N_SLOT, dtype="int64"), np.zeros(N_SLOT), np.zeros(N_SLOT, dtype="int64")
    i = 0
    for k, a0, a1, b0, b1 in blocks:
        t = (n - 1) * (MAX_N + 1) + k
        for aa, bb, sg in ((a1, b0, 1), (a0 - 1, b0, -1), (a1, b1 + 1, -1), (a0 - 1, b1 + 1, 1)):
            idx[i], sgn[i], tab[i] = (t * (N + 1) + aa) * (N + 2) + bb, sg, t
            i += 1
    return idx, sgn, tab


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
    ex_member = np.zeros((S, MAX_N, N_RULES))  # rule membership of each example
    ex_valid = np.zeros((S, MAX_N))
    n_ex = np.zeros(S, dtype="int64")
    set_corner = np.zeros((S, N_SLOT), dtype="int64")
    set_sign = np.zeros((S, N_SLOT))
    set_table = np.zeros((S, N_SLOT), dtype="int64")
    for i, ex in enumerate(sets):
        for k, x in enumerate(ex):
            ex_member[i, k] = RULES[:, x - 1]
            ex_valid[i, k] = 1.0
        n_ex[i] = len(ex)
        set_corner[i], set_sign[i], set_table[i] = _corners(len(ex), _blocks(ex))
    pair_idx, pairs, pidx = [], [], {}
    for s, t in zip(set_i, tgt):
        if (s, t) not in pidx:
            pidx[(s, t)] = len(pairs)
            pairs.append((s, t))
        pair_idx.append(pidx[(s, t)])
    pair_set = np.array([s for s, _ in pairs], dtype="int64")
    pair_tgt = np.array([t for _, t in pairs])
    pair_corner = np.zeros((len(pairs), N_SLOT), dtype="int64")
    pair_sign = np.zeros((len(pairs), N_SLOT))
    pair_table = np.zeros((len(pairs), N_SLOT), dtype="int64")
    for j, (s, t) in enumerate(pairs):
        pair_corner[j], pair_sign[j], pair_table[j] = _corners(len(sets[s]), _blocks(sets[s], t))
    unsupported_round = np.array([float(t % 10 == 0 and not any(e % 10 == 0 for e in sets[s])) for s, t in pairs])
    direction = np.array([float(t > max(sets[s])) - float(t < min(sets[s])) for s, t in pairs])
    return {
        "example_member": ex_member,
        "example_valid": ex_valid,
        "n_examples": n_ex,
        "set_corner": set_corner,
        "set_sign": set_sign,
        "set_table": set_table,
        "pair_corner": pair_corner,
        "pair_sign": pair_sign,
        "pair_table": pair_table,
        "rule_member_pair": RULES[:, pair_tgt - 1].T.copy(),
        "pair_set": pair_set,
        "unsupported_round_pair": unsupported_round,
        "direction_pair": direction,
        "pair_idx": np.array(pair_idx, dtype="int64"),
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    example_member = pm.Data("example_member", np.zeros((1, MAX_N, N_RULES)))
    example_valid = pm.Data("example_valid", np.zeros((1, MAX_N)))
    n_examples = pm.Data("n_examples", np.ones(1, dtype="int64"))
    set_corner = pm.Data("set_corner", np.zeros((1, N_SLOT), dtype="int64"))
    set_sign = pm.Data("set_sign", np.zeros((1, N_SLOT)))
    set_table = pm.Data("set_table", np.zeros((1, N_SLOT), dtype="int64"))
    pair_corner = pm.Data("pair_corner", np.zeros((1, N_SLOT), dtype="int64"))
    pair_sign = pm.Data("pair_sign", np.zeros((1, N_SLOT)))
    pair_table = pm.Data("pair_table", np.zeros((1, N_SLOT), dtype="int64"))
    rule_member_pair = pm.Data("rule_member_pair", np.zeros((1, N_RULES)))
    pair_set = pm.Data("pair_set", np.zeros(1, dtype="int64"))
    unsupported_round_pair = pm.Data("unsupported_round_pair", np.zeros(1))
    direction_pair = pm.Data("direction_pair", np.zeros(1))
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
    prime_bonus = pm.Normal("prime_bonus", 0.0, 3.0)
    odd_cost = pm.Normal("odd_cost", 0.0, 3.0)
    rep_bonus = pm.Normal("repdigit_bonus", 0.0, 2.0)
    fives_bonus = pm.Normal("fives_bonus", 0.0, 2.0)
    evidence_growth = pm.Normal("evidence_growth", 0.0, 1.0)
    ev_w_n = pt.exp(evidence_growth * np.log(np.arange(1, MAX_N + 1) / 3.0))  # by set size
    ev_w = ev_w_n[n_examples - 1]
    round_veto = pm.Beta("round_veto", 1.0, 4.0)
    log_sigma = pm.Normal("log_sigma", np.log(20.0), 1.0)
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))
    tilt = pm.Normal("forward_tilt", 0.0, 1.0)
    # The one change: the examples are taken to be chosen by a helpful teacher who
    # prefers diagnostic numbers, so a concept's effective size is the sum of its
    # members' distinctiveness raised to this power (0 = random sampling).
    alpha = pm.Normal("pedagogical_strength", 0.0, 1.0)
    w = pt.exp(alpha * LOG_DISTINCT)  # teacher's preference for each number
    z_all = pt.sum(w)
    z_rule = pt.dot(RULES, w)  # effective size of each rule
    cw = pt.concatenate([pt.zeros(1), pt.cumsum(w)])
    z_iv = pt.maximum(cw[1:][None, :] - cw[:-1][:, None], 1e-12)  # [a, b] effective size

    rule_logp = (-cost * (RULE_COST - 1.0) + end_bonus * IS_END - exp_cost * IS_EXP - sq_cost * IS_SQ
                 - mult_cost * LOG_FACTOR + prime_bonus * IS_PRIME - odd_cost * IS_ODD + rep_bonus * IS_REP + fives_bonus * IS_FIVES)
    rule_logp = rule_logp - pt.logsumexp(rule_logp)
    iv_logp = np.log(IV_SIZE) - IV_SIZE / sigma + row_bonus * IV_ROW + IV_MASK
    iv_logp = iv_logp - pt.logsumexp(iv_logp)

    stray = NOISE / z_all
    rule_ll = pt.sum(example_valid[:, :, None] * pt.log((1 - NOISE) * example_member / z_rule[None, None, :] + stray), axis=1)
    lp_rule = beta * (ev_w[:, None] * rule_ll + pt.log(lam) + rule_logp[None, :])  # S x R

    # Interval log-posterior for every set size n, number k of examples inside,
    # and interval [a, b]; a set picks rectangles of (a, b) with fixed k.
    nn = np.arange(1, MAX_N + 1)[:, None, None, None]
    kk = np.arange(MAX_N + 1)[None, :, None, None]
    K_MASK = np.where(kk <= nn, 0.0, -1e10)  # more examples inside than shown: impossible
    l_in = pt.log((1 - NOISE) / z_iv + stray)
    lp_q = beta * (ev_w_n[:, None, None, None] * (kk * l_in[None, None] + (nn - kk) * pt.log(stray))
                   + pt.log1p(-lam) + iv_logp[None, None]) + IV_MASK[None, None] + K_MASK
    shift = pt.max(lp_q, axis=(2, 3))  # per (set size, count) table
    q = pt.exp(lp_q - shift[:, :, None, None])
    cs = pt.cumsum(pt.cumsum(q, axis=2)[:, :, :, ::-1], axis=3)[:, :, :, ::-1]
    tab = pt.zeros((MAX_N, MAX_N + 1, N + 1, N + 2))
    tab = pt.set_subtensor(tab[:, :, 1:, 1:N + 1], cs).flatten()
    shift = shift.flatten()
    m_set = pt.max(lp_rule, axis=1)
    e_rule = pt.exp(lp_rule - m_set[:, None])

    def interval_mass(corner, sign, table, m):
        scale = pt.exp(pt.clip(shift[table] - m[:, None], -700.0, 700.0))
        return pt.sum(tab[corner] * sign * scale, axis=1)

    z = pt.sum(e_rule, axis=1) + interval_mass(set_corner, set_sign, set_table, m_set)
    num = (pt.sum(e_rule[pair_set] * rule_member_pair, axis=1)
           + interval_mass(pair_corner, pair_sign, pair_table, m_set[pair_set]))
    p_concept = num / z[pair_set]
    pc = pt.clip(p_concept * (1 - round_veto * unsupported_round_pair), 1e-6, 1 - 1e-6)
    p_raw = pm.math.sigmoid(pt.log(pc) - pt.log1p(-pc) + tilt * direction_pair)[pair_idx]
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(lapse * 0.5 + (1 - lapse) * p_raw, 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
