"""First-example-overridden concept search.

People revise their guess about the concept as each example arrives, so the
later examples carry the final say: the first example shown is the one most
readily dismissed as a stray the program did not really mean, while later
examples are trusted to belong to the concept. Concepts are otherwise built as
in costly_squares_final_digit_concepts (per-primitive cost, "ends in d"
discount, extra cost on powers/cubes and on squares, Erlang-weighted intervals,
size principle, near-all-or-none commitment). The one change: the stray rate is
position dependent, with a separate fitted stray rate for the first example and
for later examples (the claim: first > later), so the winning concept may
ignore a first example that breaks the later examples' pattern, and reordering
the same examples changes the answer.
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
    return np.array(r, dtype=float), np.array(c, dtype=float), ends, exp_, sq


RULES, RULE_COST, IS_END, IS_EXP, IS_SQ = _rules()
_iv = [(a, b) for a in range(1, N + 1) for b in range(a, N + 1)]
INTERVALS = np.array([(NUMS >= a) & (NUMS <= b) for a, b in _iv], dtype=float)
MEMBER = np.vstack([RULES, INTERVALS])  # H x 100
SIZES = MEMBER.sum(1)
INT_SIZES = INTERVALS.sum(1)
LOG_INT_SIZES = np.log(INT_SIZES)


def prepare_observed(rows):
    # Per distinct set: whether each concept contains the first example, and
    # how many of the later examples it contains.
    sets, idx, set_i, tgt = [], {}, [], []
    for r in rows:
        s = str(r["number_set"]).strip()
        if s not in idx:
            idx[s] = len(sets)
            sets.append([int(x) for x in s.split(",")])
        set_i.append(idx[s])
        tgt.append(int(float(r["target"])) - 1)
    first_in = np.zeros((len(sets), MEMBER.shape[0]))
    later_in = np.zeros((len(sets), MEMBER.shape[0]))
    n_later = np.zeros(len(sets))
    for i, ex in enumerate(sets):
        first_in[i] = MEMBER[:, ex[0] - 1]
        for x in ex[1:]:
            later_in[i] += MEMBER[:, x - 1]
        n_later[i] = len(ex) - 1
    return {
        "first_in": first_in,
        "later_in": later_in,
        "n_later": n_later,
        "set_idx": np.array(set_i, dtype="int64"),
        "target_idx": np.array(tgt, dtype="int64"),
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    first_in = pm.Data("first_in", np.zeros((1, MEMBER.shape[0])))
    later_in = pm.Data("later_in", np.zeros((1, MEMBER.shape[0])))
    n_later = pm.Data("n_later", np.zeros(1))
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
    log_sigma = pm.Normal("log_sigma", np.log(20.0), 1.0)
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))
    # Position-dependent stray rates: the first example vs. every later one.
    stray_first = pm.Beta("stray_first", 1.0, 9.0)
    stray_later = pm.Beta("stray_later", 1.0, 9.0)

    # Log-probability of one example under each concept, if it is (f) or is
    # not (g) a member; the strong-sampling term depends only on concept size.
    f1 = pt.log((1 - stray_first) / SIZES + stray_first / N)
    g1 = pt.log(stray_first / N)
    f2 = pt.log((1 - stray_later) / SIZES + stray_later / N)
    g2 = pt.log(stray_later / N)
    set_loglik = (
        first_in * f1[None, :] + (1 - first_in) * g1
        + later_in * f2[None, :] + (n_later[:, None] - later_in) * g2
    )

    rule_logp = -cost * (RULE_COST - 1.0) + end_bonus * IS_END - exp_cost * IS_EXP - sq_cost * IS_SQ
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
