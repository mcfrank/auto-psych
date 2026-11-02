"""Ascending range sampling.

People think of a range-type program as counting upward from where its range
starts, so they expect the examples to sit toward the bottom of the range,
with the rest of the range above them. A range starting just below the
examples and running well above them explains them better than a range centred
on them, so people generalise more to numbers a little above the examples than
to numbers equally far below. Rule concepts are judged as in
multiplicative_not_exponential_concepts (complexity-costed rules, extra cost
on powers/squares/cubes, Erlang interval prior, near-all-or-none commitment).

The one change: inside an interval [a, b] an example at relative position
u = (x - a) / (b - a + 1) is sampled with probability proportional to
exp(-k u), with a fitted gradient k >= 0 (k = 0 is uniform sampling).
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
_iv = np.array([(a, b) for a in range(1, N + 1) for b in range(a, N + 1)])
IV_A, IV_B = _iv[:, 0], _iv[:, 1]
INTERVALS = np.array([(NUMS >= a) & (NUMS <= b) for a, b in _iv], dtype=float)
MEMBER = np.vstack([RULES, INTERVALS])  # H x 100
INT_SIZES = INTERVALS.sum(1)
LOG_INT_SIZES = np.log(INT_SIZES)
# Relative positions j / L of every slot j = 0..L-1 of each interval (for the normaliser).
SLOT_U = np.arange(N)[None, :] / INT_SIZES[:, None]  # I x 100
SLOT_MASK = (np.arange(N)[None, :] < INT_SIZES[:, None]).astype(float)
NUM_U = np.where(INTERVALS > 0, (NUMS[None, :] - IV_A[:, None]) / INT_SIZES[:, None], 0.0)
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
    rule_ll = np.zeros((len(sets), RULES.shape[0]))
    counts = np.zeros((len(sets), N))  # how often each number appears among a set's examples
    rsize = RULES.sum(1)
    for i, ex in enumerate(sets):
        for x in ex:
            rule_ll[i] += np.log(RULES[:, x - 1] * (1 - NOISE) / rsize + NOISE / N)
            counts[i, x - 1] += 1.0
    return {
        "rule_loglik": rule_ll,
        "ex_counts": counts,
        "set_idx": np.array(set_i, dtype="int64"),
        "target_idx": np.array(tgt, dtype="int64"),
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    rule_loglik = pm.Data("rule_loglik", np.zeros((1, RULES.shape[0])))
    ex_counts = pm.Data("ex_counts", np.zeros((1, N)))
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))

    lam = pm.Beta("lambda_rules", 2.0, 2.0)  # prior mass on rules vs intervals
    cost = pm.HalfNormal("primitive_cost", sigma=5.0)  # log-prior cost per extra primitive
    exp_cost = pm.Normal("exponent_cost", 0.0, 3.0)  # extra cost of power/square/cube concepts
    log_beta = pm.Normal("log_beta", 0.5, 1.0)  # commitment exponent (log scale)
    beta = pm.Deterministic("beta", pt.exp(log_beta))
    lapse = pm.Beta("lapse", 1.0, 10.0)
    log_sigma = pm.Normal("log_sigma", np.log(20.0), 1.0)  # Erlang scale of interval size
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))
    k = pm.HalfNormal("ascent_gradient", sigma=2.0)  # bottom-heavy sampling within a range

    # Interval likelihood with ascending (bottom-heavy) sampling of examples.
    log_norm = pt.log(pt.sum(SLOT_MASK * pt.exp(-k * SLOT_U), axis=1))  # I
    # log p(number | interval): position u of each number in each interval.
    p_num = INTERVALS * (1 - NOISE) * pt.exp(-k * NUM_U - log_norm[:, None]) + NOISE / N  # I x 100
    int_loglik = pt.dot(ex_counts, pt.log(p_num).T)  # S x I
    set_loglik = pt.concatenate([rule_loglik, int_loglik], axis=1)

    rule_logp = -cost * (RULE_COST - 1.0) - exp_cost * IS_EXP
    rule_logp = rule_logp - pt.logsumexp(rule_logp)
    int_logp = LOG_INT_SIZES - INT_SIZES / sigma
    int_logp = int_logp - pt.logsumexp(int_logp)
    log_prior = pt.concatenate([pt.log(lam) + rule_logp, pt.log1p(-lam) + int_logp])
    w = pt.special.softmax(beta * (set_loglik + log_prior[None, :]), axis=1)  # S x H
    p_in = pt.dot(w, pt.as_tensor_variable(MEMBER))  # S x 100
    p_raw = p_in[set_idx, target_idx]
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(lapse * 0.5 + (1 - lapse) * p_raw, 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
