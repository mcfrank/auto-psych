"""Weakened size principle (partial strong sampling).

People do Bayesian concept learning over mathematical rules and numerical
intervals, but apply the size principle only partially: the likelihood that a
concept generated an example scales as size^(-gamma) instead of size^(-1), with
gamma fitted (gamma = 1 is ideal strong sampling, gamma = 0 is weak sampling).
Discounting the size penalty keeps broad concepts (odd, even, wide intervals)
alive, so people generalise along parity and across regions more than the
ideal learner. "Yes" is the posterior probability the concept contains the
target, with a small response lapse.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N = 100
NUMS = np.arange(1, N + 1)


def _rules():
    r = [NUMS % 2 == 1, NUMS % 2 == 0]
    primes = np.array([n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1)) for n in NUMS])
    r.append(primes)
    r.append(np.isin(NUMS, [k * k for k in range(1, 11)]))
    r.append(np.isin(NUMS, [k ** 3 for k in range(1, 5)]))
    for b in range(2, 11):
        r.append(np.isin(NUMS, [b ** k for k in range(0, 8) if b ** k <= N]))
    p2 = [2 ** k for k in range(0, 7)]
    r.append(np.isin(NUMS, [x for x in p2 if x != 32]))
    r.append(np.isin(NUMS, p2 + [37]))
    for m in list(range(3, 10)) + [11, 12]:
        r.append(NUMS % m == 0)
    for d in range(10):
        r.append(NUMS % 10 == d)
    for d in range(10):
        r.append(np.array([str(d) in str(n) for n in NUMS]))
    return np.array(r, dtype=float)


RULES = _rules()
_iv = [(a, b) for a in range(1, N + 1) for b in range(a, N + 1)]
INTERVALS = np.array([(NUMS >= a) & (NUMS <= b) for a, b in _iv], dtype=float)
MEMBER = np.vstack([RULES, INTERVALS])  # H x 100
N_RULES = RULES.shape[0]
N_INT = INTERVALS.shape[0]
IS_RULE = np.r_[np.ones(N_RULES), np.zeros(N_INT)]
LOG_REL_SIZE = np.log(MEMBER.sum(1) / N)  # log(|h| / N) <= 0
NOISE = 0.1


def prepare_observed(rows):
    sets, idx, set_i, tgt = [], {}, [], []
    for r in rows:
        s = str(r["number_set"]).strip()
        if s not in idx:
            idx[s] = len(sets)
            sets.append([int(x) for x in s.split(",")])
        set_i.append(idx[s])
        tgt.append(int(float(r["target"])) - 1)
    # Per set: how many examples each concept contains, and the set size.
    n_in = np.zeros((len(sets), MEMBER.shape[0]))
    n_ex = np.zeros(len(sets))
    for i, ex in enumerate(sets):
        for x in ex:
            n_in[i] += MEMBER[:, x - 1]
        n_ex[i] = len(ex)
    return {
        "n_in": n_in,
        "n_ex": n_ex,
        "set_idx": np.array(set_i, dtype="int64"),
        "target_idx": np.array(tgt, dtype="int64"),
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    n_in = pm.Data("n_in", np.zeros((1, MEMBER.shape[0])))
    n_ex = pm.Data("n_ex", np.zeros(1))
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))

    lam = pm.Beta("lambda_rules", 2.0, 2.0)  # prior mass on rules vs intervals
    gamma = pm.Beta("gamma", 2.0, 2.0)  # strength of the size principle
    lapse = pm.Beta("lapse", 1.0, 10.0)

    # p(x | h) = (1-NOISE) * [x in h] * |h|^-gamma * N^(gamma-1) + NOISE / N
    # (gamma = 1: strong sampling; gamma = 0: weak sampling, 1/N for members)
    in_scale = pt.exp(-gamma * pt.as_tensor_variable(LOG_REL_SIZE)) / N  # H
    log_p_member = pt.log((1 - NOISE) * in_scale + NOISE / N)  # H
    loglik = n_in * log_p_member[None, :] + (n_ex[:, None] - n_in) * np.log(NOISE / N)

    log_prior = IS_RULE * (pt.log(lam) - np.log(N_RULES)) + (1 - IS_RULE) * (
        pt.log1p(-lam) - np.log(N_INT)
    )
    w = pt.special.softmax(loglik + log_prior[None, :], axis=1)  # S x H
    p_in = pt.dot(w, pt.as_tensor_variable(MEMBER))  # S x 100
    p_raw = p_in[set_idx, target_idx]
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(lapse * 0.5 + (1 - lapse) * p_raw, 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
