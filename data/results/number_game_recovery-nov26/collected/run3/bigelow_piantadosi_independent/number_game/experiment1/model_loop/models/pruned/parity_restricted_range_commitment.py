"""Sharpened concept selection with parity-restricted ranges.

Refinement of `sharpened_map_concept_selection`: people still commit, nearly
all-or-none, to the concept that best explains the examples under the size
principle (posterior raised to a power beta), but the concepts they can commit
to include parity-restricted ranges ("even numbers between 16 and 64") alongside
mathematical rules and plain intervals. The one change is adding these
parity-by-interval conjunctions (each distinct one counted once), which take a
fitted share of the interval prior.
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
# Parity-restricted ranges: all odd (or all even) numbers from a to b, a and b of that parity.
_par = [(a, b) for s in (1, 2) for a in range(s, N + 1, 2) for b in range(a, N + 1, 2)]
PAR_IV = np.array([(NUMS >= a) & (NUMS <= b) & (NUMS % 2 == a % 2) for a, b in _par], dtype=float)
MEMBER = np.vstack([RULES, INTERVALS, PAR_IV])  # H x 100
N_RULES, N_INT, N_PAR = RULES.shape[0], INTERVALS.shape[0], PAR_IV.shape[0]
IS_RULE = np.r_[np.ones(N_RULES), np.zeros(N_INT + N_PAR)]
IS_INT = np.r_[np.zeros(N_RULES), np.ones(N_INT), np.zeros(N_PAR)]
IS_PAR = np.r_[np.zeros(N_RULES + N_INT), np.ones(N_PAR)]
SIZES = MEMBER.sum(1)
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

    lam = pm.Beta("lambda_rules", 2.0, 2.0)  # prior mass on rules vs ranges
    rho = pm.Beta("rho_parity", 2.0, 2.0)  # share of range mass on parity-restricted ranges
    log_beta = pm.Normal("log_beta", 0.5, 1.0)  # sharpening exponent (log scale)
    beta = pm.Deterministic("beta", pt.exp(log_beta))
    lapse = pm.Beta("lapse", 1.0, 10.0)

    log_prior = (
        IS_RULE * (pt.log(lam) - np.log(N_RULES))
        + IS_INT * (pt.log1p(-lam) + pt.log1p(-rho) - np.log(N_INT))
        + IS_PAR * (pt.log1p(-lam) + pt.log(rho) - np.log(N_PAR))
    )
    logpost = beta * (set_loglik + log_prior[None, :])
    w = pt.special.softmax(logpost, axis=1)  # S x H
    p_in = pt.dot(w, pt.as_tensor_variable(MEMBER))  # S x 100
    p_raw = p_in[set_idx, target_idx]
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(lapse * 0.5 + (1 - lapse) * p_raw, 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
