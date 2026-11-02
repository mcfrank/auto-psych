"""Serial-position-weighted concept search.

People search for the simplest concept that explains the examples
(complexity-costed rules and intervals, size principle, near-all-or-none
commitment), but they do not weigh the examples equally: the evidence each
example contributes depends on its position in the order shown, growing or
shrinking steadily from the first example to the last at a fitted rate
(primacy if the rate is negative, recency if positive). The one change from
simplicity_cost_concept_search is this per-position evidence weight; weights
are normalised to sum to the number of examples, so a rate of 0 recovers it.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N = 100
NUMS = np.arange(1, N + 1)
MAX_EX = 4


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
    for d in range(10):
        r.append(NUMS % 10 == d)
        c.append(2)
    for d in range(10):
        r.append(np.array([str(d) in str(n) for n in NUMS]))
        c.append(2)
    return np.array(r, dtype=float), np.array(c, dtype=float)


RULES, RULE_COST = _rules()
_iv = [(a, b) for a in range(1, N + 1) for b in range(a, N + 1)]
INTERVALS = np.array([(NUMS >= a) & (NUMS <= b) for a, b in _iv], dtype=float)
MEMBER = np.vstack([RULES, INTERVALS])  # H x 100
N_RULES = RULES.shape[0]
N_INT = INTERVALS.shape[0]
SIZES = MEMBER.sum(1)
NOISE = 0.1
H = MEMBER.shape[0]


def prepare_observed(rows):
    sets, idx, set_i, tgt = [], {}, [], []
    for r in rows:
        s = str(r["number_set"]).strip()
        if s not in idx:
            idx[s] = len(sets)
            sets.append([int(x) for x in s.split(",")])
        set_i.append(idx[s])
        tgt.append(int(float(r["target"])) - 1)
    # Per-position log-likelihood of each example (zero-padded) and a mask.
    pos_ll = np.zeros((len(sets), MAX_EX, H))
    mask = np.zeros((len(sets), MAX_EX))
    for i, ex in enumerate(sets):
        for j, x in enumerate(ex[:MAX_EX]):
            pin = MEMBER[:, x - 1] * (1 - NOISE) / SIZES
            pos_ll[i, j] = np.log(pin + NOISE / N)
            mask[i, j] = 1.0
    return {
        "pos_loglik": pos_ll,
        "pos_mask": mask,
        "set_idx": np.array(set_i, dtype="int64"),
        "target_idx": np.array(tgt, dtype="int64"),
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    pos_loglik = pm.Data("pos_loglik", np.zeros((1, MAX_EX, H)))
    pos_mask = pm.Data("pos_mask", np.ones((1, MAX_EX)))
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))

    lam = pm.Beta("lambda_rules", 2.0, 2.0)
    cost = pm.HalfNormal("primitive_cost", sigma=5.0)
    log_beta = pm.Normal("log_beta", 0.5, 1.0)
    beta = pm.Deterministic("beta", pt.exp(log_beta))
    lapse = pm.Beta("lapse", 1.0, 10.0)
    # Serial-position rate: <0 primacy, >0 recency, 0 equal weighting.
    rate = pm.Normal("position_rate", 0.0, 1.0)

    pos = pt.as_tensor_variable(np.arange(MAX_EX, dtype=float))
    raw_w = pt.exp(rate * pos)[None, :] * pos_mask  # S x 4
    n_ex = pos_mask.sum(axis=1, keepdims=True)
    wts = n_ex * raw_w / raw_w.sum(axis=1, keepdims=True)
    set_loglik = (wts[:, :, None] * pos_loglik).sum(axis=1)  # S x H

    rule_logp = -cost * (RULE_COST - 1.0)
    rule_logp = rule_logp - pt.logsumexp(rule_logp)
    log_prior = pt.concatenate(
        [pt.log(lam) + rule_logp, pt.log1p(-lam) - np.log(N_INT) * pt.ones(N_INT)]
    )
    logpost = beta * (set_loglik + log_prior[None, :])
    w = pt.special.softmax(logpost, axis=1)
    p_in = pt.dot(w, pt.as_tensor_variable(MEMBER))
    p_raw = p_in[set_idx, target_idx]
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(lapse * 0.5 + (1 - lapse) * p_raw, 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
