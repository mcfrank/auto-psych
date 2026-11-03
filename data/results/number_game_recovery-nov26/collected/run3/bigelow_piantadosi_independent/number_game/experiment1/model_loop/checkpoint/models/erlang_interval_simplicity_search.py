"""Complexity-costed concept search with an Erlang size prior on intervals.

Refinement of simplicity_cost_concept_search: the one change is that intervals
are no longer equally likely a priori; their prior follows an Erlang density on
interval size (log prior = log size - size / sigma, fitted sigma), as in
Tenenbaum's number game, so tight ranges around the examples are less natural
and rules win more often. Original hypothesis follows.

People search for a concept by building it from simple mental primitives, and
every extra primitive a concept needs costs them: the simplest concepts (odd,
even) come to mind readily, while compound concepts (multiples of 7, ending in
2, powers of 3, a rule with an exception) are penalised in proportion to how
many primitives describe them, on top of the size principle and the
near-all-or-none commitment to the best-explaining concept. The one change from
sharpened MAP concept selection is the prior on rules: log prior falls by a
fitted cost per description primitive instead of being uniform over rules.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N = 100
NUMS = np.arange(1, N + 1)


def _rules():
    # Each rule with its description length (number of primitives).
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
    log_sigma = pm.Normal("log_sigma", np.log(20.0), 1.0)  # Erlang scale of interval size
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))

    rule_logp = -cost * (RULE_COST - 1.0)
    rule_logp = rule_logp - pt.logsumexp(rule_logp)  # normalised within rules
    int_logp = LOG_INT_SIZES - INT_SIZES / sigma
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
