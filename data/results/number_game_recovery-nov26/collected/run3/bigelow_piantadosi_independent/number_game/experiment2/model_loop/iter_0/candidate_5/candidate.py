"""Final-digit multiplicative concepts with a default answer that drifts over the session.

Refinement of final_digit_multiplicative_concepts: people build number
concepts from mental primitives (a fitted cost per extra primitive, a fitted
extra cost on power/square/cube concepts, a fitted prior discount on "ends in
d" rules, Erlang-weighted intervals, the size principle, near-all-or-none
commitment). The one change is in the response stage: on a fitted fraction of
trials people answer without consulting the inferred concept, and the default
answer they then give drifts from leaning "yes" early in the session toward
leaning "no" late in it, at a fitted rate, so the yes rate falls over the
session beyond what the stimulus order implies.
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
IS_ENDS = np.zeros(len(RULE_COST))
IS_ENDS[-20:-10] = 1.0  # the ten "ends in d" rules
_iv = [(a, b) for a in range(1, N + 1) for b in range(a, N + 1)]
INTERVALS = np.array([(NUMS >= a) & (NUMS <= b) for a, b in _iv], dtype=float)
MEMBER = np.vstack([RULES, INTERVALS])  # H x 100
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
        "session_pos": np.array(
            [float(r["trial_index"]) / 450.0 for r in rows], dtype="float64"
        ),
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    set_loglik = pm.Data("set_loglik", np.zeros((1, MEMBER.shape[0])))
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))
    session_pos = pm.Data("session_pos", np.zeros(1, dtype="float64"))

    lam = pm.Beta("lambda_rules", 2.0, 2.0)  # prior mass on rules vs intervals
    cost = pm.HalfNormal("primitive_cost", sigma=5.0)  # log-prior cost per extra primitive
    exp_cost = pm.Normal("exponent_cost", 0.0, 3.0)  # extra log-prior cost of power/square/cube concepts
    end_bonus = pm.Normal("ending_bonus", 0.0, 3.0)  # log-prior discount on "ends in d" rules
    log_beta = pm.Normal("log_beta", 0.5, 1.0)  # commitment exponent (log scale)
    beta = pm.Deterministic("beta", pt.exp(log_beta))
    lapse = pm.Beta("lapse", 2.0, 5.0)  # fraction of answers given without the concept
    # default answer when not consulting the concept: logit-linear in session
    # position (0 at the start, ~1 at the end)
    guess_start = pm.Normal("guess_start", 0.0, 1.5)
    guess_slope = pm.Normal("guess_slope", 0.0, 2.0)
    log_sigma = pm.Normal("log_sigma", np.log(20.0), 1.0)  # Erlang scale of interval size
    sigma = pm.Deterministic("sigma", pt.exp(log_sigma))

    rule_logp = -cost * (RULE_COST - 1.0) - exp_cost * IS_EXP + end_bonus * IS_ENDS
    rule_logp = rule_logp - pt.logsumexp(rule_logp)  # normalised within rules
    int_logp = LOG_INT_SIZES - INT_SIZES / sigma
    int_logp = int_logp - pt.logsumexp(int_logp)  # normalised within intervals
    log_prior = pt.concatenate([pt.log(lam) + rule_logp, pt.log1p(-lam) + int_logp])
    logpost = beta * (set_loglik + log_prior[None, :])
    w = pt.special.softmax(logpost, axis=1)  # S x H
    p_in = pt.dot(w, pt.as_tensor_variable(MEMBER))  # S x 100
    p_raw = p_in[set_idx, target_idx]
    guess = pm.math.sigmoid(guess_start + guess_slope * session_pos)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(lapse * guess + (1 - lapse) * p_raw, 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
