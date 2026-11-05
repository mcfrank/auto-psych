"""Sample-size-neglect Bayesian concept learning.

People infer the hidden program as a Bayesian concept learner would: rules
(odd/even, primes, squares, cubes, powers of 2-10, multiples of 3-50, ends in
digit d) and magnitude intervals (Erlang width prior) are weighed by the size
principle with a uniform-noise sampling term, and the posterior probability
that the target belongs drives a graded (logistic) yes/no readout. The one
distortion is sample-size neglect: the total log-likelihood of a set of n
examples is down-weighted by n^(-rho), so evidence grows much more slowly with
the number of examples than normative inference says (rho = 0 is normative,
rho = 1 treats any set as worth a single example).
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMS = np.arange(1, 101)


def _isprime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def _rules():
    r = {"odd": NUMS % 2 == 1, "even": NUMS % 2 == 0,
         "prime": np.array([_isprime(n) for n in NUMS]),
         "square": np.isin(NUMS, [k * k for k in range(1, 11)]),
         "cube": np.isin(NUMS, [1, 8, 27, 64])}
    for b in range(2, 11):
        r[f"pow{b}"] = np.isin(NUMS, [b ** i for i in range(1, 8)])
    for k in range(3, 51):
        r[f"mult{k}"] = NUMS % k == 0
    for d in range(10):
        r[f"end{d}"] = NUMS % 10 == d
    return r


_R = _rules()
IS_DIGIT = np.array([n.startswith("end") for n in _R])
RULE_M = np.array(list(_R.values()), dtype="float64")
N_RULES = RULE_M.shape[0]
_IV = [(a, b) for a in range(1, 101) for b in range(a, 101)]
INT_M = np.zeros((len(_IV), 100))
for _i, (_a, _b) in enumerate(_IV):
    INT_M[_i, _a - 1:_b] = 1.0
WIDTH = INT_M.sum(axis=1)
HYP_M = np.vstack([RULE_M, INT_M])
LOG_SIZE = np.log(HYP_M.sum(axis=1))
N_DIG = float(IS_DIGIT.sum())
N_MATH = float(N_RULES - IS_DIGIT.sum())


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    uniq = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(uniq)}
    n_in = np.stack([HYP_M[:, np.array(s) - 1].sum(axis=1) for s in uniq])
    n_out = np.array([len(s) for s in uniq], dtype="float64")[:, None] - n_in
    return {
        "n_in": n_in,
        "n_out": n_out,
        "evidence_weight_log_n": np.log(np.array([len(s) for s in uniq], dtype="float64")),
        "row_cell": np.array([index[s] * 100 + int(r["target"]) - 1 for s, r in zip(sets, rows)], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    n_in = pm.Data("n_in", np.zeros((1, HYP_M.shape[0])))
    n_out = pm.Data("n_out", np.zeros((1, HYP_M.shape[0])))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    log_n = pm.Data("evidence_weight_log_n", np.zeros(1))

    # Normative learner: prior on rules vs intervals, digit vs arithmetic rules,
    # interval width scale, sampling noise.
    rule_share = pm.Beta("rule_share", 2.0, 2.0)
    digit_share = pm.Beta("digit_share", 2.0, 2.0)
    log_width_scale = pm.Normal("log_width_scale", mu=2.5, sigma=1.0)
    noise = pm.Beta("noise", 2.0, 5.0)
    # The distortion: sample-size neglect (0 = normative, 1 = n examples count as one).
    rho = pm.Beta("rho", 2.0, 2.0)
    # Graded decision readout of the posterior probability.
    gamma = pm.HalfNormal("gamma", sigma=1.0)
    bias = pm.Normal("bias", mu=0.0, sigma=1.5)

    log_noise = pt.log(noise) - np.log(100.0)
    log_in = pt.logaddexp(pt.log1p(-noise) - LOG_SIZE, log_noise)
    log_lik = (n_in * log_in[None, :] + n_out * log_noise) * pt.exp(-rho * log_n)[:, None]

    lp_int = np.log(WIDTH) - WIDTH / pt.exp(log_width_scale)
    lp_int = lp_int - pt.logsumexp(lp_int)
    lp_rule = pt.switch(IS_DIGIT, pt.log(digit_share) - np.log(N_DIG),
                        pt.log1p(-digit_share) - np.log(N_MATH))
    log_prior = pt.concatenate([pt.log(rule_share) + lp_rule, pt.log1p(-rule_share) + lp_int])

    log_joint = log_lik + log_prior[None, :]
    post = pt.exp(log_joint - pt.logsumexp(log_joint, axis=1, keepdims=True))
    p_in = pt.dot(post, pt.as_tensor(HYP_M)).reshape((-1,))[row_cell]
    p_in = pt.clip(p_in, 1e-9, 1 - 1e-9)
    perceived = bias + gamma * (pt.log(p_in) - pt.log1p(-p_in))

    p_yes = pm.Deterministic("p_yes", pt.clip(pm.math.sigmoid(perceived), 1e-6, 1 - 1e-6))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
