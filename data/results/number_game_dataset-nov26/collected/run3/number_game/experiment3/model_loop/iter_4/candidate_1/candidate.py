"""Pairwise consequential-region exemplar model.

People do not infer one concept that explains all the examples together; they
store each example as a separate exemplar and judge the target by how similar it
is to each stored example on its own, where two numbers are similar to the
extent that familiar number concepts (mathematical and digit rules, magnitude
ranges), weighted toward small concepts, would contain both (Shepard's
consequential-region similarity). "Yes" follows the target's average similarity
to the stored examples.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N = np.arange(1, 101)


def _isprime(n):
    return n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1))


def _rules():
    props = [N % 2 == 1, N % 2 == 0, np.array([_isprime(n) for n in N]),
             np.isin(N, [k * k for k in range(1, 11)]),
             np.isin(N, [k ** 3 for k in range(1, 5)])]
    for b in range(2, 11):
        props.append(np.isin(N, [b ** k for k in range(1, 8)]))
    props.append(np.isin(N, [2, 4, 8, 16, 64]))
    props.append(np.isin(N, [2, 4, 8, 16, 32, 37, 64]))
    for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]:
        props.append(N % m == 0)
    for d in range(10):
        props.append(N % 10 == d)
    for d in range(10):
        props.append(np.array([str(d) in str(n) for n in N]))
    # families: parity, special numbers, multiples, ends-in, contains
    fam = np.array([0, 0] + [1] * 14 + [2] * 9 + [3] * 10 + [4] * 10)
    return np.array(props, dtype="float64"), fam


RULES, FAM = _rules()                      # (45, 100)
RSIZE = RULES.sum(1)
NFAM = 5
FAM_ONEHOT = np.eye(NFAM)[FAM]             # (45, 5)
LEN = np.arange(1, 101, dtype="float64")
# number of intervals of length L containing both e and t: (100*100, 100)
_e = N[:, None, None]
_t = N[None, :, None]
_L = np.arange(1, 101)[None, None, :]
_lo = np.maximum(np.maximum(_e, _t) - _L + 1, 1)
_hi = np.minimum(np.minimum(_e, _t), 100 - _L + 1)
PAIR_COUNT = np.maximum(_hi - _lo + 1, 0).astype("float64").reshape(10000, 100)
N_INTERVALS = 101.0 - LEN


def prepare_observed(rows):
    T = len(rows)
    ex_idx = np.zeros((T, 4), dtype="int64")
    ex_mask = np.zeros((T, 4), dtype="float64")
    tgt = np.zeros(T, dtype="int64")
    rating = np.zeros(T, dtype="int64")
    for i, r in enumerate(rows):
        ex = [int(x) for x in str(r["number_set"]).split(",")]
        t = int(r["target"])
        for j, x in enumerate(ex[:4]):
            ex_idx[i, j] = (x - 1) * 100 + (t - 1)
            ex_mask[i, j] = 1.0
        tgt[i] = t - 1
        rating[i] = int(float(r["rating"]))
    return {"pair_idx": ex_idx, "pair_mask": ex_mask, "rating": rating}


with pm.Model() as model:
    pair_idx = pm.Data("pair_idx", np.zeros((1, 4), dtype="int64"))
    pair_mask = pm.Data("pair_mask", np.ones((1, 4), dtype="float64"))

    # share of concept prior on rules vs magnitude ranges
    rule_logit = pm.Normal("rule_logit", mu=0.0, sigma=2.0)
    # strength of the size principle on concepts
    log_tau = pm.Normal("log_tau", mu=0.0, sigma=1.0)
    # prior preference for wide (>0) vs narrow (<0) ranges
    range_size_power = pm.Normal("range_size_power", mu=0.0, sigma=1.0)
    # family weights for rules (parity family fixed at 0)
    fam_raw = pm.Normal("fam_weight", mu=0.0, sigma=2.0, shape=NFAM - 1)
    # readout
    criterion = pm.Normal("criterion", mu=0.0, sigma=2.0)
    log_slope = pm.Normal("log_slope", mu=0.0, sigma=1.0)

    tau = pt.exp(log_tau)
    a = pm.math.sigmoid(rule_logit)
    fam_w = pt.concatenate([pt.zeros(1), fam_raw])
    lw_rule = pt.dot(FAM_ONEHOT, fam_w)
    w_rule = pt.exp(lw_rule - pt.logsumexp(lw_rule))       # (45,)
    rule_weight = a * w_rule * RSIZE ** (-tau)              # prior * size principle
    joint_rule = pt.dot(RULES.T * rule_weight[None, :], RULES)  # (100,100)

    lw_len = range_size_power * np.log(LEN)
    lw_len = lw_len - pt.logsumexp(lw_len + np.log(N_INTERVALS))
    len_weight = (1.0 - a) * pt.exp(lw_len - tau * np.log(LEN))
    joint_int = pt.dot(PAIR_COUNT, len_weight).reshape((100, 100))

    joint = joint_rule + joint_int
    sim = joint / pt.diag(joint)[:, None]                   # P(t in C | e in C)
    sim_flat = pt.clip(sim.reshape((10000,)), 1e-9, 1.0)

    s = sim_flat[pair_idx]                                  # (T, 4)
    mean_sim = (s * pair_mask).sum(1) / pair_mask.sum(1)
    z = criterion + pt.exp(log_slope) * pt.log(mean_sim)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(z))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
