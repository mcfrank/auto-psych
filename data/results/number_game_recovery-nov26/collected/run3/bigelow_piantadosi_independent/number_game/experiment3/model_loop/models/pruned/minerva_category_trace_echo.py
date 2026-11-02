"""MINERVA-style echo retrieval from stored number-category traces.

People hold memory traces of familiar number categories (parity, primes,
squares, cubes, powers, times tables, final digits, contained digits,
repdigits, and intervals of the number line). The examples act as a probe:
each trace is activated by a power of its overlap with the examples (Jaccard:
shared members / members in either). People say yes in proportion to how
strongly the activation-weighted echo contains the target. Retrieval sharpness
is a fixed power of overlap, not a size principle growing with the number of
examples.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N = np.arange(1, 101)


def _rule_traces():
    rows = []
    rows.append(N % 2 == 1)
    rows.append(N % 2 == 0)
    rows.append(np.array([n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1)) for n in N]))
    rows.append(np.isin(N, [k * k for k in range(1, 11)]))
    rows.append(np.isin(N, [k ** 3 for k in range(1, 5)]))
    for b in range(2, 11):
        rows.append(np.isin(N, [b ** k for k in range(0, 8) if b ** k <= 100]))
    for m in range(3, 13):
        rows.append(N % m == 0)
    for d in range(10):
        rows.append(N % 10 == d)
    for d in range(10):
        rows.append(np.array([str(d) in str(n) for n in N]))
    rows.append(np.isin(N, [11 * k for k in range(1, 10)]))
    return np.array(rows, dtype=float)


RULES = _rule_traces()
_iv = [(a, b) for a in range(1, 101) for b in range(a, 101)]
INTERVALS = np.array([(N >= a) & (N <= b) for a, b in _iv], dtype=float)
TRACES = np.vstack([RULES, INTERVALS])  # K x 100
IS_RULE = np.r_[np.ones(len(RULES)), np.zeros(len(INTERVALS))]
SIZES = TRACES.sum(1)


def prepare_observed(rows):
    sets = []
    idx = {}
    set_idx = np.zeros(len(rows), dtype="int64")
    tgt = np.zeros(len(rows), dtype="int64")
    y = np.zeros(len(rows), dtype="int64")
    for i, r in enumerate(rows):
        s = r["number_set"]
        if s not in idx:
            idx[s] = len(sets)
            sets.append(s)
        set_idx[i] = idx[s]
        tgt[i] = int(r["target"]) - 1
        y[i] = int(float(r["rating"]))
    log_s = np.zeros((len(sets), TRACES.shape[0]))
    nomatch = np.zeros_like(log_s)
    for u, s in enumerate(sets):
        ex = sorted(set(int(x) for x in s.split(",")))
        probe = np.zeros(100)
        probe[np.array(ex) - 1] = 1.0
        inter = TRACES @ probe
        union = SIZES + len(ex) - inter
        jac = inter / union
        log_s[u] = np.log(np.maximum(jac, 1e-12))
        nomatch[u] = (inter == 0).astype(float)
        log_s[u][inter == 0] = 0.0
    return {
        "log_sim": log_s,
        "no_match": nomatch,
        "set_idx": set_idx,
        "target_idx": tgt,
        "rating": y,
    }


with pm.Model() as model:
    log_sim = pm.Data("log_sim", np.zeros((1, TRACES.shape[0])))
    no_match = pm.Data("no_match", np.zeros((1, TRACES.shape[0])))
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))

    # retrieval power (MINERVA uses 3), strength of category traces vs interval traces
    tau = pm.LogNormal("tau", mu=np.log(3.0), sigma=0.7)
    log_rule_strength = pm.Normal("log_rule_strength", mu=3.0, sigma=3.0)
    lapse = pm.Beta("lapse", alpha=1.0, beta=9.0)
    guess = pm.Beta("guess", alpha=2.0, beta=2.0)

    logw = pt.as_tensor_variable(IS_RULE) * log_rule_strength
    log_act = tau * log_sim + logw[None, :] - 1e4 * no_match
    w = pt.special.softmax(log_act, axis=1)  # U x K
    echo = pt.dot(w, pt.as_tensor_variable(TRACES))  # U x 100
    p_trace = echo[set_idx, target_idx]
    p = lapse * guess + (1.0 - lapse) * p_trace
    p_yes = pm.Deterministic("p_yes", pt.clip(p, 1e-6, 1 - 1e-6))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
