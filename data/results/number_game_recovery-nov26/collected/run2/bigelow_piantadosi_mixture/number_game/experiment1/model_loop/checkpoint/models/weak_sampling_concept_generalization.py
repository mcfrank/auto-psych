"""Weak-sampling concept generalization.

People entertain mathematical rules and numeric ranges as candidate concepts but
treat the examples only as labelled members of the concept, not as random draws
from it (weak sampling, no size principle): every concept consistent with the
examples stays credible in proportion to its prior, so more examples rule
concepts out but never sharpen generalization toward the narrowest one.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N = 100


def _rules():
    nums = range(1, N + 1)
    rules = [
        {x for x in nums if x % 2 == 1},
        {x for x in nums if x % 2 == 0},
        {x for x in nums if x > 1 and all(x % d for d in range(2, int(x ** 0.5) + 1))},
        {k * k for k in range(1, 11)},
        {k ** 3 for k in range(1, 5)},
    ]
    for b in range(2, 11):
        rules.append({b ** k for k in range(1, 8) if b ** k <= N})
    p2 = {2 ** k for k in range(1, 7)}
    rules.append(p2 - {32})
    rules.append(p2 | {37})
    for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]:
        rules.append({x for x in nums if x % m == 0})
    for d in range(10):
        rules.append({x for x in nums if x % 10 == d})
    for d in range(10):
        rules.append({x for x in nums if str(d) in str(x)})
    return rules


def _hypotheses():
    rules = _rules()
    mem = []
    for r in rules:
        v = np.zeros(N)
        v[[x - 1 for x in r]] = 1.0
        mem.append(v)
    sizes = []
    for lo in range(1, N + 1):
        for hi in range(lo, N + 1):
            v = np.zeros(N)
            v[lo - 1:hi] = 1.0
            mem.append(v)
            sizes.append(hi - lo + 1)
    return np.array(mem), len(rules), np.array(sizes, dtype="float64")


MEM, N_RULES, INT_SIZES = _hypotheses()


def prepare_observed(rows):
    sets = []
    index = {}
    set_idx, tgt = [], []
    for r in rows:
        s = r["number_set"].strip()
        if s not in index:
            index[s] = len(sets)
            sets.append([int(x) for x in s.split(",")])
        set_idx.append(index[s])
        tgt.append(int(r["target"]) - 1)
    # misses[s, h]: number of examples of set s that hypothesis h excludes
    misses = np.zeros((len(sets), MEM.shape[0]))
    for i, ex in enumerate(sets):
        cols = MEM[:, [x - 1 for x in ex]]
        misses[i] = len(ex) - cols.sum(axis=1)
    return {
        "misses": misses,
        "set_idx": np.array(set_idx, dtype="int64"),
        "target_idx": np.array(tgt, dtype="int64"),
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    misses = pm.Data("misses", np.zeros((1, MEM.shape[0]), dtype="float64"))
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    target_idx = pm.Data("target_idx", np.zeros(1, dtype="int64"))

    # Share of prior on rules (rest on intervals), Erlang interval-size scale,
    # probability an example falls outside the concept, response lapse.
    lam = pm.Beta("lam", 2.0, 2.0)
    sigma = pm.LogNormal("sigma", mu=np.log(10.0), sigma=1.0)
    eps = pm.Beta("eps", 1.0, 9.0)
    lapse = pm.Beta("lapse", 1.0, 19.0)

    int_logw = pt.log(INT_SIZES) - INT_SIZES / sigma
    int_logp = int_logw - pt.logsumexp(int_logw)
    log_prior = pt.concatenate([
        pt.zeros(N_RULES) + pt.log(lam) - np.log(N_RULES),
        pt.log1p(-lam) + int_logp,
    ])
    # Weak sampling: consistent examples cost nothing; each exclusion costs log eps.
    log_post = log_prior[None, :] + misses * pt.log(eps)
    post = pt.exp(log_post - pt.logsumexp(log_post, axis=1, keepdims=True))
    gen = pt.dot(post, pt.as_tensor_variable(MEM))  # sets x targets
    p_raw = gen[set_idx, target_idx]
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(lapse * 0.5 + (1 - lapse) * p_raw, 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
