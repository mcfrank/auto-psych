"""Range-bounded rule learning.

People never entertain an unbounded mathematical rule on its own: every concept
they consider is a rule confined to a numerical range (e.g. "even numbers
between 20 and 60"; the rule "any number" gives plain intervals), and they infer
which range-bounded rule produced the examples under the size principle,
answering yes by the posterior probability that the concept contains the
target. A rule bounded tightly around the examples always beats the same rule
over all of 1-100, so shared properties (parity etc.) are used strongly inside
the examples' span and generalize sharply less outside it.

Hypotheses: rule x interval [a,b] (uniform prior over both), extension = rule
restricted to [a,b]. Since the prior is uniform, the posterior depends only on
the extension size, so per set we precompute histograms of consistent
hypotheses by size (total, and those containing the target). Free parameters:
an evidence exponent gamma on the size-principle likelihood and a lapse.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

N = 100
NUMS = np.arange(1, N + 1)


def _rules():
    r = [np.ones(N, dtype=bool), NUMS % 2 == 1, NUMS % 2 == 0]
    r.append(np.array([n > 1 and all(n % d for d in range(2, int(n ** 0.5) + 1)) for n in NUMS]))
    r.append(np.isin(NUMS, [k * k for k in range(1, 11)]))
    r.append(np.isin(NUMS, [k ** 3 for k in range(1, 5)]))
    for b in range(2, 11):
        r.append(np.isin(NUMS, [b ** k for k in range(0, 8) if b ** k <= N]))
    for m in list(range(3, 10)) + [11, 12]:
        r.append(NUMS % m == 0)
    for d in range(10):
        r.append(NUMS % 10 == d)
    for d in range(10):
        r.append(np.array([str(d) in str(n) for n in NUMS]))
    return np.array(r, dtype=bool)


RULES = _rules()
LOG_SIZES = np.log(np.arange(1, N + 1, dtype=float))  # sizes 1..100
NEG = -1e4


def _set_tables(ex, targets):
    lo, hi = min(ex), max(ex)
    total = np.zeros(N + 1)
    tgt = {t: np.zeros(N + 1) for t in targets}
    a = np.arange(1, lo + 1)  # lower bounds
    b = np.arange(hi, N + 1)  # upper bounds
    for rule in RULES:
        if not all(rule[x - 1] for x in ex):
            continue
        c = np.r_[0, np.cumsum(rule)]
        S = c[b][None, :] - c[a - 1][:, None]  # sizes, all >= 1
        total += np.bincount(S.ravel(), minlength=N + 1)
        for t in targets:
            if not rule[t - 1]:
                continue
            sub = S[: min(t, lo), max(t, hi) - hi:]
            if sub.size:
                tgt[t] += np.bincount(sub.ravel(), minlength=N + 1)
    return total[1:], {t: v[1:] for t, v in tgt.items()}


def _logc(c):
    return np.where(c > 0, np.log(np.maximum(c, 1e-300)), NEG)


def prepare_observed(rows):
    sets, idx, set_i, tg = [], {}, [], []
    for r in rows:
        s = str(r["number_set"]).strip()
        if s not in idx:
            idx[s] = len(sets)
            sets.append([int(x) for x in s.split(",")])
        set_i.append(idx[s])
        tg.append(int(float(r["target"])))
    need = [set() for _ in sets]
    for i, t in zip(set_i, tg):
        need[i].add(t)
    tot_tab, tgt_tab = [], []
    for ex, ts in zip(sets, need):
        tot, tt = _set_tables(ex, sorted(ts))
        tot_tab.append(tot)
        tgt_tab.append(tt)
    log_total = np.array([_logc(v) for v in tot_tab])
    log_in = np.array([_logc(tgt_tab[i][t]) for i, t in zip(set_i, tg)])
    n_ex = np.array([float(len(sets[i])) for i in set_i])
    return {
        "log_total": log_total,
        "log_in": log_in,
        "set_idx": np.array(set_i, dtype="int64"),
        "n_ex": n_ex,
        "rating": np.array([int(float(r["rating"])) for r in rows], dtype="int64"),
    }


with pm.Model() as model:
    log_total = pm.Data("log_total", np.zeros((1, N)))
    log_in = pm.Data("log_in", np.zeros((1, N)))
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))
    n_ex = pm.Data("n_ex", np.ones(1))

    log_gamma = pm.Normal("log_gamma", 0.0, 1.0)  # evidence weight of size principle
    gamma = pm.Deterministic("gamma", pt.exp(log_gamma))
    lapse = pm.Beta("lapse", 1.0, 10.0)

    loglik = -gamma * n_ex[:, None] * LOG_SIZES[None, :]  # trials x sizes
    num = pt.logsumexp(log_in + loglik, axis=1)
    den = pt.logsumexp(log_total[set_idx] + loglik, axis=1)
    p_raw = pt.exp(pt.minimum(num - den, 0.0))
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(lapse * 0.5 + (1 - lapse) * p_raw, 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
