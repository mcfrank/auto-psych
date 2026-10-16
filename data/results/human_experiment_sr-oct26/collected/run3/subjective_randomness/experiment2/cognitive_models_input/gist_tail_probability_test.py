"""Randomness as an intuitive significance test on a sequence's gist.

People summarise each sequence by its gist (head count, number of switches,
longest run) and judge how often a fair coin -- believed to switch with
probability q (fitted, possibly > 1/2) -- would produce a gist at least as
rare as this one. Perceived randomness is the log of that tail probability
(rarity comparisons are fuzzy: a type counts as "as rare or rarer" through a
smooth step on the log-probability scale). The sequence with the higher tail
probability is chosen as more random; people differ in decision sensitivity.
"""

import itertools
import math
from collections import Counter

import numpy as np
import pymc as pm
import pytensor.tensor as pt

TAU = 0.3  # fuzziness of "as rare or rarer", in log-probability units


def _gist(seq):
    k = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    longest = cur = 1
    for x, y in zip(seq, seq[1:]):
        cur = cur + 1 if x == y else 1
        longest = max(longest, cur)
    return (seq.count("H"), k, longest)


def _types_of_length(n):
    """Every gist of length-n sequences with how many sequences share it."""
    c = Counter(_gist("".join(t)) for t in itertools.product("HT", repeat=n))
    return sorted(c.items())


_TYPE_CACHE = {}


def _types(n):
    if n not in _TYPE_CACHE:
        _TYPE_CACHE[n] = _types_of_length(n)
    return _TYPE_CACHE[n]


def prepare_observed(rows):
    """Table of distinct sequences (their gist and all gists of their length) plus per-trial indices."""
    index = {}
    seqs = []
    idx_a, idx_b, pid, y = [], [], [], []
    for r in rows:
        ids = []
        for key in ("sequence_a", "sequence_b"):
            s = str(r[key]).strip().upper()
            if s not in index:
                index[s] = len(seqs)
                seqs.append(s)
            ids.append(index[s])
        idx_a.append(ids[0])
        idx_b.append(ids[1])
        pid.append(int(r.get("participant_id", 0)))
        y.append(int(r.get("chose_left", 0)))
    width = max(len(_types(len(s))) for s in seqs)
    u = len(seqs)
    t_logcnt = np.zeros((u, width))
    t_k = np.zeros((u, width))
    t_mask = np.zeros((u, width))
    own_logcnt = np.zeros(u)
    own_k = np.zeros(u)
    seq_len = np.zeros(u)
    for i, s in enumerate(seqs):
        n = len(s)
        types = _types(n)
        g = _gist(s)
        for j, (t, cnt) in enumerate(types):
            t_logcnt[i, j] = math.log(cnt)
            t_k[i, j] = t[1]
            t_mask[i, j] = 1.0
            if t == g:
                own_logcnt[i] = math.log(cnt)
                own_k[i] = t[1]
        seq_len[i] = n
    return {
        "idx_a": np.asarray(idx_a, dtype="int64"),
        "idx_b": np.asarray(idx_b, dtype="int64"),
        "participant_id": np.asarray(pid, dtype="int64"),
        "chose_left": np.asarray(y, dtype="int64"),
        "t_logcnt": t_logcnt,
        "t_k": t_k,
        "t_mask": t_mask,
        "own_logcnt": own_logcnt,
        "own_k": own_k,
        "seq_len": seq_len,
    }


with pm.Model() as model:
    idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
    idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    t_logcnt = pm.Data("t_logcnt", np.zeros((1, 1)))
    t_k = pm.Data("t_k", np.zeros((1, 1)))
    t_mask = pm.Data("t_mask", np.ones((1, 1)))
    own_logcnt = pm.Data("own_logcnt", np.zeros(1))
    own_k = pm.Data("own_k", np.zeros(1))
    seq_len = pm.Data("seq_len", np.full(1, 2.0))

    # Believed switch probability of a fair coin (logit scale; 0 = normative).
    logit_q = pm.Normal("logit_q", mu=0.0, sigma=1.0)
    log_q = -pt.softplus(-logit_q)
    log_1mq = -pt.softplus(logit_q)

    n_tr = (seq_len - 1.0)[:, None]
    # log P(gist) under the believed chance process (first flip free, so the sum over sequences is 2).
    log_pt = t_logcnt + t_k * log_q + (n_tr - t_k) * log_1mq - math.log(2.0)
    log_own = own_logcnt + own_k * log_q + (seq_len - 1.0 - own_k) * log_1mq - math.log(2.0)
    # Fuzzy "as rare or rarer": smooth step on the log-probability scale.
    log_w = -pt.softplus(-(log_own[:, None] - log_pt) / TAU)
    terms = pt.switch(t_mask > 0.5, log_pt + log_w, -1e10)
    log_tail = pm.math.logsumexp(terms, axis=1, keepdims=False)

    # Person-specific decision sensitivity (non-centred log-normal population).
    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=400)
    beta = pt.exp(mu_log_beta + sigma_log_beta * z_beta)

    score_diff = log_tail[idx_a] - log_tail[idx_b]
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta[participant_id] * score_diff))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
