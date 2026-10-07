"""Person-specific attentional lapses added to the heads-rigged second-order motif judgment.

Refinement of `heads_rigged_second_order_side_habit` (described below). Single
change: on a person-specific share of trials lambda_i (logit-normal population,
non-centred, 400 slots; mean prior around 5%) the person does not judge and
picks a side at random, so p_left = lambda_i / 2 + (1 - lambda_i) * p_judged.
Unlike a low graded sensitivity, a lapse caps a person's agreement even on
clear-cut pairs. Addresses the critique that participants' agreement with the
majority varies more across people than the model's person-level sensitivity
produces.

Description of the refined model follows.

Heads-rigged trick-coin suspicion in the second-order motif side-habit judgment.

Refinement of the incumbent `second_order_motif_side_habit` (described below).
Single change, taken from `heads_rigged_coin_suspicion`: the biased-coin
("trick coin") generator no longer has a uniform prior over its heads rate but
a Beta(2 s, 2 (1 - s)) prior with s fitted (s > 0.5: coins rigged towards heads
are suspected more), so a head-heavy sequence is more readily explained as a
rigged coin and looks less random than its tail-heavy mirror. Unlike a face
bias of the chance coin (linear in the heads difference), this acts through the
regular-explanation evidence, so it matters most for lopsided sequences whose
trick-coin explanation competes with the motif and Markov explanations.
Addresses the FDR-surviving critique that people choose the head-heavier
sequence less often than the H/T-symmetric incumbent predicts.

Description of the incumbent follows.

Person-specific side habit added to the second-order chance coin motif-suspicion judgment.

Refinement of `second_order_chance_motif_suspicion`: the single change is that each
person has a habitual leaning to the Left or Right button (non-centred population
with a shared mean, 400 slots), added to the decision variable outside the
length-scaled randomness evidence, so it decides near-ties. Taken from
`person_side_habit_switch_belief`. Addresses the critique that per-person Left
choice rates vary more than the incumbent produces.

Description of the incumbent follows.

Refinement of `person_specific_motif_suspicion` (description of the incumbent follows).

Refinement of `person_specific_chance_switch_belief`: a sequence looks random to
the extent the person's fair coin (with their own believed switch rate) explains
it better than the regular generators (Markov coin, biased coin, repeating motif
with slips; unknowns integrated out), the log evidence difference weighed per
flip by (n / 5) ** -gamma, with person-specific decision sensitivity.

Single change: each person gives the repeating-motif generator their own prior
weight exp(a_i) relative to the Markov and biased coins (weights 1 : 1 : exp(a_i);
non-centred population, 400 slots), so people differ in how readily periodic
sequences such as perfect alternation are explained away as patterns.

Single change made here: people's model of a fair coin is second-order. After a
switch, their believed log-odds of switching again (rather than repeating) shift
by a shared fitted amount d, entering the chance log-likelihood as
(d / 2) * (switch-after-switch count - repeat-after-switch count). With d < 0
people believe chance rarely switches twice in a row, so sustained alternation
looks less random even among already highly alternating sequences.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

PERIODS = (1, 2, 3, 4)


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _summary(seq):
    seq = seq.strip().upper()
    n = len(seq)
    k = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    h = seq.count("H")
    log_markov = math.log(0.5) + _log_beta(k + 1, (n - 1 - k) + 1)
    trans = [1 if x != y else 0 for x, y in zip(seq, seq[1:])]
    ss = sum(1 for u, v in zip(trans, trans[1:]) if u == 1 and v == 1)
    sr = sum(1 for u, v in zip(trans, trans[1:]) if u == 1 and v == 0)
    out = {"k": float(k), "lm": log_markov, "h": float(h), "alt2": float(ss - sr)}
    for p in PERIODS:
        compared = max(n - p, 0)
        mism = sum(1 for i in range(p, n) if seq[i] != seq[i - p])
        out[f"c{p}"] = float(compared)
        out[f"m{p}"] = float(mism)
    out["n"] = float(n)
    return out


def prepare_observed(rows):
    """Table of distinct sequences (regular-generator statistics) plus per-trial indices."""
    index = {}
    table = []
    idx_a, idx_b, switch_diff, alt2_diff, pid, y = [], [], [], [], [], []
    for r in rows:
        ia_ib = []
        for key in ("sequence_a", "sequence_b"):
            seq = str(r[key]).strip().upper()
            if seq not in index:
                index[seq] = len(table)
                table.append(_summary(seq))
            ia_ib.append(index[seq])
        idx_a.append(ia_ib[0])
        idx_b.append(ia_ib[1])
        switch_diff.append(table[ia_ib[0]]["k"] - table[ia_ib[1]]["k"])
        alt2_diff.append(table[ia_ib[0]]["alt2"] - table[ia_ib[1]]["alt2"])
        pid.append(int(r.get("participant_id", 0)))
        y.append(int(r.get("chose_left", 0)))
    out = {
        "idx_a": np.asarray(idx_a, dtype="int64"),
        "idx_b": np.asarray(idx_b, dtype="int64"),
        "switch_diff": np.asarray(switch_diff, dtype="float64"),
        "alt2_diff": np.asarray(alt2_diff, dtype="float64"),
        "participant_id": np.asarray(pid, dtype="int64"),
        "chose_left": np.asarray(y, dtype="int64"),
        "seq_lmark": np.asarray([t["lm"] for t in table], dtype="float64"),
        "seq_heads": np.asarray([t["h"] for t in table], dtype="float64"),
        "seq_len": np.asarray([t["n"] for t in table], dtype="float64"),
    }
    for p in PERIODS:
        out[f"seq_cmp{p}"] = np.asarray([t[f"c{p}"] for t in table], dtype="float64")
        out[f"seq_mis{p}"] = np.asarray([t[f"m{p}"] for t in table], dtype="float64")
    return out


with pm.Model() as model:
    idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
    idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
    switch_diff = pm.Data("switch_diff", np.zeros(1, dtype="float64"))
    alt2_diff = pm.Data("alt2_diff", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    seq_lmark = pm.Data("seq_lmark", np.zeros(1, dtype="float64"))
    seq_heads = pm.Data("seq_heads", np.zeros(1, dtype="float64"))
    seq_len = pm.Data("seq_len", np.full(1, 2.0))
    seq_cmp = {p: pm.Data(f"seq_cmp{p}", np.zeros(1, dtype="float64")) for p in PERIODS}
    seq_mis = {p: pm.Data(f"seq_mis{p}", np.zeros(1, dtype="float64")) for p in PERIODS}


    # Slip rate of the repeating-motif generator.
    logit_eps = pm.Normal("logit_eps", mu=-2.0, sigma=1.0)
    eps = pm.Deterministic("eps", pm.math.sigmoid(logit_eps))
    log_eps = pt.log(eps)
    log_1m_eps = pt.log1p(-eps)

    # Trick coin with a lopsided prior over its heads rate: Beta(2 s, 2 (1 - s)),
    # s > 0.5 means coins rigged towards heads are suspected more.
    logit_s = pm.Normal("logit_s", mu=0.0, sigma=1.0)
    s_heads = pm.Deterministic("s_heads", pm.math.sigmoid(logit_s))
    a_h = 2.0 * s_heads
    a_t = 2.0 - a_h
    seq_lbias = (
        pt.gammaln(seq_heads + a_h) + pt.gammaln(seq_len - seq_heads + a_t)
        - pt.gammaln(seq_len + 2.0)
        - pt.gammaln(a_h) - pt.gammaln(a_t) + math.lgamma(2.0)
    )

    # Log marginal likelihood of each distinct sequence under the regular generators.
    motif_terms = []
    for p in PERIODS:
        free = pt.minimum(float(p), seq_len)
        motif_terms.append(
            -free * math.log(2.0)
            + (seq_cmp[p] - seq_mis[p]) * log_1m_eps
            + seq_mis[p] * log_eps
        )
    log_motif = pm.math.logsumexp(pt.stack(motif_terms, axis=0), axis=0, keepdims=False) - math.log(len(PERIODS))

    # Person-specific prior weight of the motif generator among the regular
    # explanations (weights 1 : 1 : exp(a_i)); the normaliser is the same for
    # both sequences of a pair, so it cancels in the difference.
    mu_a = pm.Normal("mu_a", mu=0.0, sigma=1.0)
    sigma_a = pm.HalfNormal("sigma_a", sigma=1.0)
    z_a = pm.Normal("z_a", mu=0.0, sigma=1.0, shape=400)
    a_motif = (mu_a + sigma_a * z_a)[participant_id]

    def _log_regular(idx):
        return pm.math.logsumexp(
            pt.stack([seq_lmark[idx], seq_lbias[idx], log_motif[idx] + a_motif], axis=0),
            axis=0,
            keepdims=False,
        )

    log_regular_diff = _log_regular(idx_a) - _log_regular(idx_b)

    # Person-specific decision sensitivity (non-centred log-normal population).
    mu_log_beta = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=400)
    beta = pt.exp(mu_log_beta + sigma_log_beta * z_beta)

    # Length normalisation of the evidence (gamma = 0: total evidence, 1: per flip).
    gamma = pm.Normal("gamma", mu=0.5, sigma=0.5)
    trial_len = seq_len[idx_a]
    length_scale = pt.exp(-gamma * pt.log(trial_len / 5.0))

    # Person-specific believed switch probability of a fair coin enters the
    # score as switch_diff * logit_q_i, scaled by that person's sensitivity
    # beta_i. Parameterised directly as the product w_i = beta_i * logit_q_i
    # (non-centred population; w_i < 0: the person believes chance is streaky)
    # to avoid the beta-by-logit_q funnel.
    mu_w = pm.Normal("mu_w", mu=0.0, sigma=1.0)
    sigma_w = pm.HalfNormal("sigma_w", sigma=0.5)
    z_w = pm.Normal("z_w", mu=0.0, sigma=1.0, shape=400)
    w = mu_w + sigma_w * z_w

    # Second-order chance coin: shared shift d in the believed log-odds of
    # switching right after a switch (d < 0: chance rarely switches twice running).
    d_switch2 = pm.Normal("d_switch2", mu=0.0, sigma=1.0)

    pid = participant_id
    log_chance2_diff = 0.5 * d_switch2 * alt2_diff
    eta = (w[pid] * switch_diff + beta[pid] * (log_chance2_diff - log_regular_diff)) * length_scale

    # Person-specific side habit (non-centred population, shared mean bias).
    mu_side = pm.Normal("mu_side", mu=0.0, sigma=0.5)
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=400)
    side = mu_side + sigma_side * z_side

    p_judged = pm.math.sigmoid(eta + side[pid])

    # Person-specific lapse rate: share of trials answered at random.
    mu_lapse = pm.Normal("mu_lapse", mu=-3.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=400)
    lapse = pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse)[pid]

    p_left = pm.Deterministic("p_left", 0.5 * lapse + (1.0 - lapse) * p_judged)

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
