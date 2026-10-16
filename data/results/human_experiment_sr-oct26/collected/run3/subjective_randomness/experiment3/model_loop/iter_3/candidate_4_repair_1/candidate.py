"""Gambler's run-length belief added to the pseudo-flip-diluted heads-rigged motif judgment.

Refinement of `short_sequence_pseudoflip_dilution` (described below). Single
change, taken from `gamblers_run_second_order_side_habit`: in people's model of
a fair coin the believed log-odds of switching rise with the length of the
current run beyond two flips, by a shared fitted amount g per extra flip. Each
transition made after a run of L >= 3 identical flips contributes
(g / 2) * (L - 2) * (+1 if it switches, -1 if it repeats) to the chance
log-likelihood, so a long streak is improbable under chance roughly
quadratically in its length and the first breaks of a long streak are strongly
rewarded. Addresses the FDR-surviving critique that among long pairs with at
most two switches people choose the more-switching sequence more often than
the incumbent predicts. (Geometry only: the trick coin's heads-prior logit gets
a tighter Normal(0, 0.5) prior, ruling out a degenerate tails-rigged mode.)

Description of the refined model follows.

Pseudo-flip dilution of the per-flip evidence weighting for short sequences.

Refinement of `heavy_tailed_signed_sensitivity_motif` (described below). Single
change: the length normalisation of the evidence. The incumbent scales the
evidence by (n / 5) ** -gamma, which with gamma > 0 inflates the evidence of
2- and 3-flip pairs without bound as n shrinks. Here people weigh the evidence
per flip of a padded length, as if a short sequence's few flips were read
alongside c imagined unremarkable flips: scale = ((n + c) / (5 + c)) ** -gamma,
with c > 0 fitted (c -> 0 recovers the incumbent). So very short pairs are
judged less decisively than a pure per-flip scaling implies. Addresses the
critique that in pairs of length 2-3 people choose the more-switching sequence
less often than the incumbent predicts.

Description of the refined model follows.

Heavy-tailed signed person sensitivity on the heads-rigged second-order motif judgment.

Refinement of `attentive_lapse_heads_rigged_motif` (described below). Single
change: the way a person departs from the shared judgment. Instead of a lapse
lambda_i (random guessing, which can pull agreement with the majority only down
to chance) on top of a log-normal sensitivity, each person's sensitivity to the
regular-explanation evidence is drawn from a heavy-tailed population on the
signed scale, beta_i = mu_b + sigma_b * z_i with z_i ~ StudentT(4) (non-centred,
400 slots): most people sit near the group mean, a few are near-indifferent
(beta_i ~ 0, guessing on every pair) and a few reverse (beta_i < 0, picking the
sequence that looks less random to everyone else), so agreement with the
majority can fall below chance. The lapse is removed: a single smooth
per-person scale replaces the lapse/sensitivity pair, which traded off and,
with a signed commitment, gave a bimodal posterior that did not converge.
Addresses the FDR-surviving critique that agreement with the majority varies
across people more than the lapse model produces.

Description of the refined model follows.

Person-specific attentional lapses added to the heads-rigged second-order motif judgment.

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
    gam = 0.0
    run = 1
    for x, y in zip(seq, seq[1:]):
        if run >= 3:
            gam += (run - 2) * (1.0 if x != y else -1.0)
        run = 1 if x != y else run + 1
    out = {"k": float(k), "lm": log_markov, "h": float(h), "alt2": float(ss - sr), "gam": gam}
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
    idx_a, idx_b, switch_diff, alt2_diff, gam_diff, pid, y = [], [], [], [], [], [], []
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
        gam_diff.append(table[ia_ib[0]]["gam"] - table[ia_ib[1]]["gam"])
        pid.append(int(r.get("participant_id", 0)))
        y.append(int(r.get("chose_left", 0)))
    out = {
        "idx_a": np.asarray(idx_a, dtype="int64"),
        "idx_b": np.asarray(idx_b, dtype="int64"),
        "switch_diff": np.asarray(switch_diff, dtype="float64"),
        "alt2_diff": np.asarray(alt2_diff, dtype="float64"),
        "gam_diff": np.asarray(gam_diff, dtype="float64"),
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
    gam_diff = pm.Data("gam_diff", np.zeros(1, dtype="float64"))
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
    # Prior tightened (sigma 0.5): with sigma 1 a degenerate mode at s ~ 0.02
    # (a near-certain tails-rigged coin) trapped a chain; the data put s near 0.5.
    logit_s = pm.Normal("logit_s", mu=0.0, sigma=0.5)
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

    # Person-specific signed sensitivity (non-centred, heavy-tailed population):
    # most people near the group mean, a few near zero or reversed.
    mu_b = pm.Normal("mu_b", mu=1.0, sigma=0.5)
    sigma_b = pm.HalfNormal("sigma_b", sigma=0.5)
    z_beta = pm.StudentT("z_beta", nu=4.0, mu=0.0, sigma=1.0, shape=400)
    beta = mu_b + sigma_b * z_beta

    # Length normalisation of the evidence (gamma = 0: total evidence, 1: per flip).
    gamma = pm.Normal("gamma", mu=0.5, sigma=0.5)
    trial_len = seq_len[idx_a]
    # Pseudo-flip dilution: c imagined unremarkable flips pad the length.
    log_c = pm.Normal("log_c", mu=0.5, sigma=1.0)
    c_pad = pm.Deterministic("c_pad", pt.exp(log_c))
    length_scale = pt.exp(-gamma * pt.log((trial_len + c_pad) / (5.0 + c_pad)))

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
    # Gambler's run-length belief: shared rise g in the believed log-odds of
    # switching per flip the current run lasts beyond two (g > 0: long streaks
    # are improbable under chance beyond their switch count).
    g_run = pm.Normal("g_run", mu=0.0, sigma=0.3)
    log_chance2_diff = 0.5 * d_switch2 * alt2_diff + 0.5 * g_run * gam_diff
    eta = (w[pid] * switch_diff + beta[pid] * (log_chance2_diff - log_regular_diff)) * length_scale

    # Person-specific side habit (non-centred population, shared mean bias).
    mu_side = pm.Normal("mu_side", mu=0.0, sigma=0.5)
    sigma_side = pm.HalfNormal("sigma_side", sigma=0.3)
    z_side = pm.Normal("z_side", mu=0.0, sigma=1.0, shape=400)
    side = mu_side + sigma_side * z_side

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(eta + side[pid]))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
